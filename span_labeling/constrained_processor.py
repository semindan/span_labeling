"""
SubstringCopyLogitsProcessor - Constrained Decoding for vLLM

This logits processor ensures that generated text spans are exact substrings
from the input text, and optionally that labels come from an allowed set.

Usage with vLLM server:
    vllm serve Qwen/Qwen2.5-3B-Instruct \
        --port 8000 \
        --logits-processors constrained_decoder:SubstringCopyLogitsProcessor

Usage with OpenAI client:
    client.chat.completions.create(
        model="default",
        messages=[...],
        extra_body={
            "vllm_xargs": {
                "input_text": "John visited Paris",
                "constrained_key": "text",
                "allowed_labels": ["PER", "LOC"],
                "label_key": "label",
            }
        }
    )

The processor automatically extracts the model name from vllm_config and
loads the corresponding tokenizer from HuggingFace.
"""

import re
import sys
from typing import Optional

import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


class SubstringCopyLogitsProcessor(LogitsProcessor):
    """Constrain generation to copy substrings from input text.

    At each decode step, allow any token whose decoded piece keeps the value
    a prefix of at least one substring of the original input sentence.
    Also allow the closing quote to end the value and lift constraints.

    Performance optimizations:
    - Pre-computes all substrings and builds an index
    - Caches valid starting positions for each prefix
    - Avoids repeated string searches and concatenations
    """

    class RequestState:
        """Per-request state for substring matching."""

        def __init__(
            self,
            input_text: str,
            constrained_key: str,
            output_ids: list[int],
            allowed_labels: Optional[list[str]] = None,
            label_key: str = "label",
            max_substring_len: int = 200,
        ):
            self.input_text = input_text
            self.constrained_key = constrained_key
            self.allowed_labels = allowed_labels
            self.label_key = label_key
            self.max_substring_len = max_substring_len

            # Reference to the output token IDs (updated by vLLM engine)
            self.output_ids = output_ids

            # Text field state
            self.prefix_end_pos: Optional[int] = None
            self.value_so_far = ""
            self._last_logged_len = 0
            self.span_index = 0
            self.initial_value_after_quote = ""
            self._last_prefix_match_end_char = 0
            self.value_prefix_re = re.compile(
                rf'"{re.escape(constrained_key)}"\s*:\s*"'
            )

            # Label field state
            self.label_prefix_end_pos: Optional[int] = None
            self.label_value_so_far = ""
            self._last_logged_label_len = 0
            self.initial_label_after_quote = ""
            self._last_label_prefix_match_end_char = 0
            self.label_value_prefix_re = re.compile(
                rf'"{re.escape(label_key)}"\s*:\s*"'
            )

            # Performance optimization: build substring index upfront
            self._substring_index: dict[str, list[int]] = {}
            self._build_substring_index()

            # Cache for valid starting positions (resets each span)
            self._valid_starts: list[int] = []

        def _build_substring_index(self):
            """Pre-compute all substrings and their starting positions."""
            max_len = min(len(self.input_text), self.max_substring_len)

            for start in range(len(self.input_text)):
                for end in range(
                    start + 1, min(start + max_len + 1, len(self.input_text) + 1)
                ):
                    substring = self.input_text[start:end]
                    if substring not in self._substring_index:
                        self._substring_index[substring] = []
                    self._substring_index[substring].append(start)

        def get_valid_starts(self, prefix: str) -> list[int]:
            """Get all starting positions where prefix appears in input_text."""
            if prefix == "":
                return list(range(len(self.input_text)))
            return self._substring_index.get(prefix, [])

    @classmethod
    def validate_params(cls, params: SamplingParams):
        """Validate that required parameters are present in extra_args."""
        if not params.extra_args:
            return

        input_text = params.extra_args.get("input_text")
        constrained_key = params.extra_args.get("constrained_key")

        if input_text is not None and not isinstance(input_text, str):
            raise ValueError(f"input_text must be str, got {type(input_text)}")
        if constrained_key is not None and not isinstance(constrained_key, str):
            raise ValueError(
                f"constrained_key must be str, got {type(constrained_key)}"
            )

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        self.device = device
        self.req_state: dict[int, SubstringCopyLogitsProcessor.RequestState] = {}

        # Extract model name from vllm_config
        model_name = vllm_config.model_config.model

        print(
            f"[LogitsProcessor] Loading tokenizer for {model_name}...", file=sys.stderr
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Build id2piece mapping and identify quote tokens
        self.id2piece: dict[int, str] = {}
        self.quote_token_ids: set[int] = set()

        try:
            vocab_size = tokenizer.vocab_size
        except Exception:
            try:
                vocab_size = len(tokenizer)
            except Exception:
                vocab_size = 50000

        print(
            f"[LogitsProcessor] Building token mappings for {vocab_size} tokens...",
            file=sys.stderr,
        )

        for tid in range(vocab_size):
            try:
                s = tokenizer.decode(
                    [tid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                s = ""

            if s is None:
                s = ""

            self.id2piece[tid] = s

            if s == '"' or s.startswith('"'):
                self.quote_token_ids.add(tid)

        print(
            f"[LogitsProcessor] Found {len(self.quote_token_ids)} quote tokens. Ready!",
            file=sys.stderr,
        )

        # Token prefiltering sets (per-request cache)
        self._potentially_valid_tokens: dict[int, set[int]] = {}
        self._potentially_valid_label_tokens: dict[int, set[int]] = {}

    def is_argmax_invariant(self) -> bool:
        """This processor affects greedy sampling."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        """Handle batch updates for added/removed/moved requests."""
        if not batch_update:
            return

        # Process added requests
        for index, params, _prompt_ids, output_ids in batch_update.added:
            if params is None or not params.extra_args:
                continue

            self.validate_params(params)

            input_text = params.extra_args.get("input_text")
            constrained_key = params.extra_args.get("constrained_key")
            max_substring_len = params.extra_args.get("max_substring_len", 200)
            allowed_labels = params.extra_args.get("allowed_labels")
            label_key = params.extra_args.get("label_key", "label")

            if input_text and constrained_key:
                # Initialize request state with reference to output_ids
                self.req_state[index] = self.RequestState(
                    input_text=input_text,
                    constrained_key=constrained_key,
                    output_ids=output_ids,  # Store reference
                    allowed_labels=allowed_labels,
                    label_key=label_key,
                    max_substring_len=max_substring_len,
                )

                if allowed_labels:
                    print(
                        f"[LogitsProcessor] Request {index} allowed_labels: {allowed_labels}",
                        file=sys.stderr,
                    )

                # Prefilter tokens for this request
                self._prefilter_tokens(index)
                if allowed_labels:
                    self._prefilter_label_tokens(index)

        if self.req_state:
            # Process removed requests
            for index in batch_update.removed:
                self.req_state.pop(index, None)
                self._potentially_valid_tokens.pop(index, None)
                self._potentially_valid_label_tokens.pop(index, None)

            # Process moved requests
            for adx, bdx, direct in batch_update.moved:
                a_state = self.req_state.pop(adx, None)
                b_state = self.req_state.pop(bdx, None)
                a_tokens = self._potentially_valid_tokens.pop(adx, None)
                b_tokens = self._potentially_valid_tokens.pop(bdx, None)
                a_label_tokens = self._potentially_valid_label_tokens.pop(adx, None)
                b_label_tokens = self._potentially_valid_label_tokens.pop(bdx, None)

                if a_state is not None:
                    self.req_state[bdx] = a_state
                    if a_tokens is not None:
                        self._potentially_valid_tokens[bdx] = a_tokens
                    if a_label_tokens is not None:
                        self._potentially_valid_label_tokens[bdx] = a_label_tokens

                if direct == MoveDirectionality.SWAP and b_state is not None:
                    self.req_state[adx] = b_state
                    if b_tokens is not None:
                        self._potentially_valid_tokens[adx] = b_tokens
                    if b_label_tokens is not None:
                        self._potentially_valid_label_tokens[adx] = b_label_tokens

    def _prefilter_tokens(self, req_index: int):
        """Build a set of token IDs that contain characters present in input_text."""
        state = self.req_state[req_index]
        input_chars = set(state.input_text)
        potentially_valid = set()

        for token_id, piece in self.id2piece.items():
            if not piece:
                continue
            # A token is potentially valid if all its characters appear in input
            if set(piece).issubset(input_chars):
                potentially_valid.add(token_id)

        self._potentially_valid_tokens[req_index] = potentially_valid

    def _prefilter_label_tokens(self, req_index: int):
        """Build a set of token IDs that could match allowed labels."""
        state = self.req_state[req_index]
        if not state.allowed_labels:
            self._potentially_valid_label_tokens[req_index] = set()
            return

        # Collect all characters used in any label
        label_chars = set()
        for label in state.allowed_labels:
            label_chars.update(label)

        potentially_valid = set()
        for token_id, piece in self.id2piece.items():
            if not piece:
                continue
            # A token is potentially valid if all its characters appear in some label
            if set(piece).issubset(label_chars):
                potentially_valid.add(token_id)

        self._potentially_valid_label_tokens[req_index] = potentially_valid

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply constraints to logits for all active requests in the batch."""
        if not self.req_state:
            return logits

        # Process each request in the batch
        for req_index, state in self.req_state.items():
            if req_index >= logits.shape[0]:
                continue  # Safety check

            # Apply constraints for this request using its output_ids reference
            logits[req_index] = self._apply_request_constraints(
                req_index, state, logits[req_index]
            )

        return logits

    def _think_block_check(self, decoded_text: str) -> bool:
        """Check if we are currently inside a <think>...</think> block."""
        last_think_start = decoded_text.rfind("<think>")
        if last_think_start != -1:
            last_think_end = decoded_text.rfind("</think>")
            if last_think_end == -1 or last_think_end < last_think_start:
                return True
        return False

    def _apply_label_constraints(
        self,
        req_index: int,
        state: RequestState,
        request_logits: torch.Tensor,
        decoded_text: str,
    ) -> torch.Tensor:
        """Apply label constraints for a single request."""
        output_ids = state.output_ids

        if state.allowed_labels and state.label_prefix_end_pos is None:
            decoded_tail = decoded_text
            last_match = None
            for mm in state.label_value_prefix_re.finditer(decoded_tail):
                last_match = mm
            if (
                last_match
                and last_match.end() > state._last_label_prefix_match_end_char
            ):
                state.label_prefix_end_pos = len(output_ids)
                state.initial_label_after_quote = decoded_tail[last_match.end() :]
                state._last_logged_label_len = 0
                state._last_label_prefix_match_end_char = last_match.end()
                print(
                    f"[LogitsProcessor] Detected label prefix for span #{state.span_index} at position {state.label_prefix_end_pos}; prefilled label chars={len(state.initial_label_after_quote)}",
                    file=sys.stderr,
                )

        # Apply label constraint if active
        if state.label_prefix_end_pos is not None:
            pieces = [
                self.id2piece.get(tid, "")
                for tid in output_ids[state.label_prefix_end_pos :]
            ]
            state.label_value_so_far = state.initial_label_after_quote + "".join(pieces)

            # Log newly generated tokens inside the label value
            current_len = len(output_ids) - state.label_prefix_end_pos
            new_pieces = []
            if current_len > state._last_logged_label_len:
                start = state.label_prefix_end_pos + state._last_logged_label_len
                new_ids = output_ids[start : state.label_prefix_end_pos + current_len]
                for i, tid in enumerate(new_ids, 1):
                    piece = self.id2piece.get(tid, "")
                    new_pieces.append(piece)
                    print(
                        f"[LogitsProcessor] Emitted label token {i + state._last_logged_label_len}: id={tid}, piece={piece!r}",
                        file=sys.stderr,
                    )
                # Check if we've emitted a closing quote (using the pieces we just logged)
                if any('"' in p for p in new_pieces):
                    print(
                        f"[LogitsProcessor] Detected closing quote for label in span #{state.span_index}. Lifting label constraints.",
                        file=sys.stderr,
                    )
                    state.label_prefix_end_pos = None
                    state.label_value_so_far = ""
                    state.initial_label_after_quote = ""
                    state._last_logged_label_len = 0
                    # Return early since we are no longer in label mode
                    return request_logits

                state._last_logged_label_len = current_len

            # Compute allowed next token IDs for labels
            label_allowed_ids: list[int] = []
            vt = state.label_value_so_far

            # Check which tokens could extend the current label prefix
            for token_id in self._potentially_valid_label_tokens[req_index]:
                piece = self.id2piece.get(token_id, "")
                if not piece:
                    continue

                candidate = vt + piece
                # Check if this candidate is a prefix of any allowed label
                for label in state.allowed_labels:
                    if label.startswith(candidate):
                        label_allowed_ids.append(token_id)
                        break

            # Apply label constraints
            # Calculate allowed_ids first
            allowed_ids = set(label_allowed_ids)

            # Only allow closing quote if the current value is a valid complete label
            if state.label_value_so_far in state.allowed_labels:
                allowed_ids.update(self.quote_token_ids)
            if not allowed_ids:
                # Dead end. Force close quote to exit gracefully (even if invalid).
                allowed_ids.update(self.quote_token_ids)

            kept = {tid: request_logits[tid].item() for tid in allowed_ids}
            request_logits[:] = float("-inf")
            for tid, val in kept.items():
                request_logits[tid] = val

        return request_logits

    def _apply_text_constraints(
        self,
        req_index: int,
        state: RequestState,
        request_logits: torch.Tensor,
        decoded_text: Optional[str],
    ) -> torch.Tensor:
        """Apply constraints for a single request.

        This is the core logic ported from the original implementation.
        """
        output_ids = state.output_ids  # This is a reference that updates automatically

        # Handle text field constraint
        if state.prefix_end_pos is None:
            decoded_tail = decoded_text
            last_match = None
            for mm in state.value_prefix_re.finditer(decoded_tail):
                last_match = mm
            if last_match and last_match.end() > state._last_prefix_match_end_char:
                state.prefix_end_pos = len(output_ids)
                state.initial_value_after_quote = decoded_tail[last_match.end() :]
                state._last_logged_len = 0
                state._last_prefix_match_end_char = last_match.end()
                # Reset valid starts for new span
                state._valid_starts = list(range(len(state.input_text)))
                print(
                    f"[LogitsProcessor] Detected value prefix for span #{state.span_index} at position {state.prefix_end_pos}; prefilled value chars={len(state.initial_value_after_quote)}",
                    file=sys.stderr,
                )

        # Apply text constraint if active
        if state.prefix_end_pos is not None:
            pieces = [
                self.id2piece.get(tid, "") for tid in output_ids[state.prefix_end_pos :]
            ]
            state.value_so_far = state.initial_value_after_quote + "".join(pieces)

            # Log newly generated tokens
            current_len = len(output_ids) - state.prefix_end_pos
            if current_len > state._last_logged_len:
                start = state.prefix_end_pos + state._last_logged_len
                new_ids = output_ids[start : state.prefix_end_pos + current_len]
                for i, tid in enumerate(new_ids, 1):
                    piece = self.id2piece.get(tid, "")
                    print(
                        f"[LogitsProcessor] Emitted token {i + state._last_logged_len}: id={tid}, piece={piece!r}",
                        file=sys.stderr,
                    )
                # Check for closing quote
                new_pieces = pieces[state._last_logged_len : current_len]
                if any('"' in p for p in new_pieces):
                    print(
                        f"[LogitsProcessor] Detected closing quote for span #{state.span_index}. Lifting constraints.",
                        file=sys.stderr,
                    )
                    state.prefix_end_pos = None
                    state.value_so_far = ""
                    state.initial_value_after_quote = ""
                    state._last_logged_len = 0
                    state._valid_starts = []
                    state.span_index += 1
                    return request_logits
                state._last_logged_len = current_len

            # Compute allowed next token IDs
            prefix_allowed_ids: list[int] = []
            vt = state.value_so_far

            # Get valid starting positions using pre-computed index
            if vt == "":
                starts = state._valid_starts
            else:
                starts = state.get_valid_starts(vt)

            # Optimization: collect valid next characters
            next_chars = set()
            for s in starts:
                end_pos = s + len(vt)
                if end_pos < len(state.input_text):
                    next_chars.add(state.input_text[end_pos : end_pos + 1])

            # Check tokens that could extend the prefix
            for token_id in self._potentially_valid_tokens[req_index]:
                piece = self.id2piece.get(token_id, "")
                if not piece:
                    continue

                # Quick rejection: if token doesn't start with any valid next char
                if next_chars and piece[0] not in next_chars:
                    continue

                # Check if this token extends any valid start position
                candidate = vt + piece
                if candidate in state._substring_index:
                    # Verify at least one start position matches
                    for cand_start in state._substring_index[candidate]:
                        if cand_start in starts:
                            prefix_allowed_ids.append(token_id)
                            break

            # Apply constraints
            allowed_ids = set(prefix_allowed_ids)
            allowed_ids.update(self.quote_token_ids)
            kept = {tid: request_logits[tid].item() for tid in allowed_ids}
            request_logits[:] = float("-inf")
            for tid, val in kept.items():
                request_logits[tid] = val

        return request_logits

    def _apply_request_constraints(
        self,
        req_index: int,
        state: RequestState,
        request_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply constraints for a single request.

        This is the core logic ported from the original implementation.
        """
        output_ids = state.output_ids
        # Decode text once for checking tags and prefix searching
        decoded_text = "".join(self.id2piece.get(tid, "") for tid in output_ids)

        if self._think_block_check(decoded_text):
            return request_logits

        request_logits = self._apply_label_constraints(
            req_index, state, request_logits, decoded_text
        )
        if state.label_prefix_end_pos is not None:
            return request_logits
        request_logits = self._apply_text_constraints(
            req_index, state, request_logits, decoded_text
        )

        return request_logits
