# SPDX-License-Identifier: Apache-2.0
"""
Simple HTTP server exposing a JSON API to detect grammatical-error spans.

Endpoint:
  POST /detect_spans
  Body: {"prompt": "...", "input_text": "...", "dataset": "error"}
  Response: {"input_text": "...", "spans": [{"text": "..."}, ...], "raw": "<model_json>"}

This server uses vLLM with a custom logits processor that constrains each
returned span (the JSON field "text") to be an exact substring of the input.

The server acts like Ollama: it accepts a pre-formatted prompt from the client
and applies constrained decoding. It does NOT build prompts itself.

No external web framework is required; it uses Python's built-in HTTP server.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

from span_labeling.prompt_utils import build_json_schema


class SubstringCopyLogitsProcessor:
    """Constrain generation only while producing the given JSON field value.

    At each decode step, allow any token whose decoded piece keeps the value
    a prefix of at least one substring of the original input sentence.
    Also allow the closing quote to end the value and lift constraints.

    Performance optimizations:
    - Pre-computes all substrings and builds an index
    - Caches valid starting positions for each prefix
    - Avoids repeated string searches and concatenations
    """

    def __init__(
        self,
        input_text: str,
        id2piece: dict[int, str],
        constrained_key: str,
        quote_token_ids: Optional[list[int]] = None,
        max_substring_len: int = 200,
        allowed_labels: Optional[list[str]] = None,
        label_key: str = "label",
    ) -> None:
        self.input_text = input_text
        self.id2piece = id2piece
        self.prefix_end_pos: Optional[int] = None
        self.value_so_far = ""
        self._last_logged_len = 0
        self.span_index = 0
        # Tokens that include a '"' anywhere — allow them during constrained steps
        self.quote_token_ids = set(quote_token_ids or [])
        # Regex to detect the constrained JSON key value opening, allowing whitespace around colon
        self.value_prefix_re = re.compile(rf'"{re.escape(constrained_key)}"\s*:\s*"')
        # Holds already-generated characters after the opening quote if the model
        # produced part of the span in the same token as the opening quote
        self.initial_value_after_quote = ""
        # Track the last character offset of a detected prefix to avoid re-triggering
        self._last_prefix_match_end_char = 0

        # Configuration for memory/speed tradeoff
        self.max_substring_len = max_substring_len

        # Performance optimization: build substring index upfront
        # Maps each substring to its starting positions in input_text
        self._substring_index: dict[str, list[int]] = {}
        self._build_substring_index()

        # Cache for valid starting positions (resets each span)
        self._valid_starts: list[int] = []

        # Pre-filter tokens by whether they could ever match input_text
        self._prefilter_tokens()

        # Label constraint support
        self.allowed_labels = allowed_labels
        self.label_key = label_key
        self.label_prefix_end_pos: Optional[int] = None
        self.label_value_so_far = ""
        self._last_logged_label_len = 0
        self.label_value_prefix_re = re.compile(rf'"{re.escape(label_key)}"\s*:\s*"')
        self.initial_label_after_quote = ""
        self._last_label_prefix_match_end_char = 0
        # Pre-filter tokens for label matching if labels are provided
        if self.allowed_labels:
            self._prefilter_label_tokens()

    def _build_substring_index(self):
        """Pre-compute all substrings and their starting positions.

        For long texts, this uses O(n^2) memory but makes lookups O(1).
        For very long texts (>10k chars), we limit max substring length.
        """
        max_len = min(len(self.input_text), self.max_substring_len)

        for start in range(len(self.input_text)):
            for end in range(
                start + 1, min(start + max_len + 1, len(self.input_text) + 1)
            ):
                substring = self.input_text[start:end]
                if substring not in self._substring_index:
                    self._substring_index[substring] = []
                self._substring_index[substring].append(start)

    def _prefilter_tokens(self):
        """Build a set of token IDs that contain characters present in input_text.

        This allows quick rejection of tokens that can never match.
        """
        input_chars = set(self.input_text)
        self._potentially_valid_tokens = set()

        for token_id, piece in self.id2piece.items():
            if not piece:
                continue
            # A token is potentially valid if all its characters appear in input
            if set(piece).issubset(input_chars):
                self._potentially_valid_tokens.add(token_id)

    def _prefilter_label_tokens(self):
        """Build a set of token IDs that could match allowed labels.

        This allows quick rejection of tokens that can never match any label.
        """
        if not self.allowed_labels:
            self._potentially_valid_label_tokens = set()
            return

        # Collect all characters used in any label
        label_chars = set()
        for label in self.allowed_labels:
            label_chars.update(label)

        self._potentially_valid_label_tokens = set()

        for token_id, piece in self.id2piece.items():
            if not piece:
                continue
            # A token is potentially valid if all its characters appear in some label
            if set(piece).issubset(label_chars):
                self._potentially_valid_label_tokens.add(token_id)

    def _get_valid_starts(self, prefix: str) -> list[int]:
        """Get all starting positions where prefix appears in input_text.

        Uses pre-computed index for O(1) lookup instead of O(n) search.
        """
        if prefix == "":
            return list(range(len(self.input_text)))
        return self._substring_index.get(prefix, [])

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Handle text field constraint
        if self.prefix_end_pos is None:
            # Use regex detection to find the constrained JSON key value opening
            decoded_tail = "".join(self.id2piece.get(tid, "") for tid in output_ids)
            last_match = None
            for mm in self.value_prefix_re.finditer(decoded_tail):
                last_match = mm
            if last_match and last_match.end() > self._last_prefix_match_end_char:
                self.prefix_end_pos = len(output_ids)
                # Characters after the opening quote already present in decoded_tail
                self.initial_value_after_quote = decoded_tail[last_match.end() :]
                self._last_logged_len = 0
                self._last_prefix_match_end_char = last_match.end()
                # Reset valid starts for new span
                self._valid_starts = list(range(len(self.input_text)))
                print(
                    f"[LogitsProcessor] Detected value prefix for span #{self.span_index} at position {self.prefix_end_pos}; prefilled value chars={len(self.initial_value_after_quote)}",
                    file=sys.stderr,
                )

        # Handle label field constraint
        if self.allowed_labels and self.label_prefix_end_pos is None:
            decoded_tail = "".join(self.id2piece.get(tid, "") for tid in output_ids)
            last_match = None
            for mm in self.label_value_prefix_re.finditer(decoded_tail):
                last_match = mm
            if last_match and last_match.end() > self._last_label_prefix_match_end_char:
                self.label_prefix_end_pos = len(output_ids)
                self.initial_label_after_quote = decoded_tail[last_match.end() :]
                self._last_logged_label_len = 0
                self._last_label_prefix_match_end_char = last_match.end()
                print(
                    f"[LogitsProcessor] Detected label prefix for span #{self.span_index} at position {self.label_prefix_end_pos}; prefilled label chars={len(self.initial_label_after_quote)}",
                    file=sys.stderr,
                )

        # Apply label constraint if active
        if self.label_prefix_end_pos is not None:
            pieces = [
                self.id2piece.get(tid, "") for tid in output_ids[self.label_prefix_end_pos :]
            ]
            self.label_value_so_far = self.initial_label_after_quote + "".join(pieces)

            # Log newly generated tokens inside the label value
            current_len = len(output_ids) - self.label_prefix_end_pos
            if current_len > self._last_logged_label_len:
                start = self.label_prefix_end_pos + self._last_logged_label_len
                new_ids = output_ids[start : self.label_prefix_end_pos + current_len]
                for i, tid in enumerate(new_ids, 1):
                    piece = self.id2piece.get(tid, "")
                    print(
                        f"[LogitsProcessor] Emitted label token {i + self._last_logged_label_len}: id={tid}, piece={piece!r}",
                        file=sys.stderr,
                    )
                self._last_logged_label_len = current_len

            # Compute allowed next token IDs for labels
            label_allowed_ids: list[int] = []
            vt = self.label_value_so_far

            # Check which tokens could extend the current label prefix to match an allowed label
            for token_id in self._potentially_valid_label_tokens:
                piece = self.id2piece.get(token_id, "")
                if not piece:
                    continue

                candidate = vt + piece
                # Check if this candidate is a prefix of any allowed label
                for label in self.allowed_labels:
                    if label.startswith(candidate):
                        label_allowed_ids.append(token_id)
                        break

            # Apply label constraints
            if label_allowed_ids:
                allowed_ids = set(label_allowed_ids)
                allowed_ids.update(self.quote_token_ids)
                kept = {tid: logits[tid].item() for tid in allowed_ids}
                logits[:] = float("-inf")
                for tid, val in kept.items():
                    logits[tid] = val

            # Check if the most recently emitted token contains a closing quote
            if current_len > self._last_logged_label_len:
                new_pieces = pieces[self._last_logged_label_len :]
                if any('"' in p for p in new_pieces):
                    print(
                        f"[LogitsProcessor] Detected closing quote for label in span #{self.span_index}. Lifting label constraints.",
                        file=sys.stderr,
                    )
                    self.label_prefix_end_pos = None
                    self.label_value_so_far = ""
                    self.initial_label_after_quote = ""
                    self._last_logged_label_len = 0

            return logits

        # Apply text constraint if active
        if self.prefix_end_pos is not None:
            pieces = [
                self.id2piece.get(tid, "") for tid in output_ids[self.prefix_end_pos :]
            ]
            # Reconstruct the current value including any chars produced alongside the opening quote
            self.value_so_far = self.initial_value_after_quote + "".join(pieces)

            # Log newly generated tokens inside the value
            current_len = len(output_ids) - self.prefix_end_pos
            if current_len > self._last_logged_len:
                start = self.prefix_end_pos + self._last_logged_len
                new_ids = output_ids[start : self.prefix_end_pos + current_len]
                for i, tid in enumerate(new_ids, 1):
                    piece = self.id2piece.get(tid, "")
                    print(
                        f"[LogitsProcessor] Emitted token {i + self._last_logged_len}: id={tid}, piece={piece!r}",
                        file=sys.stderr,
                    )
                self._last_logged_len = current_len

            # Compute allowed next token IDs
            prefix_allowed_ids: list[int] = []
            vt = self.value_so_far

            # Get valid starting positions using pre-computed index
            if vt == "":
                starts = self._valid_starts
            else:
                starts = self._get_valid_starts(vt)

            # Early exit if no valid continuations exist
            if not starts:
                # No valid prefixes, but don't mask to avoid FSM dead-ends
                return logits

            # Optimization: for each valid start position, pre-compute what the next
            # character(s) could be, then filter tokens by that
            next_chars = set()
            for s in starts:
                end_pos = s + len(vt)
                if end_pos < len(self.input_text):
                    # Collect next few characters for faster filtering
                    next_chars.add(self.input_text[end_pos : end_pos + 1])

            # Only check tokens that could start with one of the valid next characters
            for token_id in self._potentially_valid_tokens:
                piece = self.id2piece.get(token_id, "")
                if not piece:
                    continue

                # Quick rejection: if token doesn't start with any valid next char, skip
                if next_chars and piece[0] not in next_chars:
                    continue

                # Check if this token extends any valid start position
                candidate = vt + piece
                # Use index lookup instead of startswith check on full string
                if candidate in self._substring_index:
                    # Verify at least one start position matches
                    for cand_start in self._substring_index[candidate]:
                        if cand_start in starts:
                            prefix_allowed_ids.append(token_id)
                            break

            # If we have prefix-viable tokens, allow them and also allow quote tokens as closers.
            # If none are prefix-viable, avoid masking entirely to prevent grammar FSM dead-ends.
            if prefix_allowed_ids:
                allowed_ids = set(prefix_allowed_ids)
                allowed_ids.update(self.quote_token_ids)
                kept = {tid: logits[tid].item() for tid in allowed_ids}
                logits[:] = float("-inf")
                for tid, val in kept.items():
                    logits[tid] = val

            # Update valid starts for next iteration (incremental filtering)
            # Only keep starts that are still valid with current prefix
            if starts and vt:
                self._valid_starts = starts

            # Check if the most recently emitted token contains a closing quote
            # Only check new tokens since last call (not all pieces from the beginning)
            if current_len > self._last_logged_len:
                new_pieces = pieces[self._last_logged_len :]
                if any('"' in p for p in new_pieces):
                    print(
                        f"[LogitsProcessor] Detected closing quote (any-quote token) for span #{self.span_index}. Lifting constraints.",
                        file=sys.stderr,
                    )
                    self.prefix_end_pos = None
                    self.value_so_far = ""
                    self.initial_value_after_quote = ""
                    self._last_logged_len = 0
                    self._valid_starts = []
                    self.span_index += 1

        return logits


class SubstringCopyLogitsProcessorAdapter(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)

    @classmethod
    def validate_params(cls, params: SamplingParams):
        return None

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        input_text: Optional[str] = (
            params.extra_args.get("input_text") if params.extra_args else None
        )
        id2piece: Optional[dict[int, str]] = (
            params.extra_args.get("id2piece") if params.extra_args else None
        )
        quote_token_ids: Optional[list[int]] = (
            params.extra_args.get("quote_token_ids") if params.extra_args else None
        )
        constrained_key: Optional[str] = (
            params.extra_args.get("constrained_key") if params.extra_args else None
        )
        max_substring_len: int = (
            params.extra_args.get("max_substring_len", 200)
            if params.extra_args
            else 200
        )
        allowed_labels: Optional[list[str]] = (
            params.extra_args.get("allowed_labels") if params.extra_args else None
        )
        label_key: str = (
            params.extra_args.get("label_key", "label")
            if params.extra_args
            else "label"
        )
        if not input_text or not id2piece or not constrained_key:
            return None
        return SubstringCopyLogitsProcessor(
            input_text=input_text,
            id2piece=id2piece,
            constrained_key=constrained_key,
            quote_token_ids=quote_token_ids,
            max_substring_len=max_substring_len,
            allowed_labels=allowed_labels,
            label_key=label_key,
        )


class ModelState:
    """Encapsulates vLLM model and tokenizer state."""

    def __init__(self, mode: str, model: str):
        logits = [SubstringCopyLogitsProcessorAdapter] if mode == "constrained" else []
        self.llm = LLM(
            model=model,
            max_model_len=8192,
            logits_processors=logits,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.id2piece: dict[int, str] = {}
        self.quote_token_ids: set[int] = set()
        self._build_token_maps()

    def _build_token_maps(self):
        """Build id->piece map and identify quote-containing tokens."""
        try:
            vocab_size = self.tokenizer.vocab_size
        except Exception:
            try:
                vocab_size = len(self.tokenizer)
            except Exception:
                vocab_size = 0

        for tid in range(vocab_size):
            try:
                s = self.tokenizer.decode(
                    [tid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                s = ""
            if s is None:
                s = ""
            self.id2piece[tid] = s
            if '"' in s:
                self.quote_token_ids.add(tid)

    def detect_spans(
        self,
        input_text: str,
        prompt: str,
        mode: str,
        constrained_key: str,
        dataset: str,
        system_message: str,
        sampling_params_dict: dict,
        allowed_labels: Optional[list[str]] = None,
        label_key: str = "label",
    ) -> dict:
        """Run inference with constrained decoding on the input prompt."""
        # Use the pre-formatted prompt from the client (already built via build_prompt)
        # The server acts like Ollama: accepts a prompt, applies constrained decoding, returns JSON

        # Structured JSON schema: build dynamically from the prompt config's "format"
        # so the server works for all datasets (ner, error, multigec, wmt, synthetic).
        json_schema = build_json_schema("json", dataset)
        structured_outputs = None
        if mode != "unconstrained":
            structured_outputs = StructuredOutputsParams(json=json_schema)

        # Use sampling parameters provided by client
        sp_kwargs = sampling_params_dict.copy()
        if structured_outputs is not None:
            sp_kwargs["structured_outputs"] = structured_outputs
        if mode == "constrained":
            sp_kwargs["extra_args"] = {
                "input_text": input_text,
                "id2piece": self.id2piece,
                "quote_token_ids": list(self.quote_token_ids),
                "constrained_key": constrained_key,
                "allowed_labels": allowed_labels,
                "label_key": label_key,
            }
        sampling_params = SamplingParams(**sp_kwargs)

        # Use system message provided by client
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        outputs = self.llm.chat(messages=[messages], sampling_params=[sampling_params])
        raw = outputs[0].outputs[0].text

        try:
            spans = json.loads(raw)
            if not isinstance(spans, list):
                spans = []
        except Exception:
            spans = []

        return {"input_text": input_text, "spans": spans, "raw": raw}


class Handler(BaseHTTPRequestHandler):
    # Class attributes for model state and mode (set once on server startup)
    model_state: ModelState = None  # type: ignore
    server_mode: str = "constrained"

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/detect_spans":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
            prompt = payload["prompt"]
            input_text = payload["input_text"]
            dataset = payload["dataset"]
            system_message = payload.get(
                "system_message", "You are a helpful assistant."
            )
            sampling_params_dict = payload.get("sampling_params", {})
            constrained_key = (
                payload.get("constrained_key")
                if self.server_mode == "constrained"
                else None
            )
            allowed_labels = payload.get("allowed_labels")
            label_key = payload.get("label_key", "label")

            result = self.model_state.detect_spans(
                input_text=input_text,
                prompt=prompt,
                mode=self.server_mode,
                dataset=dataset,
                constrained_key=constrained_key,
                system_message=system_message,
                sampling_params_dict=sampling_params_dict,
                allowed_labels=allowed_labels,
                label_key=label_key,
            )
            data = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)

            err = {"error": str(e)}
            data = json.dumps(err).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def run(host: str, port: int, mode: str, model: str):
    # Initialize model state once at server startup
    model_state = ModelState(mode=mode, model=model)
    # Bind the model state and mode to the handler class
    Handler.model_state = model_state
    Handler.server_mode = mode
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[Server] Listening on http://{host}:{port} (mode={mode}, model={model})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error span vLLM server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=5454, help="Port to listen on")
    parser.add_argument(
        "--mode",
        choices=["constrained", "json", "unconstrained"],
        default="constrained",
        help="Decoding mode: constrained (structured JSON + logits processor), json (structured JSON only), unconstrained (no constraints)",
    )
    parser.add_argument(
        "--model",
        default="microsoft/Phi-4-mini-instruct",
        help="Model name or path to load",
    )
    args = parser.parse_args()
    run(host=args.host, port=args.port, mode=args.mode, model=args.model)
