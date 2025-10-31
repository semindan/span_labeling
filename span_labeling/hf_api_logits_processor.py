from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from transformers.generation.logits_process import LogitsProcessor


def _decode_token_piece(tokenizer, token_id: int) -> str:
    """
    Decode a single token id to its string piece without cleaning or special token skipping.
    Works for BPE/SPM alike.
    """
    # Using decode on a single id is the most robust across fast/slow tokenizers.
    return tokenizer.decode(
        [int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def parse_xml_format(format_string: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse the XML format string to extract tag name and attribute info.

    Returns:
            (tag_name, attribute_name, attribute_placeholder)
    """
    import re

    match = re.match(
        r'<(\w+)(?:\s+(\w+)="([^"]+)")?>.*?</\1>', format_string, flags=re.DOTALL
    )
    if match:
        tag_name = match.group(1)
        attr_name = match.group(2) if match.group(2) else None
        attr_placeholder = match.group(3) if match.group(3) else None
        return tag_name, attr_name, attr_placeholder
    raise ValueError(f"Could not parse XML format: {format_string}")


def build_allowed_tags(prompt_config: dict) -> Tuple[List[str], str]:
    """
    From a prompt configuration (yaml), build the list of allowed opening tag strings
    and the single closing tag string.

    Example outputs:
      - openings: ['<entity type="PERSON">', '<entity type="ORG">', '<entity type="LOC">']
      - closing:  '</entity>'
    """
    fmt = prompt_config.get("format", "")
    tag_name, attr_name, attr_placeholder = parse_xml_format(fmt)

    openings: List[str]
    if attr_name and attr_placeholder:
        if "labels" in prompt_config:
            labels = prompt_config["labels"]
            openings = [f'<{tag_name} {attr_name}="{lab}">' for lab in labels]
        elif "label_dict" in prompt_config:
            # Use keys as labels
            labels = list(prompt_config["label_dict"].keys())
            openings = [f'<{tag_name} {attr_name}="{str(k)}">' for k in labels]
        else:
            # Allow any attribute value — but we need a finite set for constrained decoding.
            # Fall back to allowing only the tag name without constraining attribute values.
            # In practice, the yaml always contains either labels or label_dict in this repo.
            openings = [
                f'<{tag_name} {attr_name}="'
            ]  # Will be treated as prefixes only
    else:
        openings = [f"<{tag_name}>"]

    closing = f"</{tag_name}>"
    return openings, closing


@dataclass
class _State:
    # COPY or TAG
    mode: str = "COPY"
    # True if between an opening and its matching closing tag
    inside_tag: bool = False
    # Index into the tokenized input text (token ids) indicating how much was copied
    input_tok_pos: int = 0
    # Tag building buffer and candidates (used in TAG mode)
    tag_buffer: str = ""
    candidate_tags: Optional[List[str]] = None
    # Whether we're consuming the first token of a tag (the one that contains '<')
    first_tag_token: bool = False
    # Track how many tokens we have seen so far to update state incrementally
    seen_len: int = 0
    # If we captured the full remaining input with a single token (plus suffix), request stop
    finish_after_cut: bool = False
    # Number of trailing characters from the last token piece to cut off the final output
    cut_suffix_chars: int = 0
    # True immediately after finishing an opening tag; used to handle leading spaces correctly
    just_opened_tag: bool = False


class XmlConstrainedLogitsProcessor(LogitsProcessor):
    """
    COPY/TAG constrained decoding using HuggingFace LogitsProcessor.

    Rules:
    - COPY mode: allow either the next token from the input text OR a token that starts a valid tag.
      If inside_tag is True, the only tag allowed to start is the closing tag.
    - TAG mode: restrict to continue one of the candidate tag strings until it's fully emitted,
      then switch back to COPY, toggling inside_tag accordingly.

    Notes:
    - Batch size 1 only.
    - Greedy decoding recommended (do_sample=False) for determinism.
    - Assumes the model won't emit '<' or '>' from the prompt itself beyond our constraints.
    """

    def __init__(
        self,
        tokenizer,
        input_text_token_ids: Sequence[int],
        allowed_opening_tags: List[str],
        closing_tag: str,
        eos_token_id: Optional[int] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_token_ids = list(int(t) for t in input_text_token_ids)
        self.allowed_opening_tags = allowed_opening_tags
        self.closing_tag = closing_tag
        self.state = _State()
        self.debug = debug
        self.step_counter = 0
        self.prompt_length = 0  # Add this line

        # Cache token pieces for the full vocab
        self.vocab_size = self.tokenizer.vocab_size
        # Some tokenizers don't expose vocab_size; fall back to model_max_length sized inference is unsafe.
        if self.vocab_size is None:
            # Best effort: try len(tokenizer)
            self.vocab_size = len(self.tokenizer)

        # Note: We'll extend this list dynamically if we encounter token IDs beyond vocab_size
        # (which can happen with additional special tokens)
        self._token_pieces: List[str] = [
            _decode_token_piece(self.tokenizer, i) for i in range(self.vocab_size)
        ]
        self._max_cached_id = self.vocab_size - 1

        if self.debug:
            print(f"[DEBUG] Initializing XmlConstrainedLogitsProcessor")
            print(f"[DEBUG] Full vocab size: {self.vocab_size}")
            print(f"[DEBUG] Input text tokens: {len(self.input_token_ids)} tokens")
            print(
                f"[DEBUG] Input text: {self.tokenizer.decode(self.input_token_ids, skip_special_tokens=False)}"
            )
            print(f"[DEBUG] Allowed opening tags: {self.allowed_opening_tags}")
            print(f"[DEBUG] Closing tag: {self.closing_tag}")

        # Precompute starter token ids for opening and closing tags
        self._opening_starter_ids = self._compute_starter_ids(
            self.allowed_opening_tags, require_slash=False
        )
        self._closing_starter_ids = self._compute_starter_ids(
            [self.closing_tag], require_slash=True
        )

        if self.debug:
            print(
                f"[DEBUG] Opening tag starter ids ({len(self._opening_starter_ids)}): {self._opening_starter_ids[:10]}{'...' if len(self._opening_starter_ids) > 10 else ''}"
            )
            print(
                f"[DEBUG] Closing tag starter ids ({len(self._closing_starter_ids)}): {self._closing_starter_ids[:10]}{'...' if len(self._closing_starter_ids) > 10 else ''}"
            )

            # Show examples of starter tokens
            if self._opening_starter_ids:
                examples = [
                    (tid, repr(self._get_token_piece(tid)))
                    for tid in self._opening_starter_ids[:5]
                ]
                print(f"[DEBUG] Opening starter examples: {examples}")
            if self._closing_starter_ids:
                examples = [
                    (tid, repr(self._get_token_piece(tid)))
                    for tid in self._closing_starter_ids[:5]
                ]
                print(f"[DEBUG] Closing starter examples: {examples}")

        # EOS handling
        self.eos_token_id = eos_token_id
        if self.debug:
            print(f"[DEBUG] EOS token id: {self.eos_token_id}")

    # -----------------------------
    # Small utilities and debug IO
    # -----------------------------
    def _dbg(self, *lines: str):
        if not self.debug:
            return
        for ln in lines:
            print(ln)

    def _is_eos(self, tid: int) -> bool:
        return self.eos_token_id is not None and int(tid) == int(self.eos_token_id)

    def _get_token_piece(self, token_id: int) -> str:
        """
        Get the token piece for a given token id, extending the cache if necessary.
        """
        token_id = int(token_id)
        if token_id <= self._max_cached_id:
            return self._token_pieces[token_id]

        # Need to extend cache
        if self.debug and token_id > self._max_cached_id + 100:
            print(
                f"[DEBUG] Extending token cache from {self._max_cached_id} to {token_id}"
            )

        # Extend the cache up to the requested token id
        for tid in range(self._max_cached_id + 1, token_id + 1):
            self._token_pieces.append(_decode_token_piece(self.tokenizer, tid))
        self._max_cached_id = token_id

        return self._token_pieces[token_id]

    def _compute_starter_ids(
        self,
        tag_list: List[str],
        *,
        require_slash: bool = False,
    ) -> List[int]:
        """
        Compute token ids that can start a tag.

        - require_slash: when True (for closing tags), require the token piece to
          start specifically with '</' rather than just '<'. When False (for opening
          tags), allow any token starting with '<'.
        """
        starters: List[int] = []
        # Use current cache length to include any dynamically added special tokens
        for tid in range(len(self._token_pieces)):
            piece = self._get_token_piece(tid)
            if not piece:
                continue
            idx = piece.find("<")
            if idx < 0:
                continue
            suffix = piece[idx:]
            if require_slash and not suffix.startswith("</"):
                # For closing tags we must begin with '</'
                continue
            for tag in tag_list:
                if tag.startswith(suffix):
                    starters.append(tid)
                    break
        return starters

    def _enter_tag_mode(self, opening: bool, first_piece: Optional[str] = None):
        if self.debug:
            print(
                f"[DEBUG] Entering TAG mode: opening={opening}, first_piece={repr(first_piece)}"
            )
        self.state.mode = "TAG"
        self.state.first_tag_token = True
        self.state.tag_buffer = ""
        self.state.candidate_tags = (
            list(self.allowed_opening_tags) if opening else [self.closing_tag]
        )
        if self.debug:
            print(f"[DEBUG] Initial candidate tags: {self.state.candidate_tags}")
        if first_piece is not None:
            # Consume immediately the content from '<' within this first piece
            idx = first_piece.find("<")
            if idx >= 0:
                self.state.tag_buffer += first_piece[idx:]
                self.state.first_tag_token = False
                # Filter candidates now
                self.state.candidate_tags = [
                    t
                    for t in self.state.candidate_tags
                    if t.startswith(self.state.tag_buffer)
                ]
                if self.debug:
                    print(
                        f"[DEBUG] After first piece, tag_buffer={repr(self.state.tag_buffer)}, candidates={self.state.candidate_tags}"
                    )

    # -----------------------------
    # State update helpers
    # -----------------------------
    def _advance_input_pointer_with_prefix(self, prefix: str):
        if not prefix:
            return
        if self.debug:
            print(
                f"[DEBUG] COPY mode: prefix before '<' = {repr(prefix)}; attempting to advance input pointer"
            )
        consumed = ""
        while self.state.input_tok_pos < len(self.input_token_ids):
            next_id = self.input_token_ids[self.state.input_tok_pos]
            next_piece = self._get_token_piece(next_id)
            tentative = consumed + next_piece
            if prefix.startswith(tentative):
                consumed = tentative
                self.state.input_tok_pos += 1
                if self.debug:
                    print(
                        f"[DEBUG]   consumed next input token id={next_id}, piece={repr(next_piece)}; consumed now={repr(consumed)}; input_tok_pos={self.state.input_tok_pos}"
                    )
                if consumed == prefix:
                    break
            else:
                break
        if consumed != prefix and self.debug:
            print(
                f"[DEBUG]   WARNING: prefix not fully matched by input tokens (consumed={repr(consumed)})"
            )

    def _narrow_tag_candidates(self):
        assert self.state.candidate_tags is not None
        prev_candidates = self.state.candidate_tags
        self.state.candidate_tags = [
            t for t in self.state.candidate_tags if t.startswith(self.state.tag_buffer)
        ]
        if self.debug and prev_candidates != self.state.candidate_tags:
            print(
                f"[DEBUG] Narrowed candidates from {len(prev_candidates)} to {len(self.state.candidate_tags)}: {self.state.candidate_tags}"
            )

    def _finish_tag_if_complete(self):
        assert self.state.candidate_tags is not None
        if any(t == self.state.tag_buffer for t in self.state.candidate_tags):
            finished = [
                t for t in self.state.candidate_tags if t == self.state.tag_buffer
            ]
            if finished:
                was_closing = finished[0] == self.closing_tag
                if self.debug:
                    print(
                        f"[DEBUG] Tag complete: {repr(finished[0])}, was_closing={was_closing}"
                    )
                self.state.mode = "COPY"
                self.state.inside_tag = not was_closing
                self.state.tag_buffer = ""
                self.state.candidate_tags = None
                self.state.first_tag_token = False
                self.state.just_opened_tag = not was_closing
                if self.debug:
                    print(
                        f"[DEBUG] Switched to COPY mode, inside_tag={self.state.inside_tag}"
                    )

    def _handle_tag_mode_piece(self, piece: str):
        # Continue building the tag
        if self.state.first_tag_token:
            idx = piece.find("<")
            if idx >= 0:
                piece = piece[idx:]
                if self.debug:
                    print(f"[DEBUG] First tag token, extracted from '<': {repr(piece)}")
            self.state.first_tag_token = False
        self.state.tag_buffer += piece
        if self.debug:
            print(f"[DEBUG] TAG mode: buffer now = {repr(self.state.tag_buffer)}")
        self._narrow_tag_candidates()
        self._finish_tag_if_complete()

    def _copy_remaining_tail_str(self) -> str:
        remaining_pieces = [
            self._get_token_piece(i)
            for i in self.input_token_ids[self.state.input_tok_pos :]
        ]
        return "".join(remaining_pieces)

    def _handle_invalid_tag_after_input_end(self, piece: str, idx: int):
        suffix = piece[idx:]
        self.state.finish_after_cut = True
        self.state.cut_suffix_chars = len(suffix)
        if self.debug:
            print(
                f"[DEBUG] COPY mode: invalid tag start after input end; treating as bridging suffix {repr(suffix)} and forcing stop (cut {self.state.cut_suffix_chars})"
            )

    def _handle_copy_mode_piece(self, piece: str, last_id: int):
        if "<" in piece:
            # Found a tag character within the token
            idx = piece.find("<")
            tag_fragment = piece[idx:]
            # Advance input pointer for any prefix before '<' that matches input tokens
            self._advance_input_pointer_with_prefix(piece[:idx])

            # Determine if tag_fragment is a valid starter given context
            is_closing_token = tag_fragment.startswith("</")
            if self.state.inside_tag:
                valid_start = self.closing_tag.startswith(tag_fragment)
                can_enter = is_closing_token and valid_start
                enter_opening = False
            else:
                valid_start = any(
                    t.startswith(tag_fragment) for t in self.allowed_opening_tags
                )
                can_enter = (not is_closing_token) and valid_start
                enter_opening = True

            if can_enter:
                if self.debug:
                    kind = "opening" if enter_opening else "closing"
                    print(
                        f"[DEBUG] COPY mode: detected valid {kind} tag start in piece; switching to TAG mode"
                    )
                self._enter_tag_mode(opening=enter_opening, first_piece=piece)
            else:
                if self.state.input_tok_pos >= len(self.input_token_ids):
                    self._handle_invalid_tag_after_input_end(piece, idx)
                else:
                    if self.debug:
                        print(
                            f"[DEBUG] WARNING: invalid tag start {repr(tag_fragment)} while still copying input; staying in COPY mode"
                        )
            return

        # Normal copy step: advance pointer if matches expected
        if self.state.input_tok_pos < len(self.input_token_ids):
            exp_id = self.input_token_ids[self.state.input_tok_pos]
            next_piece = self._get_token_piece(exp_id)
            if last_id == exp_id:
                self.state.input_tok_pos += 1
                if self.debug:
                    print(
                        f"[DEBUG] COPY mode: token matches expected input token, advanced to pos {self.state.input_tok_pos}"
                    )
                return

            # Immediately after an opening tag, avoid copying leading whitespace
            if self.state.just_opened_tag and next_piece and next_piece[:1].isspace():
                remainder = next_piece.lstrip()
                if piece == remainder:
                    self.state.input_tok_pos += 1
                    self.state.just_opened_tag = False
                    if self.debug:
                        print(
                            f"[DEBUG] COPY mode: consumed token remainder after opening tag (skipped leading space); advanced to pos {self.state.input_tok_pos}"
                        )
                    return

            # Special case: token piece contains the entire remaining input tail + extra suffix
            remaining_str = self._copy_remaining_tail_str()
            if remaining_str and piece.startswith(remaining_str):
                suffix = piece[len(remaining_str) :]
                self.state.input_tok_pos = len(self.input_token_ids)
                self.state.finish_after_cut = True
                self.state.cut_suffix_chars = len(suffix)
                if self.debug:
                    print(
                        f"[DEBUG] COPY mode: bridging token consumed full tail; suffix={repr(suffix)}; will request stop and cut {self.state.cut_suffix_chars} chars"
                    )
                return

            # Should not happen if constraints were respected, but don't crash
            if self.debug:
                print(
                    f"[DEBUG] WARNING: COPY mode token mismatch! Expected {exp_id} ({repr(self._get_token_piece(exp_id))}), got {last_id} ({repr(piece)})"
                )
        else:
            if self.debug:
                print(f"[DEBUG] COPY mode: finished copying all input tokens")

    def _update_state_from_input(self, input_ids: torch.LongTensor):
        # Process any newly appended tokens to update COPY/TAG state
        cur_len = input_ids.shape[-1]
        if self.state.seen_len == 0:
            # First call: nothing was generated yet (scores correspond to the next token)
            self.state.seen_len = cur_len
            if self.debug:
                print(f"[DEBUG] First call, prompt length: {cur_len}")
            return

        if cur_len <= self.state.seen_len:
            return

        # Iterate over newly added tokens since last call
        for pos in range(self.state.seen_len, cur_len):
            last_id = int(input_ids[0, pos].item())
            piece = self._get_token_piece(last_id)

            if self.debug:
                print(
                    f"\n[DEBUG] Processing generated token at pos {pos}: id={last_id}, piece={repr(piece)}"
                )
                print(
                    f"[DEBUG] Current state: mode={self.state.mode}, inside_tag={self.state.inside_tag}, input_tok_pos={self.state.input_tok_pos}/{len(self.input_token_ids)}"
                )

            if self.state.mode == "TAG":
                self._handle_tag_mode_piece(piece)
            else:
                self._handle_copy_mode_piece(piece, last_id)

        self.state.seen_len = cur_len

    def _mask_all_but(
        self, scores: torch.FloatTensor, allowed_ids: List[int]
    ) -> torch.FloatTensor:
        scores_processed = torch.full_like(scores, -math.inf)
        if len(allowed_ids) == 0:
            return scores_processed
        scores_processed[:, allowed_ids] = scores[:, allowed_ids]
        return scores_processed

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Batch size 1 only
        if input_ids.shape[0] != 1:
            raise ValueError(
                "XmlConstrainedLogitsProcessor currently supports batch_size=1 only"
            )

        self.step_counter += 1
        if self.debug:
            print(f"\n{'='*80}")
            print(f"[DEBUG] Step {self.step_counter}: Logits processor called")
            print(
                f"[DEBUG] Input shape: {input_ids.shape}, Scores shape: {scores.shape}"
            )
            # Print the sequence of tokens generated so far
            if input_ids.shape[-1] > 0:
                # Store prompt length on first call
                if self.step_counter == 1:
                    self.prompt_length = input_ids.shape[-1]
                # Decode only generated tokens (after prompt)
                if input_ids.shape[-1] > self.prompt_length:
                    generated_tokens = input_ids[0, self.prompt_length :]
                    generated_text = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=False
                    )
                    print(f"[DEBUG] Generated sequence so far: {repr(generated_text)}")
                else:
                    print(f"[DEBUG] Generated sequence so far: (none yet)")

        # If a previous token consumed the full tail plus a suffix, force stop by allowing only EOS
        if self.state.finish_after_cut:
            if self.debug:
                print("[DEBUG] finish_after_cut flagged: forcing EOS only")
            if self.eos_token_id is None:
                return self._mask_all_but(scores, [])
            return self._mask_all_but(scores, [int(self.eos_token_id)])

        # Update internal state using what has been generated so far
        self._update_state_from_input(input_ids)

        if self.debug:
            print(f"\n[DEBUG] Current state after update:")
            print(f"  mode: {self.state.mode}")
            print(f"  inside_tag: {self.state.inside_tag}")
            print(
                f"  input_tok_pos: {self.state.input_tok_pos}/{len(self.input_token_ids)}"
            )
            if self.state.mode == "TAG":
                print(f"  tag_buffer: {repr(self.state.tag_buffer)}")
                print(f"  candidate_tags: {self.state.candidate_tags}")
                print(f"  first_tag_token: {self.state.first_tag_token}")

        # Compute allowed ids for the NEXT token
        if self.state.mode == "COPY":
            allowed = self._allowed_ids_copy(scores)
            return self._mask_all_but(scores, allowed)
        else:
            allowed_next = self._allowed_ids_tag(scores)
            if not allowed_next:
                if self.debug:
                    print(
                        f"[DEBUG] WARNING: No tokens allowed in TAG mode! This shouldn't happen."
                    )
                return self._mask_all_but(scores, [])
            return self._mask_all_but(scores, list(dict.fromkeys(allowed_next)))

    # -----------------------------
    # Allowed token calculators
    # -----------------------------
    def _allowed_ids_copy(self, scores: torch.FloatTensor) -> List[int]:
        allowed: List[int] = []

        # Option A: copy next input token
        if self.state.input_tok_pos < len(self.input_token_ids):
            next_input_id = self.input_token_ids[self.state.input_tok_pos]
            next_piece = self._get_token_piece(next_input_id)
            if self.state.just_opened_tag and next_piece and next_piece[:1].isspace():
                remainder = next_piece.lstrip()
                remainder_ids = [
                    tid for tid, p in enumerate(self._token_pieces) if p == remainder
                ]
                if remainder_ids:
                    allowed.extend(remainder_ids)
                    if self.debug:
                        print(
                            f"[DEBUG] COPY mode: just opened tag; preferring remainder token(s) for next piece without leading space: {remainder_ids}"
                        )
                else:
                    allowed.append(next_input_id)
                    if self.debug:
                        print(
                            f"[DEBUG] COPY mode: no exact remainder token found; allowing next input token {next_input_id} ({repr(next_piece)})"
                        )
            else:
                allowed.append(next_input_id)
                if self.debug:
                    print(
                        f"[DEBUG] COPY mode: allowing next input token {next_input_id} ({repr(next_piece)})"
                    )
        else:
            # If finished copying and not inside a tag, allow EOS
            if self.eos_token_id is not None and not self.state.inside_tag:
                allowed.append(int(self.eos_token_id))
                if self.debug:
                    print(
                        f"[DEBUG] COPY mode: finished input, allowing EOS token {self.eos_token_id}"
                    )

        # Option A2: allow a single bridging token that contains the entire remaining input tail + suffix
        if self.state.input_tok_pos < len(self.input_token_ids):
            remaining_str = self._copy_remaining_tail_str()
            if remaining_str:
                bridge_ids: List[int] = []
                for tid, piece in enumerate(self._token_pieces):
                    if not piece:
                        continue
                    if piece.startswith(remaining_str):
                        if self._is_eos(tid):
                            continue
                        bridge_ids.append(tid)
                if bridge_ids:
                    if self.debug:
                        print(
                            f"[DEBUG] COPY mode: allowing {len(bridge_ids)} bridging tokens that start with full remaining tail"
                        )
                    allowed.extend(bridge_ids)

        # Option B: start a tag
        # Only allow tag starters if:
        # 1. We're inside a tag (need to allow closing tag), OR
        # 2. We still have input to copy (can add opening tags while copying)
        # This prevents empty tags at the end of the input
        if self.state.inside_tag:
            starters = self._closing_starter_ids
            if self.debug:
                print(
                    f"[DEBUG] COPY mode: inside tag, allowing {len(starters)} closing tag starters"
                )
            allowed.extend(starters)
        elif self.state.input_tok_pos < len(self.input_token_ids):
            starters = self._opening_starter_ids
            if self.debug:
                print(
                    f"[DEBUG] COPY mode: outside tag with remaining input, allowing {len(starters)} opening tag starters"
                )
            allowed.extend(starters)
        else:
            if self.debug:
                print(
                    f"[DEBUG] COPY mode: finished input and outside tag, NOT allowing opening tag starters (prevents empty tags)"
                )

        # Deduplicate
        allowed = list(dict.fromkeys(allowed))

        if self.debug:
            print(f"[DEBUG] Total allowed tokens in COPY mode: {len(allowed)}")
            # Show top-k token probabilities
            try:
                probs = torch.softmax(scores[0], dim=-1)
                top_probs, top_ids = torch.topk(probs, k=min(10, len(probs)))
                print(f"[DEBUG] Top 10 unconstrained predictions:")
                for i, (tid, prob) in enumerate(
                    zip(top_ids.tolist(), top_probs.tolist())
                ):
                    allowed_mark = "✓" if tid in allowed else "✗"
                    try:
                        piece_repr = repr(self._get_token_piece(tid)[:20])
                    except (IndexError, KeyError):
                        piece_repr = f"<unknown token {tid}>"
                    print(
                        f"  {i+1}. {allowed_mark} id={tid:5d} prob={prob:.4f} piece={piece_repr}"
                    )

                # Show probabilities of allowed tokens
                if len(allowed) <= 20:
                    print(f"[DEBUG] Allowed token probabilities:")
                    for tid in allowed:
                        prob = probs[tid].item() if tid < len(probs) else 0.0
                        try:
                            piece_repr = repr(self._get_token_piece(tid)[:30])
                        except (IndexError, KeyError):
                            piece_repr = f"<unknown token {tid}>"
                        print(f"  id={tid:5d} prob={prob:.4f} piece={piece_repr}")
            except Exception as e:
                print(f"[DEBUG] Exception during debug printing: {e}")

        return allowed

    def _allowed_ids_tag(self, scores: torch.FloatTensor) -> List[int]:
        assert self.state.candidate_tags is not None

        allowed_next: List[int] = []
        cur_buf = self.state.tag_buffer
        first_piece = self.state.first_tag_token

        if self.debug:
            print(f"[DEBUG] TAG mode: searching for tokens that continue tag")
            print(f"  current buffer: {repr(cur_buf)}")
            print(f"  candidates: {self.state.candidate_tags}")
            print(f"  first_tag_token: {first_piece}")

        # For each vocab token, test if appending its piece keeps a candidate prefix
        for tid in range(len(self._token_pieces)):
            piece = self._get_token_piece(tid)
            if self._is_eos(tid) or piece == "":
                continue
            if first_piece:
                idx = piece.find("<")
                if idx < 0:
                    continue
                test_buf = cur_buf + piece[idx:]
            else:
                test_buf = cur_buf + piece
            for cand in self.state.candidate_tags:
                if cand.startswith(test_buf):
                    allowed_next.append(tid)
                    break

        if self.debug:
            print(f"[DEBUG] TAG mode: found {len(allowed_next)} allowed tokens")
            try:
                probs = torch.softmax(scores[0], dim=-1)
                top_probs, top_ids = torch.topk(probs, k=min(10, len(probs)))
                print(f"[DEBUG] Top 10 unconstrained predictions:")
                for i, (tid, prob) in enumerate(
                    zip(top_ids.tolist(), top_probs.tolist())
                ):
                    allowed_mark = "✓" if tid in allowed_next else "✗"
                    try:
                        piece_repr = repr(self._get_token_piece(tid)[:20])
                    except (IndexError, KeyError):
                        piece_repr = f"<unknown token {tid}>"
                    print(
                        f"  {i+1}. {allowed_mark} id={tid:5d} prob={prob:.4f} piece={piece_repr}"
                    )

                # Show some allowed tokens
                if len(allowed_next) <= 20:
                    print(f"[DEBUG] All allowed token probabilities:")
                    for tid in allowed_next:
                        prob = probs[tid].item() if tid < len(probs) else 0.0
                        piece = self._get_token_piece(tid)
                        if first_piece:
                            idx = piece.find("<")
                            test_piece = piece[idx:] if idx >= 0 else piece
                        else:
                            test_piece = piece
                        test_result = cur_buf + test_piece
                        print(
                            f"  id={tid:5d} prob={prob:.4f} piece={repr(piece[:30])} -> {repr(test_result[:40])}"
                        )
                else:
                    print(f"[DEBUG] Sample of allowed tokens (first 10):")
                    for tid in allowed_next[:10]:
                        prob = probs[tid].item() if tid < len(probs) else 0.0
                        try:
                            piece_repr = repr(self._get_token_piece(tid)[:30])
                        except (IndexError, KeyError):
                            piece_repr = f"<unknown token {tid}>"
                        print(f"  id={tid:5d} prob={prob:.4f} piece={piece_repr}")
            except Exception as e:
                print(f"[DEBUG] Exception during debug printing: {e}")

        return allowed_next
