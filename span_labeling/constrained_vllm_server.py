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
    """

    def __init__(
        self,
        input_text: str,
        id2piece: dict[int, str],
        constrained_key: str,
        quote_token_ids: Optional[list[int]] = None,
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

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
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
                print(
                    f"[LogitsProcessor] Detected value prefix for span #{self.span_index} at position {self.prefix_end_pos}; prefilled value chars={len(self.initial_value_after_quote)}",
                    file=sys.stderr,
                )

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
            # Find starts where vt is a prefix of input_text starting at that index
            starts: list[int] = []
            if vt == "":
                starts = list(range(len(self.input_text)))
            else:
                start = 0
                while True:
                    idx = self.input_text.find(vt, start)
                    if idx == -1:
                        break
                    starts.append(idx)
                    start = idx + 1

            for token_id, piece in self.id2piece.items():
                if not piece:
                    continue
                candidate = vt + piece
                for s in starts:
                    if self.input_text.startswith(candidate, s):
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

            # If any emitted piece contains a quote at this point, treat the value as closed
            if any('"' in p for p in pieces):
                print(
                    f"[LogitsProcessor] Detected closing quote (any-quote token) for span #{self.span_index}. Lifting constraints.",
                    file=sys.stderr,
                )
                self.prefix_end_pos = None
                self.value_so_far = ""
                self.initial_value_after_quote = ""
                self._last_logged_len = 0
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
        if not input_text or not id2piece or not constrained_key:
            return None
        return SubstringCopyLogitsProcessor(
            input_text=input_text,
            id2piece=id2piece,
            constrained_key=constrained_key,
            quote_token_ids=quote_token_ids,
        )


class ModelState:
    """Encapsulates vLLM model and tokenizer state."""

    def __init__(
        self,
        mode: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        logits = [SubstringCopyLogitsProcessorAdapter] if mode == "constrained" else []
        self.llm = LLM(
            model=model,
            max_model_len=2048,
            logits_processors=logits,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.id2piece: dict[int, str] = {}
        self.quote_token_ids: set[int] = set()
        self.temperature = temperature
        self.max_tokens = max_tokens
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

        sp_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if structured_outputs is not None:
            sp_kwargs["structured_outputs"] = structured_outputs
        if mode == "constrained":
            sp_kwargs["extra_args"] = {
                "input_text": input_text,
                "id2piece": self.id2piece,
                "quote_token_ids": list(self.quote_token_ids),
                "constrained_key": constrained_key,
            }
        sampling_params = SamplingParams(**sp_kwargs)

        # The prompt is pre-formatted by the client and should include system message if needed
        # Pass it directly as a user message
        messages = [
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
            constrained_key = (
                payload.get("constrained_key")
                if self.server_mode == "constrained"
                else None
            )

            result = self.model_state.detect_spans(
                input_text=input_text,
                prompt=prompt,
                mode=self.server_mode,
                dataset=dataset,
                constrained_key=constrained_key,
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


def run(
    host: str,
    port: int,
    mode: str,
    model: str,
    temperature: float,
    max_tokens: int,
):
    # Initialize model state once at server startup
    model_state = ModelState(
        mode=mode,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Bind the model state and mode to the handler class
    Handler.model_state = model_state
    Handler.server_mode = mode
    server = ThreadingHTTPServer((host, port), Handler)
    print(
        f"[Server] Listening on http://{host}:{port} (mode={mode}, model={model}, temp={temperature}, max_tokens={max_tokens})"
    )
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()
    run(
        host=args.host,
        port=args.port,
        mode=args.mode,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
