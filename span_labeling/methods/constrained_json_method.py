import json
from typing import Dict

import requests

from span_labeling.config import get_vllm_api_url
from span_labeling.methods.json_method import JSONSpanLabeler


class ConstrainedJSONSpanLabeler(JSONSpanLabeler):
    """Call the local vLLM server and parse JSON spans.

    Inherits JSON parsing logic from JSONSpanLabeler to keep results identical in
    shape to other JSON-based methods. The server implements constrained decoding
    so that each returned span's "text" is an exact substring of the input.
    """

    # Keep the same name so the same prompt config (method="json") is used
    name: str = "json"

    def __init__(self, api_url: str | None = None):
        # We don't need a model_name here, but keep signature simple
        super().__init__(model_name="")
        self.api_url = (api_url or get_vllm_api_url()).rstrip("/")

    def run(self, entry: Dict) -> str:
        """Call the vLLM server with the formatted prompt and return the model's raw JSON.

        The returned string becomes entry["response"], which the inherited
        parse_response() will parse into spans.
        """
        payload = {
            "prompt": entry["prompt"],
            "input_text": entry["text"],
            "dataset": entry["key"],
            "constrained_key": "text",
        }

        try:
            resp = requests.post(
                f"{self.api_url}/detect_spans", json=payload, timeout=1800
            )
            resp.raise_for_status()
            data = resp.json()
            # Server returns {"raw": "[ {\"text\": ...} ]", ...}
            raw = data.get("raw", "")
            # Fallback: if raw missing, synthesize from spans list as JSON
            if not raw and isinstance(data.get("spans"), list):
                raw = json.dumps(data["spans"], ensure_ascii=False)
            return raw
        except requests.exceptions.RequestException as e:
            # Surface a simple text error; predict() will include it in output
            return f"[API ERROR] {e}"
