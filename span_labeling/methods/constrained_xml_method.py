from typing import Dict

import requests

from span_labeling.config import get_constrained_xml_debug, get_hf_api_url
from span_labeling.methods.xml_method import XMLSpanLabeler


class ConstrainedXMLSpanLabeler(XMLSpanLabeler):
    """Call the local FastAPI server and parse XML tags to spans.

    Inherits XML parsing logic from XMLSpanLabeler to keep results identical in
    shape to other XML-based methods.
    """

    # Keep the same name so the same prompt config (method="xml") is used
    name: str = "xml"

    def __init__(self, api_url: str | None = None):
        # We don't need a model_name here, but keep signature simple
        super().__init__(model_name="")
        self.api_url = (api_url or get_hf_api_url()).rstrip("/")

    def run(self, entry: Dict) -> str:
        """Call the API with dataset+method+text and return the annotated text.

        The returned string becomes entry["response"], which the inherited
        parse_response() will parse into spans.
        """
        # The API builds the full prompt server-side using (method, dataset).
        # We send only the raw input text and the identifiers.
        payload = {
            "text": entry["text"],
            "method": self.name,  # keep method name consistent with prompts ("xml")
            "dataset": entry["key"],  # e.g., "ner", "error", "multigec", ...
            "debug": get_constrained_xml_debug(),
        }

        try:
            resp = requests.post(f"{self.api_url}/annotate", json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()

            # The server returns both annotated_text and full_output; we want the
            # pure tagged text as the response to keep parsing logic simple.
            return data.get("annotated_text", "")
        except requests.exceptions.RequestException as e:
            # Surface a simple text error; predict() will include it in output
            return f"[API ERROR] {e}"
