import json
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, RootModel
from span_labeling.methods.span_labeler import SpanLabeler
from enum import Enum


class SpanItem(BaseModel):
    text: str
    label: Optional[str] = None


class NERLabel(str, Enum):
    PER = "PER"
    ORG = "ORG"
    LOC = "LOC"


class NERSpanItem(BaseModel):
    text: str
    label: Optional[NERLabel] = None


class MultigecLabel(str, Enum):
    R = "R"
    M = "M"
    U = "U"


class MultigecSpanItem(BaseModel):
    text: str
    label: Optional[MultigecLabel] = None


class WMTLabel(str, Enum):
    MINOR = "MINOR"
    MAJOR = "MAJOR"


class WMTSpanItem(BaseModel):
    text: str
    label: Optional[WMTLabel] = None


class SyntheticSpanItem(BaseModel):
    text: str


class SpansOutput(RootModel[List[SpanItem]]):
    pass


class SpansOutputSpans(BaseModel):
    spans: List[SpanItem]


class NERSpanOutput(RootModel[List[NERSpanItem]]):
    pass


class NERSpanOutputSpans(BaseModel):
    spans: List[NERSpanItem]


class MultigecSpanOutput(RootModel[List[MultigecSpanItem]]):
    pass


class MultigecSpanOutputSpans(BaseModel):
    spans: List[MultigecSpanItem]


class WMTSpanOutput(RootModel[List[WMTSpanItem]]):
    pass


class WMTSpanOutputSpans(BaseModel):
    spans: List[WMTSpanItem]


class SyntheticSpanOutput(RootModel[List[SyntheticSpanItem]]):
    pass


class SyntheticSpanOutputSpans(BaseModel):
    spans: List[SyntheticSpanItem]


class JSONSpanLabeler(SpanLabeler):
    key: str = "json"

    @classmethod
    def get_json_schema(cls, task: str):
        """Return the JSON schema for structured outputs"""

        if task == "ner":
            print("Getting NER schema")
            return NERSpanOutput.model_json_schema()
        elif task == "multigec":
            print("Getting Multigec schema")
            return MultigecSpanOutput.model_json_schema()
        elif task == "wmt":
            print("Getting WMT schema")
            return WMTSpanOutput.model_json_schema()
        elif task == "synthetic":
            print("Getting Synthetic schema")
            return SyntheticSpanOutput.model_json_schema()

        return SpansOutput.model_json_schema()

    @classmethod
    def get_openai_json_schema(cls, task: str):
        """Return the JSON schema for structured outputs"""

        if task == "ner":
            print("Getting NER schema")
            return NERSpanOutputSpans
        elif task == "multigec":
            print("Getting Multigec schema")
            return MultigecSpanOutputSpans
        elif task == "wmt":
            print("Getting WMT schema")
            return WMTSpanOutputSpans
        elif task == "synthetic":
            print("Getting Synthetic schema")
            return SyntheticSpanOutputSpans

        return SpansOutputSpans

    def parse_response(self, entry: dict) -> List[Dict]:
        # Handle both structured outputs (already parsed) and string responses
        try:
            response = entry["response"]

            # Check if response is already a list (from structured outputs)
            if isinstance(response, list):
                # Convert Pydantic instances to dicts
                data = []
                for item in response:
                    if isinstance(item, BaseModel):
                        data.append(item.model_dump())
                    else:
                        data.append(item)
            # Otherwise parse as string
            elif isinstance(response, str):
                # Look for [...] pattern
                match = re.search(r"\[.*?\]", response, re.DOTALL)
                if match:
                    json_str = match.group()
                    data = json.loads(json_str)
                else:
                    return []

            else:
                return []

            results = []
            for item in data:
                span_text = item.get("text", "")
                label = item.get("label", "")

                # Find where this text appears
                idx = entry["text"].find(span_text)

                if idx != -1:
                    if entry["key"] == "multigec" and label == "M":
                        idx = idx + len(span_text) + 1

                    results.append(
                        {
                            "text": span_text,
                            "label": label,
                            "start": idx,
                            "end": idx + len(span_text),
                        }
                    )

            return results
        except Exception as e:
            print(f"Error: {e}")

        return []

    def parse_response_invalid(self, entry: dict) -> List[Dict]:
        try:
            response = entry["response"]

            # Check if response is already a list (from structured outputs)
            if isinstance(response, list):
                # Convert Pydantic instances to dicts
                data = []
                for item in response:
                    if isinstance(item, BaseModel):
                        data.append(item.model_dump())
                    else:
                        data.append(item)
            # Otherwise parse as string
            elif isinstance(response, str):
                # Look for [...] pattern
                match = re.search(r"\[.*?\]", response, re.DOTALL)
                if match:
                    json_str = match.group()
                    data = json.loads(json_str)
                else:
                    return []

            else:
                return []

            results = []
            for item in data:
                span_text = item.get("text", "")
                label = item.get("label", "")

                # Find where this text appears
                idx = entry["text"].find(span_text)

                if idx == -1:
                    results.append(
                        {
                            "text": span_text,
                            "label": label,
                            "start": "INVALID",
                            "end": "INVALID",
                        }
                    )

            return results
        except Exception as e:
            print(f"Error: {e}")

        return []
