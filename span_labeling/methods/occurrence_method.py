import json
import re
from typing import List, Optional
from pydantic import BaseModel, RootModel
from span_labeling.methods.json_method import JSONSpanLabeler
from enum import Enum


class SpanItemWithOccurrence(BaseModel):
    text: str
    label: Optional[str] = None
    occurrence: int


class SpansOccurrenceOutput(RootModel[List[SpanItemWithOccurrence]]):
    pass


class SpansOccurrenceOutputSpans(BaseModel):
    spans: List[SpanItemWithOccurrence]


class NERLabel(str, Enum):
    PER = "PER"
    ORG = "ORG"
    LOC = "LOC"


class NERSpanItemWithOccurrence(BaseModel):
    text: str
    label: Optional[NERLabel] = None
    occurrence: int


class MultigecLabel(str, Enum):
    R = "R"
    M = "M"
    U = "U"


class MultigecSpanItemWithOccurrence(BaseModel):
    text: str
    label: Optional[MultigecLabel] = None
    occurrence: int


class WMTLabel(str, Enum):
    MINOR = "MINOR"
    MAJOR = "MAJOR"


class WMTSpanItemWithOccurrence(BaseModel):
    text: str
    label: Optional[WMTLabel] = None
    occurrence: int


class SyntheticSpanItemWithOccurrence(BaseModel):
    text: str
    occurrence: int


class NERSpanOutputWithOccurrence(RootModel[List[NERSpanItemWithOccurrence]]):
    pass


class NERSpanOutputWithOccurrenceSpans(BaseModel):
    spans: List[NERSpanItemWithOccurrence]


class MultigecSpanOutputWithOccurrence(RootModel[List[MultigecSpanItemWithOccurrence]]):
    pass


class MultigecSpanOutputWithOccurrenceSpans(BaseModel):
    spans: List[MultigecSpanItemWithOccurrence]


class WMTSpanOutputWithOccurrence(RootModel[List[WMTSpanItemWithOccurrence]]):
    pass


class WMTSpanOutputWithOccurrenceSpans(BaseModel):
    spans: List[WMTSpanItemWithOccurrence]


class SyntheticSpanOutputWithOccurrence(
    RootModel[List[SyntheticSpanItemWithOccurrence]]
):
    pass


class SyntheticSpanOutputWithOccurrenceSpans(BaseModel):
    spans: List[SyntheticSpanItemWithOccurrence]


class JSONOccurrenceSpanLabeler(JSONSpanLabeler):
    key: str = "json_occurrence"

    @classmethod
    def get_json_schema(self, task: str, mode: str):
        if mode == "vllm":
            return self.get_vllm_json_schema(task)
        elif mode == "openai":
            return self.get_openai_json_schema(task)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def get_vllm_json_schema(self, task: str):
        """Return the JSON schema for structured outputs with occurrence"""
        if task == "ner":
            return NERSpanOutputWithOccurrenceSpans.model_json_schema()
        elif task == "multigec":
            return MultigecSpanOutputWithOccurrenceSpans.model_json_schema()
        elif task == "wmt":
            return WMTSpanOutputWithOccurrenceSpans.model_json_schema()
        elif task == "synthetic":
            return SyntheticSpanOutputWithOccurrenceSpans.model_json_schema()

        return SpansOccurrenceOutputSpans.model_json_schema()

    @classmethod
    def get_openai_json_schema(self, task: str):
        """Return the JSON schema for structured outputs with occurrence"""
        if task == "ner":
            return NERSpanOutputWithOccurrenceSpans
        elif task == "multigec":
            return MultigecSpanOutputWithOccurrenceSpans
        elif task == "wmt":
            return WMTSpanOutputWithOccurrenceSpans
        elif task == "synthetic":
            return SyntheticSpanOutputWithOccurrenceSpans
        return SpansOccurrenceOutputSpans

    def parse_response(self, entry: dict) -> list[dict]:
        # Handle both structured outputs (already parsed) and string responses
        try:
            response = entry["response"]
            if "Output:" in response:
                response = response.split("Output:")[-1].strip()

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
                occurrence = item.get("occurrence", 1)

                # Find the nth occurrence
                start = -1
                # last_found = -1
                for i in range(occurrence):
                    start = entry["text"].find(span_text, start + 1)
                #     if start != -1:
                #         last_found = start

                # if start == -1 and last_found != -1:
                #     start = last_found

                if start != -1:
                    if entry["key"] == "multigec" and label == "M":
                        start = start + len(span_text) + 1

                    results.append(
                        {
                            "text": span_text,
                            "label": label,
                            "start": start,
                            "end": start + len(span_text),
                        }
                    )

            return results
        except Exception as e:
            print(f"Error: {e}")

        return []

    def parse_response_invalid(self, entry: dict) -> list[dict]:
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
                occurrence = item.get("occurrence", 1)

                start = -1
                last_found = -1
                for i in range(occurrence):
                    start = entry["text"].find(span_text, start + 1)
                    if start != -1:
                        last_found = start

                if start == -1 and last_found != -1:
                    start = last_found

                if start == -1:
                    results.append(
                        {
                            "text": span_text,
                            "label": label,
                            "start": start,
                            "end": start + len(span_text),
                        }
                    )

            return results
        except Exception as e:
            print(f"Error: {e}")

        return []
