from __future__ import annotations
from enum import Enum

from span_labeling.methods.json_method import JSONSpanLabeler
from span_labeling.methods.occurrence_method import JSONOccurrenceSpanLabeler
from span_labeling.methods.xml_method import XMLSpanLabeler
from span_labeling.methods.index_method import IndexSpanLabeler
import json


class ErrorType(str, Enum):
    EMPTY_RESPONSE = "empty_response"
    FORMAT_ERROR = "format_error"
    SPAN_NOT_FOUND = "span_not_found"
    PARTIAL_SPAN_NOT_FOUND = "partial_span_not_found"
    INVALID_LABEL = "invalid_label"
    PARTIAL_INVALID_LABEL = "partial_invalid_label"
    EMPTY_PREDICTION = "empty_prediction"
    SUCCESS = "success"
    FORMAT_MAX_TOKENS_EXCEEDED = "format_max_tokens_exceeded"
    FORMAT_UNESCAPED_QUOTES = "format_unescaped_quotes"
    EMPTY_LABEL = "empty_label"
    PARTIAL_EMPTY_LABEL = "partial_empty_label"
    QUOTATIONS_IN_SPANS = "quotations_in_spans"
    PARTIAL_QUOTATIONS_IN_SPANS = "partial_quotations_in_spans"


def _get_labeler(method_name):
    """Return a labeler instance for the given method name."""
    if "occurrence" in method_name:
        return JSONOccurrenceSpanLabeler(None, None)
    if method_name.startswith("json"):
        return JSONSpanLabeler(None, None)
    if method_name.startswith("xml"):
        return XMLSpanLabeler(None, None)
    if method_name.startswith("index"):
        return IndexSpanLabeler(None, None)
    raise ValueError(f"Unknown method: {method_name!r}")


def _get_output_part_json(raw_response: str) -> str:
    """Extract the part of the raw response that should contain the spans, for better error analysis."""
    raw_response = str(raw_response) if raw_response is not None else ""
    if "Output:" in raw_response:
        return raw_response.split("Output:")[-1].strip()
    if "```json" in raw_response:
        return raw_response.split("```json")[-1].split("```")[0].strip()

    # structured_match = (
    #     re.search(r".*(\[.*\])", raw_response, re.DOTALL) or
    #     re.search(r".*(\{.*\})", raw_response, re.DOTALL)
    # )

    start = raw_response.rfind("[")
    end = raw_response.rfind("]")
    if start != -1 and end != -1 and start < end:
        ret = raw_response[start : end + 1].strip()
        if '"spans":' in ret:
            try:
                ret = json.loads(ret).get("spans", [])
                return str(ret)
            except json.JSONDecodeError:
                pass
        elif ret:
            return ret

    return raw_response.strip()


def _get_output_part(raw_response: str) -> str:
    raw_response = str(raw_response) if raw_response is not None else ""
    if "Output:" in raw_response:
        return raw_response.split("Output:")[-1].strip()
    return raw_response.strip()


def analyze_error_json_based(entry: dict) -> ErrorType:
    output = entry.get("output") or {}
    located_spans: list[dict] = output.get("spans") or []
    allowed_labels: list | None = entry.get("allowed_labels")
    dataset_type = entry.get("metadata", {}).get("dataset", {}).get("type", "unknown")
    method_name = entry.get("metadata", {}).get("method", {}).get("name", "unknown")
    # method_type = entry.get("metadata", {}).get("method", {}).get("type", "unknown")
    thinking = entry.get("metadata", {}).get("model", {}).get("thinking", False)
    raw_response = output.get("raw_response", "")

    raw_response = _get_output_part_json(raw_response)

    if not raw_response:
        return ErrorType.EMPTY_RESPONSE

    json_parsed = None
    json_issue = None
    try:
        json_parsed = json.loads(raw_response)
        if "spans" in json_parsed:
            json_parsed = json_parsed["spans"]
    except json.JSONDecodeError as e:
        json_parsed = None
        json_issue = e

    if raw_response == "[]" or (json_parsed is not None and json_parsed == []):
        return ErrorType.EMPTY_PREDICTION

    labeler = _get_labeler(method_name)
    if hasattr(labeler, "parse_response_invalid"):
        try:
            attempted_spans = labeler.parse_response_invalid(entry)
        except Exception:
            attempted_spans = []
    else:
        attempted_spans = []

    # if there are no valid located spans and even no attempted spans, it's probably a format error
    if not attempted_spans and not located_spans:
        if (entry["completion_tokens"] >= 4096 and not thinking) or (
            entry["completion_tokens"] >= 16384 and thinking
        ):
            return ErrorType.FORMAT_MAX_TOKENS_EXCEEDED

        if json_issue:
            if json_issue.msg == "Expecting ',' delimiter":
                return ErrorType.FORMAT_UNESCAPED_QUOTES
            elif json_issue.msg == "Expecting property name enclosed in double quotes":
                return ErrorType.FORMAT_UNESCAPED_QUOTES
            elif json_issue.msg == "Expecting ':' delimiter":
                return ErrorType.FORMAT_UNESCAPED_QUOTES
            else:
                print(f"Raw response: {raw_response}")
                print(f"JSON parsing error: {json_issue}")
                print(f"Message: {json_issue.msg}")

        return ErrorType.FORMAT_ERROR

    # attempted spans are already spans that are not in the text, so if there are some attempted spans and no located spans
    # it's a span not found error

    # we just check located spans, if there are none -> there are no spans from the text
    if len(located_spans) == 0:
        # we have no located spans, but if we have attempted spans with quotation marks, we know it's a specific error
        if all('"' in s.get("text", "") for s in attempted_spans):
            return ErrorType.QUOTATIONS_IN_SPANS

        return ErrorType.SPAN_NOT_FOUND

    # here we know we have located spans, but if we also have attempted spans, we say it's a partial span not found
    # we must have valid spans and only then we look at labels
    if len(attempted_spans) > 0:
        if any('"' in s.get("text", "") for s in attempted_spans):
            return ErrorType.PARTIAL_QUOTATIONS_IN_SPANS
        return ErrorType.PARTIAL_SPAN_NOT_FOUND

    # here we know we have located spans and we don't have any failed (attempted) spans
    # apart from synthetic datasets, we check label validity if we have valid labels in located spans
    if dataset_type != "synthetic" and allowed_labels:
        allowed = set(str(_l) for _l in allowed_labels)
        invalid = sum(
            1 for s in located_spans if str(s.get("label", "")) not in allowed
        )
        if invalid == len(located_spans):
            if all(
                str(s.get("label", "")) == "" or str(s.get("label", "")) == "None"
                for s in located_spans
            ):
                return ErrorType.EMPTY_LABEL

            return ErrorType.INVALID_LABEL
        if invalid > 0:
            if any(
                str(s.get("label", "")) == "" or str(s.get("label", "")) == "None"
                for s in located_spans
            ):
                return ErrorType.PARTIAL_EMPTY_LABEL
            return ErrorType.PARTIAL_INVALID_LABEL

    return ErrorType.SUCCESS


def analyze_error_other(entry: dict) -> ErrorType:
    output = entry.get("output") or {}
    located_spans: list[dict] = output.get("spans") or []
    allowed_labels: list | None = entry.get("allowed_labels")
    dataset_type = entry.get("metadata", {}).get("dataset", {}).get("type", "unknown")
    # method_name = entry.get("metadata", {}).get("method", {}).get("name", "unknown")
    method_type = entry.get("metadata", {}).get("method", {}).get("type", "unknown")
    thinking = entry.get("metadata", {}).get("model", {}).get("thinking", False)
    raw_response = output.get("raw_response", "")

    raw_response = _get_output_part(raw_response)

    if not raw_response:
        return ErrorType.EMPTY_RESPONSE
    if raw_response == "[]" or raw_response == "{}" or raw_response.lower() == "none":
        return ErrorType.EMPTY_PREDICTION
    if method_type == "xml" and raw_response == entry.get("text", ""):
        # in XML method, if the model just returns the text without any tags, it's an empty prediction, not a format error
        return ErrorType.EMPTY_PREDICTION

    if not located_spans:
        if (entry["completion_tokens"] >= 4096 and not thinking) or (
            entry["completion_tokens"] >= 16384 and thinking
        ):
            return ErrorType.FORMAT_MAX_TOKENS_EXCEEDED

        return ErrorType.FORMAT_ERROR

    # apart from synthetic datasets, we check label validity if we have valid labels in located spans
    if dataset_type != "synthetic" and allowed_labels:
        allowed = set(str(_l) for _l in allowed_labels)
        invalid = sum(
            1 for s in located_spans if str(s.get("label", "")) not in allowed
        )
        if invalid == len(located_spans):
            if all(
                str(s.get("label", "")) == "" or str(s.get("label", "")) == "None"
                for s in located_spans
            ):
                return ErrorType.EMPTY_LABEL

            return ErrorType.INVALID_LABEL
        if invalid > 0:
            if any(
                str(s.get("label", "")) == "" or str(s.get("label", "")) == "None"
                for s in located_spans
            ):
                return ErrorType.PARTIAL_EMPTY_LABEL
            return ErrorType.PARTIAL_INVALID_LABEL

    return ErrorType.SUCCESS


def analyze_error(
    entry: dict,
) -> ErrorType:
    """
    Classify a single result entry into one ErrorType.

    Args:
        entry:        A result entry dict (see module docstring).
    Returns:
        An ErrorType enum value.
    """

    method_type = entry.get("metadata", {}).get("method", {}).get("type", "unknown")
    if "json" in method_type:
        return analyze_error_json_based(entry)
    else:
        return analyze_error_other(entry)
