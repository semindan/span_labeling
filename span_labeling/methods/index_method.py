import re
from typing import List, Dict
from span_labeling.methods.span_labeler import SpanLabeler
from span_labeling.prompt_utils import build_prompt


class IndexSpanLabeler(SpanLabeler):
    key: str = "index"

    def __init__(self, model_name: str, enrich_prompt: bool = False, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.enrich_prompt = enrich_prompt

    def parse_response(self, entry: dict) -> List[Dict]:
        results = []

        patterns = [
            r"\[(\d+):(\d+)\]\s*=\s*(\S+)",
            r'\[(\d+):(\d+)\]\s*=\s*(\S+),\s*"([^"]*)"',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, entry["response"]):
                start = int(match.group(1))
                end = int(match.group(2))
                label = match.group(3)

                if 0 <= start < end <= len(entry["text"]):
                    span_text = entry["text"][start:end]
                    results.append(
                        {"text": span_text, "label": label, "start": start, "end": end}
                    )

        return results

    def format_prompt(self, entry: dict) -> str:
        note_extra = ""
        if self.enrich_prompt:
            note_extra = "- Rely on the character indices provided before words: char_index::word"
            entry = self.enrich(entry)
        return build_prompt(self.key, entry["key"], entry, note_extra=note_extra)

    def enrich(self, entry: dict) -> dict:
        text = entry["text"]
        model_input = entry["model_input"]

        enriched_text = self.add_char_indices(text)
        entry["model_input"] = model_input.replace(text, enriched_text)

        return entry

    def add_char_indices(self, text: str) -> str:
        """
        Add character indices before each word.

        Example:
            "Apple Inc is big"
            → "0::Apple 6::Inc 10::is 13::big"
        """
        words = text.split()
        result = []
        char_pos = 0

        for word in words:
            result.append(f"{char_pos}::{word}")
            char_pos += len(word) + 1  # +1 for space

        return " ".join(result)

    def add_char_indices_detailed(self, text: str) -> str:
        """
        Show start position for each word.

        Example:
            "Apple Inc"
            → "[0-5]::Apple [6-9]::Inc"
        """
        words = text.split()
        result = []
        char_pos = 0

        for word in words:
            end_pos = char_pos + len(word)
            result.append(f"[{char_pos}-{end_pos}]::{word}")
            char_pos = end_pos + 1

        return " ".join(result)
