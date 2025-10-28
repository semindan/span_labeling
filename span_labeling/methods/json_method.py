import json
import re
import textwrap
from typing import List, Dict
from span_labeling.methods.span_labeler import SpanLabeler


format: dict[str, str] = {
    "ner": textwrap.dedent("""
        Return a JSON list. Each item must have:
        - "text": exact text span from input
        - "label": category (PERSON, ORG, LOC)
        Example: [{"text": "Apple Inc", "label": "ORG"}]
    """),
    "synthetic": textwrap.dedent("""
        Return a JSON list. Each item must have:
        - "text": exact span or pattern from input
        Example: [{"text": "cat"}]
    """),
    "error": textwrap.dedent("""
        Return a JSON list. Each item must have:
        - "text": exact text span from input
        - "label": category (GRAMMAR, SPELLING, PUNCTUATION)
        Example: [{"text": "go", "label": "GRAMMAR"}]
    """),
    "multigec": textwrap.dedent("""
        Return a JSON list. Each item must have:
        - "text": exact text span from input
        - "label": error category (R, U or M)
        - "correction": correction text
        Example: [{"text": "teh", "label": "R", "correction": "the"}]
    """),
    "default": textwrap.dedent("""
        Return a JSON list. Each item must have:
        - "text": exact text span from input
        - "label": the category
        Example: [{"text": "Apple", "label": "ORG"}]
    """),
}


class JSONSpanLabeler(SpanLabeler):
    name: str = "json"

    # def format_prompt(self, entry: dict) -> str:
    #     return textwrap.dedent(
    #     f"""{entry['instruction']}

    #         {entry['model_input']}"

    #         {format.get(entry.get('key', None), format['default'])}

    #         JSON output:""")

    def parse_response(self, entry: dict) -> List[Dict]:
        # Find JSON in response
        try:
            # Look for [...] pattern
            match = re.search(r"\[.*?\]", entry["response"], re.DOTALL)
            if match:
                json_str = match.group()
                data = json.loads(json_str)

                results = []
                for item in data:
                    span_text = item.get("text", "")
                    label = item.get("label", "")

                    # Find where this text appears
                    idx = entry["text"].find(span_text)
                    if idx != -1:
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
