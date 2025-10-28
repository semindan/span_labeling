# methods/json_occurrence_method.py
import textwrap
from span_labeling.methods.json_method import JSONSpanLabeler


format: dict[str, str] = {
    "ner": textwrap.dedent("""
            Return a JSON list. Each item must have:
            - "text": exact span or pattern from input
            - "label": category (PERSON, ORG, LOC)
            - "occurrence": which occurrence (1 for first, 2 for second, etc.)
        """),
    "synthetic": textwrap.dedent("""
            Return a JSON list. Each item must have:
            - "text": exact span or pattern from input
            - "occurrence": which occurrence (1 for first, 2 for second, etc.)
        """),
    "error": textwrap.dedent("""
            Return a JSON list. Each item must have:
            - "text": exact span or pattern from input
            - "label": category (GRAMMAR, SPELLING, or PUNCTUATION)
            - "occurrence": which occurrence (1 for first, 2 for second, etc.)
        """),
    "multigec": textwrap.dedent("""
            Return a JSON list. Each item must have:
            - "text": exact span or pattern from input
            - "label": error category (R, U or M)
            - "correction": correction text
            - "occurrence": which occurrence (1 for first, 2 for second, etc.)
        """),
    "wmt": textwrap.dedent("""
            Return a JSON list. Each item must have:
            - "text": exact span or pattern from translation
            - "occurrence": which occurrence (1 for first, 2 for second, etc.)
        """),
    "default": textwrap.dedent("""
            Return a JSON list. Each item must have:
            - "text": exact span or pattern from input
            - "label": category (if applicable)
            - "occurrence": which occurrence (1 for first, 2 for second, etc.)
        """),
}


class JSONOccurrenceSpanLabeler(JSONSpanLabeler):
    name: str = "occurrence"

    # def format_prompt(self, entry: dict) -> str:
    #     prompt = textwrap.dedent(
    #     f"""{entry['instruction']}

    #         {entry["model_input"]}

    #         {format.get(entry.get('key', None), format['default'])}

    #         JSON output:""")

    #     return prompt

    def parse_response(self, entry: dict) -> list[dict]:
        spans = super().parse_response(entry)

        # Use occurrence to disambiguate
        for span in spans:
            occ = span.get("occurrence", 1)
            # Find the nth occurrence
            start = -1
            for i in range(occ):
                start = entry["text"].find(span["text"], start + 1)

            if start != -1:
                span["start"] = start
                span["end"] = start + len(span["text"])

        return spans
