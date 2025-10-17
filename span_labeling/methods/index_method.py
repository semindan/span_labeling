import re
import textwrap
from typing import List, Dict
from span_labeling.base import SpanLabeler


format: dict[str, str] = {
    "ner": textwrap.dedent("""
        Output each span as: [start:end] = LABEL
        Character positions are 0-indexed.
        Labels: PERSON, ORG, LOC
        Example: [0:9] = ORG
                 [25:35] = PERSON
    """),
    "synthetic": textwrap.dedent("""
        Output each span as: [start:end] = LABEL
        Character positions are 0-indexed.
        Example: [0:3] = MATCH
                 [8:11] = MATCH
    """),
    "error": textwrap.dedent("""
        Output each span as: [start:end] = LABEL
        Character positions are 0-indexed.
        Labels: GRAMMAR, SPELLING, PUNCTUATION
        Example: [3:5] = GRAMMAR
                 [10:15] = SPELLING
    """),
    "multigec": textwrap.dedent("""
        Output each span as: [start:end] = LABEL, CORRECTION
        Character positions are 0-indexed.
        Labels: R, U, M
        Example: [0:3] = R, "the"
                 [8:11] = U, ""
                 [15:20] = M, "a"
    """),
    "default": textwrap.dedent("""
        Output each span as: [start:end] = LABEL
        Character positions are 0-indexed.
        Example: [0:9] = ORG
    """),
}

class IndexSpanLabeler(SpanLabeler):
    def format_prompt(self, entry: dict) -> str:
        return textwrap.dedent(
            f"""Task: {entry['instruction']}

            Text: "{entry['text']}"

            {format.get(entry.get('key', None), format['default'])}

            Output:""")
    
    def parse_response(self, entry: dict) -> List[Dict]:
        results = []
        
        patterns = [r'\[(\d+):(\d+)\]\s*=\s*(\S+)',
                    r'\[(\d+):(\d+)\]\s*=\s*(\S+),\s*"([^"]*)"']
                    
        for pattern in patterns:
            for match in re.finditer(pattern, entry["response"]):
                start = int(match.group(1))
                end = int(match.group(2))
                label = match.group(3)
            
                if 0 <= start < end <= len(entry["text"]):
                    span_text = entry["text"][start:end]
                    results.append({
                        'text': span_text,
                        'label': label,
                        'start': start,
                        'end': end
                    })
        
        return results