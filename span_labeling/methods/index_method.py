import re
import textwrap
from typing import List, Dict
from span_labeling.methods.span_labeler import SpanLabeler

format: dict[str, str] = {
    "ner": textwrap.dedent("""
        Output each span as: [start:end] = LABEL
                           
        IMPORTANT: Character positions are 0-indexed.
        - First character is at position 0
        - Spaces count as characters
        - start is inclusive, end is exclusive
        
        Labels: PERSON, ORG, LOC
        
        Format Example:
            Text: "Apple Inc is in Cupertino"
            Positions: A=0, p=1, p=2, l=3, e=4, (space)=5, I=6...
            Output: 
            [0:9] = ORG
            [16:25] = LOC
            
        Explanation:
        - "Apple Inc" spans positions 0-9 (A at 0, c at 8, space at 9 not included)
        - "Cupertino" spans positions 16-25
    """),
    "synthetic": textwrap.dedent("""
        Output each span as: [start:end] = MATCH
                                 
        IMPORTANT: Character positions are 0-indexed.
        - First character is at position 0
        - Spaces count as characters
        - start is inclusive, end is exclusive

        Format Example:
            Text: "Apple Inc is in Cupertino"
            Positions: A=0, p=1, p=2, l=3, e=4, (space)=5, I=6...
            Output: 
            [0:9]
            [16:25]
            
        Explanation:
        - "Apple Inc" spans positions 0-9 (A at 0, c at 8, space at 9 not included)
        - "Cupertino" spans positions 16-25
                                 
        Example: [0:3] = MATCH
                 [8:11] = MATCH
    """),
    "error": textwrap.dedent("""
        Output each span as: [start:end] = LABEL

        IMPORTANT: Character positions are 0-indexed.
        - First character is at position 0
        - Spaces count as characters
        - start is inclusive, end is exclusive
                             
        Labels: GRAMMAR, SPELLING, PUNCTUATION

        Format Example:
            Text: "Apple Inc is in Cupertino"
            Positions: A=0, p=1, p=2, l=3, e=4, (space)=5, I=6...
            Output: 
            [0:9]
            [16:25]
            
        Explanation:
        - "Apple Inc" spans positions 0-9 (A at 0, c at 8, space at 9 not included)
        - "Cupertino" spans positions 16-25

        Example: [3:5] = GRAMMAR
                 [10:15] = SPELLING
    """),
    "multigec": textwrap.dedent("""
        Output each span as: [start:end] = LABEL, CORRECTION

        IMPORTANT: Character positions are 0-indexed.
        - First character is at position 0
        - Spaces count as characters
        - start is inclusive, end is exclusive
                             
        Labels: R, U, M

        Format Example:
            Text: "Apple Inc is in Cupertino"
            Positions: A=0, p=1, p=2, l=3, e=4, (space)=5, I=6...
            Output: 
            [0:9]
            [16:25]
            
        Explanation:
        - "Apple Inc" spans positions 0-9 (A at 0, c at 8, space at 9 not included)
        - "Cupertino" spans positions 16-25

        Example: [0:3] = R, "the"
                 [8:11] = U, ""
                 [15:20] = M, "a"
    """),
    "wmt": textwrap.dedent("""
    Output each error span as: [start:end] = SEVERITY

    IMPORTANT: Character positions are 0-indexed IN THE TRANSLATION.
    - First character is at position 0
    - Spaces count as characters
    - start is inclusive, end is exclusive
    
    Error Severity:
    - 0 = Minor error (understandable but not perfect)
    - 1 = Major error (changes meaning or unintelligible)

    Format Example:
        Source: "The house is very big."
        Translation: "Das Haus ist große."
        Positions in translation: D=0, a=1, s=2, (space)=3, H=4, a=5, u=6, s=7...
        Output: 
        [13:18] = 1
        
    Explanation:
    - "große" (wrong adjective form) is at positions 13-18 in the TRANSLATION
    - This is a major error (severity 1) because the grammar is wrong

    Example with minor error:
        Source: "I like cats"
        Translation: "Ich mag Katzen sehr"
        Output: [15:19] = 0
        (Extra word "sehr" is minor - meaning still clear)
    
    Compare the translation to the source text to identify errors.
    Mark errors with their severity level (0 or 1).
"""),
    "default": textwrap.dedent("""
        Output each span as: [start:end] = LABEL
                           
        IMPORTANT: Character positions are 0-indexed.
        - First character is at position 0
        - Spaces count as characters
        - start is inclusive, end is exclusive
        
        Labels: PERSON, ORG, LOC
        
        Format Example:
            Text: "Apple Inc is in Cupertino"
            Positions: A=0, p=1, p=2, l=3, e=4, (space)=5, I=6...
            Output: 
            [0:9] = ORG
            [16:25] = LOC
            
        Explanation:
        - "Apple Inc" spans positions 0-9 (A at 0, c at 8, space at 9 not included)
        - "Cupertino" spans positions 16-25
    """),
}


class IndexSpanLabeler(SpanLabeler):
    name: str = "index"

    def __init__(self, model_name: str, enrich_prompt: bool = False):
        super().__init__(model_name=model_name)
        self.enrich_prompt = enrich_prompt

    # def format_prompt(self, entry: dict) -> str:
    #     entry = self.enrich(deepcopy(entry))

    #     return textwrap.dedent(
    #     f"""{entry['instruction']}

    #         {entry['model_input']}

    #         {format.get(entry.get('key', None), format['default'])}

    #         Output:""")

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

    def enrich(self, entry: dict) -> dict:
        if not self.enrich_prompt:
            return entry

        text = entry["text"]
        model_input = entry["model_input"]

        enriched_text = add_char_indices(text)
        entry["model_input"] = (
            model_input.replace(text, enriched_text)
            + "\nRely on the character indices provided before words."
        )

        return entry


def add_char_indices(text: str) -> str:
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


def add_char_indices_detailed(text: str) -> str:
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
