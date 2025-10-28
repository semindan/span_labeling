import re
import textwrap
from typing import List, Dict
from span_labeling.methods.span_labeler import SpanLabeler

format: dict[str, str] = {
    "ner": textwrap.dedent("""
        Rewrite the whole input text with XML tags around entities.
        Only use the following labels: PERSON, ORG, LOC
        Format: <entity type="LABEL">text</entity>
        Example:
            Text: Apple is in Cupertino.  
            Tagged text: <entity type="ORG">Apple</entity> is in <entity type="LOC">Cupertino</entity>.
                           
        IMPORTANT: You must output the entire text, including non-tagged parts.
    """),
    "synthetic": textwrap.dedent("""
        Rewrite the whole input text with XML tags around matching patterns.
        Format: <match>text</match>
        Format Example:
            Text: The cat sat on the mat.
            Tagged text: The <match>cat</match> sat on the mat.

        IMPORTANT: You must output the entire text, including non-tagged parts.
    """),
    "error": textwrap.dedent("""
        Rewrite the whole input text with XML tags inserted around errors.
        Only tag the incorrect words.
        Only use the following labels: GRAMMAR, SPELLING, PUNCTUATION
        Format: <error type="LABEL">text</error>
        Format Example:
            Text: He go to school.
            Tagged text: He <error type="GRAMMAR">go</error> to school.
        
        IMPORTANT: You must output the entire text, including non-tagged parts.
    """),
    "multigec": textwrap.dedent("""
        Rewrite the whole input text with XML tags around relevant spans. Identify grammatical errors in learner-written text and provide corrections.
        There are three error types:
        - R (Replace): wrong word, needs replacement
        - M (Missing): word missing, needs insertion  
        - U (Unnecessary): extra word, needs deletion
    
        Format: <error type="TYPE" correction="FIX">text</error>
    
        For R: type is error category (VERB, NOUN, DET, PREP, SPELL, PUNCT, WO, OTHER)
        For M and U: type is just "M" or "U"
    
        Example 1 (Replace - VERB):
            Text: He go to school.
            Tagged: He <error type="VERB" correction="went">go</error> to school.
    
        Example 2 (Missing):
            Text: I need advice about university.
            Tagged: I need <error type="M" correction="some"></error>advice about <error type="M" correction="the"></error>university.
    
        Example 3 (Unnecessary):
            Text: She is very much happy.
            Tagged: She is <error type="U" correction="">very</error> much happy.
    
        Example 4 (Multiple):
            Text: She dont like the apples
            Tagged: She <error type="VERB" correction="doesn't">dont</error> like <error type="U" correction="">the</error> apples<error type="M" correction="."></error>
    
        Only tag errors. Keep correct text unchanged.
                                
        IMPORTANT: You must output the entire text, including non-tagged parts.
    """),
    "wmt": textwrap.dedent("""
    Compare the source text with the translation and identify translation errors.
    Rewrite the TRANSLATION with XML tags around errors.
    
    Error Severity:
    - 0 = Minor error (understandable, small mistake)
    - 1 = Major error (changes meaning, serious mistake)
    
    Format: <error severity="LEVEL">text</error>
    
    Example 1:
        Source: "The house is very big."
        Translation: "Das Haus ist große."
        Tagged: Das Haus ist <error severity="1">große</error>.
        (Wrong adjective form - major error)
    
    Example 2:
        Source: "I like cats"
        Translation: "Ich mag Katzen sehr"
        Tagged: Ich mag Katzen <error severity="0">sehr</error>
        (Extra word - minor error, meaning still clear)
    
    Only tag errors in the TRANSLATION. Keep correct words unchanged.
    
    IMPORTANT: You must output the entire translation text, including non-tagged parts.
"""),
    "default": textwrap.dedent("""
        Rewrite the whole input text with XML tags around relevant spans.
        Format: <entity type="LABEL">text</entity>
        Format Example:
            Text: Apple was founded.      
            Tagged text: <entity type="ORG">Apple</entity> was founded.
        
        IMPORTANT: You must output the entire text, including non-tagged parts.
    """),
}


class XMLSpanLabeler(SpanLabeler):
    name: str = "xml"

    def parse_response(self, entry: dict) -> List[Dict]:
        response = entry["response"]
        original = entry["text"]
        results = []
        tag_pattern = r"<(\w+)([^>]*)>([^<]+)</\1>"

        original_pos = 0
        response_pos = 0
        used_positions = set()

        for match in re.finditer(tag_pattern, response):
            tag_name = match.group(1)
            attributes = match.group(2)
            span_text = match.group(3)

            before = response[response_pos : match.start()]
            actual_before = re.sub(r"<[^>]+>", "", before)
            original_pos += len(actual_before)

            type_match = re.search(r'type="([^"]+)"', attributes)
            label = type_match.group(1) if type_match else tag_name.upper()

            expected_start = original_pos
            expected_end = expected_start + len(span_text)

            if (
                expected_end <= len(original)
                and original[expected_start:expected_end] == span_text
                and expected_start not in used_positions
            ):
                results.append(
                    {
                        "text": span_text,
                        "label": label,
                        "start": expected_start,
                        "end": expected_end,
                    }
                )
                used_positions.add(expected_start)
            else:
                occurrences = []
                search_pos = 0
                while True:
                    idx = original.find(span_text, search_pos)
                    if idx == -1:
                        break
                    if idx not in used_positions:
                        occurrences.append(idx)
                    search_pos = idx + 1

                if occurrences:
                    best_pos = min(occurrences, key=lambda x: abs(x - expected_start))

                    results.append(
                        {
                            "text": span_text,
                            "label": label,
                            "start": best_pos,
                            "end": best_pos + len(span_text),
                        }
                    )
                    used_positions.add(best_pos)

            original_pos = expected_end
            response_pos = match.end()

        return results
