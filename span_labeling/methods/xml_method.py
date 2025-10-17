import re
import textwrap
from typing import List, Dict
from span_labeling.base import SpanLabeler

format: dict[str, str] = {
    "ner": textwrap.dedent("""
        Rewrite the whole input text with XML tags around entities.
        Only use the following labels: PERSON, ORG, LOC
        Format: <entity type="LABEL">text</entity>
        Example:
            Text: Apple is in Cupertino.  
            Tagged text: <entity type="ORG">Apple</entity> is in <entity type="LOC">Cupertino</entity>.
    """),
    "synthetic": textwrap.dedent("""
        Rewrite the whole input text with XML tags around matching patterns.
        Format: <match>text</match>
        Format Example:
            Text: The cat sat on the mat.
            Tagged text: The <match>cat</match> sat on the mat.
    """),
    "error": textwrap.dedent("""
        Rewrite the whole input text with XML tags inserted around errors.
        Only tag the incorrect words.
        Only use the following labels: GRAMMAR, SPELLING, PUNCTUATION
        Format: <error type="LABEL">text</error>
        Format Example:
            Text: He go to school.
            Tagged text: He <error type="GRAMMAR">go</error> to school.
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
    """),
    "default": textwrap.dedent("""
        Rewrite the whole input text with XML tags around relevant spans.
        Format: <entity type="LABEL">text</entity>
        Format Example:
            Text: Apple was founded.      
            Tagged text: <entity type="ORG">Apple</entity> was founded.
    """),
}


class XMLSpanLabeler(SpanLabeler):

    def format_prompt(self, entry: dict) -> str:
        return f"""Task: {entry['instruction']}

Original text: "{entry['text']}"

{format.get(entry.get('key', None), format['default'])}

Tagged text:"""

    def parse_response(self, entry: dict) -> List[Dict]:
        results = []
        
        # Try different XML patterns
        patterns = [
            r'<entity type="([^"]+)">([^<]+)</entity>',
            r'<error type="([^"]+)">([^<]+)</error>',
            r'<match>([^<]+)</match>',
            r'<(\w+)>([^<]+)</\1>',  # <PERSON>Steve Jobs</PERSON>
            r'<error type="([^"]+)" correction="([^"]+)">([^<]+)</error>',  # <error type="R" correction="the">teh</error>
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, entry["response"]):
                label = match.group(1)
                span_text = match.group(2)
                
                idx = entry["text"].find(span_text)
                if idx != -1:
                    results.append({
                        'text': span_text,
                        'label': label,
                        'start': idx,
                        'end': idx + len(span_text)
                    })
        
        return results
    



"ajlskd;fj adslkfjl;asdj fl;ajdsl;fjl content content sdaklfjasl;dkfj kadsjflk;js adlfk;jsl kjadf content2"
"ajlskd;fj adslkfjl;asdajsadfsadf fl;ajdsl;fjl <...>content</...> content sdaklfjasl;dkfj kadsjflk;js adlfk;jsl kjadf <...>content2</...>"
