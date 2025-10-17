import re
from typing import List, Dict
from span_labeling.base import SpanLabeler

class XMLSpanLabeler(SpanLabeler):
    def format_prompt(self, entry: dict) -> str:
        return f"""Task: {entry['instruction']}

Original text: "{entry['text']}"

Rewrite the text with XML tags around relevant spans.
Format: <entity type="LABEL">text</entity>

Example: <entity type="ORG">Apple</entity> was founded by <entity type="PERSON">Steve Jobs</entity>.

Tagged text:"""

    def parse_response(self, entry: dict) -> List[Dict]:
        results = []
        
        # Try different XML patterns
        patterns = [
            r'<entity type="([^"]+)">([^<]+)</entity>',
            r'<(\w+)>([^<]+)</\1>',  # <PERSON>Steve Jobs</PERSON>
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