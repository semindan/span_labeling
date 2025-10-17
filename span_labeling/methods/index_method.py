import re
from typing import List, Dict
from span_labeling.base import SpanLabeler

class IndexSpanLabeler(SpanLabeler):
    def format_prompt(self, entry: dict) -> str:
        return f"""Task: {entry['instruction']}

Text: "{entry['text']}"

Output each span as: [start:end] = LABEL
Where start and end are character positions (0-indexed).

Example for "Apple Inc was founded":
[0:9] = ORG   (this captures "Apple Inc")

Output:"""
    
    def parse_response(self, entry: dict) -> List[Dict]:
        results = []
        
        # Pattern: [15:27] = PERSON
        pattern = r'\[(\d+):(\d+)\]\s*=\s*(\S+)'
        
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
    
    def output_format(self) -> str:
        return "[start:end] = LABEL"