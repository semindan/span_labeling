import json
import re
from typing import List, Dict
from span_labeling.base import SpanLabeler

class JSONSpanLabeler(SpanLabeler):
    def format_prompt(self, entry: dict) -> str:
        return f"""Task: {entry['instruction']}

Text: "{entry['text']}"

Return a JSON list. Each item should have:
- "text": the exact text span from the input
- "label": the category

Example: [{{"text": "Apple", "label": "ORG"}}]

JSON output:"""

    def parse_response(self, entry: dict) -> List[Dict]:
        # Find JSON in response
        try:
            # Look for [...] pattern
            match = re.search(r'\[.*?\]', entry["response"], re.DOTALL)
            if match:
                json_str = match.group()
                data = json.loads(json_str)
                
                results = []
                for item in data:
                    span_text = item.get('text', '')
                    label = item.get('label', '')
                    
                    # Find where this text appears
                    idx = entry["text"].find(span_text)
                    if idx != -1:
                        results.append({
                            'text': span_text,
                            'label': label,
                            'start': idx,
                            'end': idx + len(span_text)
                        })
                
                return results
        except:
            pass
        
        return []