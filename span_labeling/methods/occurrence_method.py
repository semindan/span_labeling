# methods/json_occurrence_method.py
from .json_method import JSONSpanLabeler
from span_labeling.registry import EXAMPLES


class JSONOccurrenceSpanLabeler(JSONSpanLabeler):
    def format_prompt(self, entry: dict) -> str:
        examples = EXAMPLES.get(entry.get('key', ''), {}).get('occurrence', '')
        if examples:
            examples = "Example:\n" + examples + "\n\n"

        return f"""Task: {entry['instruction']}

Text: "{entry['text']}"

{examples}

Return a JSON list. Each item must have:
- "text": exact span or pattern from input
- "occurrence": which occurrence (1 for first, 2 for second, etc.)


JSON output:"""
    
    def parse_response(self, entry: dict) -> list[dict]:
        spans = super().parse_response(entry)
        
        # Use occurrence to disambiguate
        for span in spans:
            occ = span.get('occurrence', 1)
            # Find the nth occurrence
            start = -1
            for i in range(occ):
                start = entry["text"].find(span['text'], start + 1)
            
            if start != -1:
                span['start'] = start
                span['end'] = start + len(span['text'])
        
        return spans