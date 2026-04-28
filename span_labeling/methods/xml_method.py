import re
from typing import List, Dict
from span_labeling.methods.span_labeler import SpanLabeler


class XMLSpanLabeler(SpanLabeler):
    key: str = "xml"

    def parse_response(self, entry: dict) -> List[Dict]:
        response = entry["response"]
        original = entry["text"]
        results = []
        tag_pattern = r"<(\w+)([^>]*)>([^<]+)</\1>"

        response = entry["response"]
        if "Output:" in response:
            response = response.split("Output:")[-1].strip()

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
