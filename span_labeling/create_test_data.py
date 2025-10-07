import json
from pathlib import Path

ner_data = [
    {
        "text": "Apple Inc. was founded by Steve Jobs in Cupertino.",
        "spans": [
            {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
            {"text": "Steve Jobs", "label": "PERSON", "start": 26, "end": 36},
            {"text": "Cupertino", "label": "LOC", "start": 40, "end": 49}
        ]
    },
    {
        "text": "Microsoft CEO Satya Nadella announced the deal in Seattle.",
        "spans": [
            {"text": "Microsoft", "label": "ORG", "start": 0, "end": 9},
            {"text": "Satya Nadella", "label": "PERSON", "start": 14, "end": 27},
            {"text": "Seattle", "label": "LOC", "start": 51, "end": 58}
        ]
    },
    {
        "text": "Google acquired YouTube for $1.65 billion in 2006.",
        "spans": [
            {"text": "Google", "label": "ORG", "start": 0, "end": 6},
            {"text": "YouTube", "label": "ORG", "start": 16, "end": 23}
        ]
    },
    {
        "text": "Amazon founder Jeff Bezos stepped down as CEO.",
        "spans": [
            {"text": "Amazon", "label": "ORG", "start": 0, "end": 6},
            {"text": "Jeff Bezos", "label": "PERSON", "start": 15, "end": 25}
        ]
    },
    {
        "text": "Tesla is building a new factory in Berlin, Germany.",
        "spans": [
            {"text": "Tesla", "label": "ORG", "start": 0, "end": 5},
            {"text": "Berlin", "label": "LOC", "start": 35, "end": 41},
            {"text": "Germany", "label": "LOC", "start": 43, "end": 50}
        ]
    },
    {
        "text": "Mark Zuckerberg founded Facebook in his Harvard dorm room.",
        "spans": [
            {"text": "Mark Zuckerberg", "label": "PERSON", "start": 0, "end": 15},
            {"text": "Facebook", "label": "ORG", "start": 24, "end": 32},
            {"text": "Harvard", "label": "ORG", "start": 40, "end": 47}
        ]
    },
    {
        "text": "The iPhone was designed by Apple in California.",
        "spans": [
            {"text": "Apple", "label": "ORG", "start": 27, "end": 32},
            {"text": "California", "label": "LOC", "start": 36, "end": 46}
        ]
    },
    {
        "text": "Elon Musk runs both Tesla and SpaceX.",
        "spans": [
            {"text": "Elon Musk", "label": "PERSON", "start": 0, "end": 9},
            {"text": "Tesla", "label": "ORG", "start": 20, "end": 25},
            {"text": "SpaceX", "label": "ORG", "start": 30, "end": 36}
        ]
    },
    {
        "text": "OpenAI was co-founded by Sam Altman in San Francisco.",
        "spans": [
            {"text": "OpenAI", "label": "ORG", "start": 0, "end": 6},
            {"text": "Sam Altman", "label": "PERSON", "start": 25, "end": 35},
            {"text": "San Francisco", "label": "LOC", "start": 39, "end": 52}
        ]
    },
    {
        "text": "Netflix started as a DVD rental service in Los Gatos.",
        "spans": [
            {"text": "Netflix", "label": "ORG", "start": 0, "end": 7},
            {"text": "Los Gatos", "label": "LOC", "start": 44, "end": 53}
        ]
    }
]

# for entry in ner_data:
#     for span in entry['spans']:
#         assert entry['text'][span['start']:span['end']] == span['text'], f"Span text mismatch: {entry['text'][span['start']:span['end']]} != {span['text']}"

# Save
output_dir = Path('data/custom')
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / 'ner_test.json', 'w') as f:
    json.dump(ner_data, f, indent=2)

print(f"Created {len(ner_data)} test examples")


# create_error_data.py
import json

error_data = [
    {
        "text": "She don't know nothing about the issue.",
        "spans": [
            {"text": "don't", "label": "GRAMMAR", "start": 4, "end": 9},
            {"text": "nothing", "label": "GRAMMAR", "start": 15, "end": 22}
        ]
    },
    {
        "text": "The data are showing that we was wrong.",
        "spans": [
            {"text": "was", "label": "GRAMMAR", "start": 29, "end": 32}
        ]
    },
    {
        "text": "Him and me went to the store yesterday.",
        "spans": [
            {"text": "Him", "label": "GRAMMAR", "start": 0, "end": 3},
            {"text": "me", "label": "GRAMMAR", "start": 8, "end": 10}
        ]
    },
    {
        "text": "I seen the movie last night.",
        "spans": [
            {"text": "seen", "label": "GRAMMAR", "start": 2, "end": 6}
        ]
    },
    {
        "text": "There going to their house over they're.",
        "spans": [
            {"text": "There", "label": "SPELLING", "start": 0, "end": 5},
            {"text": "they're", "label": "SPELLING", "start": 33, "end": 40}
        ]
    }
]


# for entry in error_data:
#     for span in entry['spans']:
#         assert entry['text'][span['start']:span['end']] == span['text'], f"Span text mismatch: {entry['text'][span['start']:span['end']]} != {span['text']}"

with open(output_dir / 'error_test.json', 'w') as f:
    json.dump(error_data, f, indent=2)

print(f"Created {len(error_data)} error test examples")


# parse_synthetic.py
import json

input_path = "/home/semin/personal_work_ms/input-labeling/word_dataset.json"

with open(input_path, 'r', encoding='utf-8') as f:
    synthetic_data = json.load(f)

# Convert to your format
formatted_data = []
for item in synthetic_data:  # Start with 5 examples
    text = item['input_sequence']
    
    # We'll test with the simplest queries first
    for query_idx, query in enumerate(item['queries'][:2]):  # Just 2 queries per text
        task_instruction = query['natural_language']
        
        # Convert answer indices to spans
        spans = []
        for span_indices in item['answers'][query_idx]:
            if len(span_indices) == 2:
                start, end = span_indices
                start = start if start == 0 else start + 1
                span_text = text[start:end]
                spans.append({
                    'text': span_text,
                    'label': "",
                    'start': start,
                    'end': end
                })
        
        formatted_data.append({
            'text': text,
            'instruction': task_instruction,
            'spans': spans
        })


with open(output_dir / 'synthetic_test.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)

print(f"Created {len(formatted_data)} synthetic examples")