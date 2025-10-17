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


# # for entry in error_data:
# #     for span in entry['spans']:
# #         assert entry['text'][span['start']:span['end']] == span['text'], f"Span text mismatch: {entry['text'][span['start']:span['end']]} != {span['text']}"

with open(output_dir / 'error_test.json', 'w') as f:
    json.dump(error_data, f, indent=2)

print(f"Created {len(error_data)} error span correction examples")


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


from pathlib import Path
import re

import pandas as pd

def read_uner(file_path):
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)

    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            # ignore comments
            if line.startswith('#'): continue
            tok_id, token, tag = line.split('\t')[:3]
            tokens.append(token)
            if "OTH" in tag or tag == "B-O":
                tag = "O"
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    # return token_docs, tag_docs

    train_dict = {"tokens": token_docs, "ner_tags": tag_docs}

    return train_dict

def read_uner_to_spans(file_path):
    """Convert CoNLL-U format to character-based span format"""
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    
    examples = []
    for doc in raw_docs:
        tokens = []
        tags = []
        
        for line in doc.split('\n'):
            if line.startswith('#'): 
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            tok_id, token, tag = parts[:3]
            tokens.append(token)
            if "OTH" in tag or tag == "B-O":
                tag = "O"
            tags.append(tag)
        
        # Reconstruct text with spaces
        text = ' '.join(tokens)
        
        # Convert BIO tags to spans
        spans = []
        current_span = None
        char_pos = 0
        
        for token, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                # Start new span
                if current_span:
                    spans.append(current_span)
                label = tag[2:]  # Remove 'B-'
                current_span = {
                    'text': token,
                    'label': label,
                    'start': char_pos,
                    'end': char_pos + len(token)
                }
            elif tag.startswith('I-') and current_span:
                # Continue span
                current_span['text'] += ' ' + token
                current_span['end'] = char_pos + len(token)
            else:
                # O tag - finish current span if any
                if current_span:
                    spans.append(current_span)
                    current_span = None
            
            char_pos += len(token) + 1  # +1 for space
        
        # Don't forget last span
        if current_span:
            spans.append(current_span)
        
        if text and spans:  # Only add if has content
            examples.append({
                'text': text,
                'spans': spans
            })
    
    return examples







# formatted_data = read_uner_to_spans("/home/semin/personal_work_ms/span_labeling_data/uner-20231114-092426/en_ewt/en_ewt-ud-dev.iob2")

# formatted_data = formatted_data[:50]

# with open(output_dir / 'ner_en_test.json', 'w', encoding='utf-8') as f:
#     json.dump(formatted_data, f, indent=2, ensure_ascii=False)

# print(f"Created {len(formatted_data)} universal ner examples")



# /home/semin/personal_work_ms/span_labeling_data/multigec/write-and-improve-corpus-2024-v2/user-prompt-final-versions/en-writeandimprove2024-final-versions-dev-sentences.m2
# def parse_multigec(file_path):
#     file_path = Path(file_path)
#     raw_text = file_path.read_text().strip()
#     raw_docs = re.split(r'\n\t?\n', raw_text)

#     examples = []

#     for i, doc in enumerate(raw_docs[:300]):


#         text = ""
#         spans = []

#         for line in doc.split('\n'):


#             if line.startswith('S '):

#                 text = line[2:].strip()


#             elif line.startswith('A '):
#                 if not text:
#                     raise ValueError("No text line found before annotations")
                
#                 line = line.lstrip('A ').strip()
#                 indices, error_type, correction, _, _, annotator_id  = line.split('|||')

#                 error_action, *error_category = error_type.split(':')
#                 start_token, end_token = map(int, indices.split())
#                 # print(
#                 #     start_token, end_token, error_type, correction
#                 # )

#                 tokens = text.split()

#                 start_char = sum(len(t) + 1 for t in tokens[:start_token])
#                 end_char = max(0, sum(len(t) + 1 for t in tokens[:end_token]) - 1) # minus trailing space 

#                 if error_action == "R":
#                     # print("TEXT:", text)
#                     # print(f"REPLACE {text[start_char:end_char]} WITH {correction}")
#                     # print("ORI:", text)
#                     # text_new = text[:start_char] + correction + text[end_char:]
#                     # print("NEW:", text_new)       
#                     pass
#                 elif error_action == "M":
#                     # print("TEXT:", text)
#                     # print(start_char, end_char, start_token, end_token)
#                     # print(f"MISSING {correction} BEFORE {tokens[start_token:start_token+1]}")
#                     # print("ORI:", text)
#                     # text_new = text[:start_char] + correction + (" " if end_char == 0 else "") + text[end_char:]
#                     # print("NEW:", text_new)
#                     end_char = start_char 
#                     pass
#                 elif error_action == "U":
#                     # print("TEXT:", text)
#                     # print(start_char, end_char, start_token, end_token)
#                     # print(f"UNNECESSARY {text[start_char:end_char]}")
#                     # print("ORI:", text)
#                     # text_new = text[:start_char] + text[end_char:]
#                     # print("NEW:", text_new)       
#                     pass
#                 elif error_action == "noop":
#                     continue


#                 current_span = {
#                     "text": text[start_char:end_char],
#                     "label": error_action,
#                     "correction": correction,
#                     "start": start_char,
#                     "end": end_char,
#                 }

#                 spans.append(current_span)
                


#         if text and spans:
#             examples.append({
#                 'text': text,
#                 'spans': spans
#             })

#     return examples


            














# formatted_data = parse_multigec("/home/semin/personal_work_ms/span_labeling_data/multigec/write-and-improve-corpus-2024-v2/user-prompt-final-versions/en-writeandimprove2024-final-versions-dev-essays.m2")

# with open(output_dir / 'multigec.json', 'w', encoding='utf-8') as f:
#     json.dump(formatted_data, f, indent=2, ensure_ascii=False)

# print(f"Created {len(formatted_data)} multigec error span correction examples")
import json


def parse_wmt(inputs_path, outputs_path, annotations_path):
    outputs_path = Path(outputs_path)
    annotations_path = Path(annotations_path)

    inputs = []
    with open(inputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))

    inputs_dict = {item["orig_example_idx"]: item for item in inputs}
    

    outputs = []
    with open(outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line))

    outputs_dict = {(item["orig_example_idx"], item["setup_id"], item["dataset"], item["split"]): item for item in outputs}

    annotations = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    annotations_final = []

    # ids = {}

    for annotation in annotations:
        idx = annotation["orig_example_idx"]
        setup_id = annotation["setup_id"]
        dataset = annotation["dataset"]
        split = annotation["split"]

        annotator_group = annotation["metadata"]["annotator_group"]

        if annotator_group != 0:
            continue

        if idx in inputs_dict and (idx, setup_id, dataset,split) in outputs_dict and annotation["annotations"]:

            # if (idx, setup_id, dataset, split) in ids:
            #     print(ids[(idx, setup_id, dataset, split)])
            #     print("DUPLICATE ANNOTATION:")
            #     print(annotation)
            #     breakpoint()
            # else:
            #     ids[(idx, setup_id, dataset, split)] = annotation

            combined = {
                "orig_example_idx": idx,
                "input": inputs_dict[idx],
                "output": outputs_dict[(idx, setup_id, dataset, split)],
                "annotation": annotation,
            }
            annotations_final.append(combined)

    examples = []

    for item in annotations_final:
        input_text = item["input"]["src"]
        output_text = item["output"]["output"]
        _annotations = item["annotation"]["annotations"]

        spans = []
        for ann in _annotations:
            spans.append({
                "text": ann["text"],
                "label": ann.get("type", ""),
                "start": ann["start"],
                "end": ann["start"] + len(ann["text"]),
            })

        examples.append({
            "source" : input_text,
            "text" : output_text,
            "spans": spans,
        })




    return examples


wmt_inputs_path = "/home/semin/personal_work_ms/span_labeling_data/mt-eval/inputs-mt-eval_wmt24-news/en-cs.jsonl"
wmt_outputs_path = "/home/semin/personal_work_ms/span_labeling_data/mt-eval/outputs-mt-eval_wmt24-news/en-cs.jsonl"
wmt_annotations_path = "/home/semin/personal_work_ms/span_labeling_data/mt-eval/llm-span-annotators span-annotation main annotations-human_mt-eval/annotations.jsonl"


annotations = parse_wmt(wmt_inputs_path, wmt_outputs_path, wmt_annotations_path)


with open(output_dir / 'wmt-cs.json', 'w', encoding='utf-8') as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)

print(f"Created {len(annotations)} wmt-cs examples")
