import json
from pathlib import Path
import re


def save_data(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} examples to {path}")


def load_data(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_synthetic(path: str, output_path: str) -> None:
    data = load_data(path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    formatted_data = []
    for item in data:
        text = item["input_sequence"]

        for query_idx, query in enumerate(item["queries"]):
            task_instruction = query["natural_language"]

            spans = []
            for span_indices in item["answers"][query_idx]:
                if len(span_indices) == 2:
                    start, end = span_indices
                    start = start if start == 0 else start + 1
                    span_text = text[start:end]
                    spans.append(
                        {"text": span_text, "label": "", "start": start, "end": end}
                    )

            formatted_data.append(
                {"text": text, "instruction": task_instruction, "spans": spans}
            )

    save_data(output_path, formatted_data)


def prepare_universal_ner(file_path: str, output_dir: str) -> None:
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding="utf-8").strip()
    raw_docs = re.split(r"\n\t?\n", raw_text)

    language_dataset_spec = file_path.parent.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for doc in raw_docs:
        tokens = []
        tags = []

        for line in doc.split("\n"):
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            tok_id, token, tag = parts[:3]
            tokens.append(token)
            if "OTH" in tag or tag == "B-O":
                tag = "O"
            tags.append(tag)

        # Reconstruct text with spaces
        text = " ".join(tokens)

        # Convert BIO tags to spans
        spans = []
        current_span = None
        char_pos = 0

        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                # Start new span
                if current_span:
                    spans.append(current_span)
                label = tag[2:]  # Remove 'B-'
                current_span = {
                    "text": token,
                    "label": label,
                    "start": char_pos,
                    "end": char_pos + len(token),
                }
            elif tag.startswith("I-") and current_span:
                # Continue span
                current_span["text"] += " " + token
                current_span["end"] = char_pos + len(token)
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
            examples.append({"text": text, "spans": spans})

    output_path = output_dir / f"uner_{language_dataset_spec}.json"
    save_data(output_path.as_posix(), examples)


def prepare_multigec(file_path, output_dir):
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding="utf-8").strip()
    raw_docs = re.split(r"\n\t?\n", raw_text)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []

    for i, doc in enumerate(raw_docs):
        text = ""
        spans = []

        for line in doc.split("\n"):
            if line.startswith("S "):
                text = line[2:].strip()

            elif line.startswith("A "):
                if not text:
                    raise ValueError("No text line found before annotations")

                line = line.lstrip("A ").strip()
                indices, error_type, correction, _, _, annotator_id = line.split("|||")

                error_action, *error_category = error_type.split(":")
                start_token, end_token = map(int, indices.split())
                # print(
                #     start_token, end_token, error_type, correction
                # )

                tokens = text.split()

                start_char = sum(len(t) + 1 for t in tokens[:start_token])
                end_char = max(
                    0, sum(len(t) + 1 for t in tokens[:end_token]) - 1
                )  # minus trailing space

                if error_action == "R":
                    # print("TEXT:", text)
                    # print(f"REPLACE {text[start_char:end_char]} WITH {correction}")
                    # print("ORI:", text)
                    # text_new = text[:start_char] + correction + text[end_char:]
                    # print("NEW:", text_new)
                    pass
                elif error_action == "M":
                    # print("TEXT:", text)
                    # print(start_char, end_char, start_token, end_token)
                    # print(f"MISSING {correction} BEFORE {tokens[start_token:start_token+1]}")
                    # print("ORI:", text)
                    # text_new = text[:start_char] + correction + (" " if end_char == 0 else "") + text[end_char:]
                    # print("NEW:", text_new)
                    end_char = start_char
                    pass
                elif error_action == "U":
                    # print("TEXT:", text)
                    # print(start_char, end_char, start_token, end_token)
                    # print(f"UNNECESSARY {text[start_char:end_char]}")
                    # print("ORI:", text)
                    # text_new = text[:start_char] + text[end_char:]
                    # print("NEW:", text_new)
                    pass
                elif error_action == "noop":
                    continue

                current_span = {
                    "text": text[start_char:end_char],
                    "label": error_action,
                    "correction": correction,
                    "start": start_char,
                    "end": end_char,
                }

                spans.append(current_span)

        if text and spans:
            examples.append({"text": text, "spans": spans})

    save_data(Path(output_dir) / "multigec_en.json", examples)


# formatted_data = read_uner_to_spans("/home/semin/personal_work_ms/span_labeling_data/uner-20231114-092426/en_ewt/en_ewt-ud-dev.iob2")


# /home/semin/personal_work_ms/span_labeling_data/multigec/write-and-improve-corpus-2024-v2/user-prompt-final-versions/en-writeandimprove2024-final-versions-dev-sentences.m2


# formatted_data = parse_multigec("/home/semin/personal_work_ms/span_labeling_data/multigec/write-and-improve-corpus-2024-v2/user-prompt-final-versions/en-writeandimprove2024-final-versions-dev-essays.m2")


def prepare_wmt(wmt_inputs_path, wmt_outputs_path, wmt_annotations_path, output_dir):
    wmt_outputs_path = Path(wmt_outputs_path)
    languages = wmt_outputs_path.stem
    wmt_annotations_path = Path(wmt_annotations_path)

    inputs = []
    with open(wmt_inputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))

    inputs_dict = {item["orig_example_idx"]: item for item in inputs}

    outputs = []
    with open(wmt_outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line))

    outputs_dict = {
        (
            item["orig_example_idx"],
            item["setup_id"],
            item["dataset"],
            item["split"],
        ): item
        for item in outputs
    }

    annotations = []
    with open(wmt_annotations_path, "r", encoding="utf-8") as f:
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

        if (
            idx in inputs_dict
            and (idx, setup_id, dataset, split) in outputs_dict
            and annotation["annotations"]
        ):
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
            spans.append(
                {
                    "text": ann["text"],
                    "label": ann.get("type", ""),
                    "start": ann["start"],
                    "end": ann["start"] + len(ann["text"]),
                }
            )

        examples.append(
            {
                "source": input_text,
                "text": output_text,
                "spans": spans,
                "source_language": languages.split("-")[0],
                "target_language": languages.split("-")[1],
            }
        )

    save_data(Path(output_dir) / f"wmt-{languages}.json", examples)


# wmt_inputs_path = "/home/semin/personal_work_ms/span_labeling_data/mt-eval/inputs-mt-eval_wmt24-news/en-cs.jsonl"
# wmt_outputs_path = "/home/semin/personal_work_ms/span_labeling_data/mt-eval/outputs-mt-eval_wmt24-news/en-cs.jsonl"
# wmt_annotations_path = "/home/semin/personal_work_ms/span_labeling_data/mt-eval/llm-span-annotators span-annotation main annotations-human_mt-eval/annotations.jsonl"


# annotations = parse_wmt(wmt_inputs_path, wmt_outputs_path, wmt_annotations_path)


# with open(output_dir / 'wmt-cs.json', 'w', encoding='utf-8') as f:
#     json.dump(annotations, f, indent=2, ensure_ascii=False)

# print(f"Created {len(annotations)} wmt-cs examples")

if __name__ == "__main__":
    # wmt_output_dir = Path("/home/semin/personal_work_ms/span_labeling/data/wmt")
    # wmt_output_dir.mkdir(parents=True, exist_ok=True)
    # wmt_inputs_path = Path("/home/semin/personal_work_ms/span_labeling_data/mt-eval/inputs-mt-eval_wmt24-news/")
    # wmt_outputs_path = Path("/home/semin/personal_work_ms/span_labeling_data/mt-eval/outputs-mt-eval_wmt24-news/")
    # wmt_annotations_path = "/home/semin/personal_work_ms/span_labeling_data/mt-eval/llm-span-annotators span-annotation main annotations-human_mt-eval/annotations.jsonl"

    # wmt_data_path = Path("/home/semin/personal_work_ms/span_labeling_data/mt-eval/outputs-mt-eval_wmt24-news")
    # wmt_language_paths = defaultdict(lambda: {"inputs": None, "outputs": None})
    # for file_path in wmt_inputs_path.glob("*.jsonl"):
    #     languages = file_path.stem
    #     wmt_language_paths[languages]["inputs"] = file_path
    # for file_path in wmt_outputs_path.glob("*.jsonl"):
    #     languages = file_path.stem
    #     wmt_language_paths[languages]["outputs"] = file_path

    # for languages, paths in wmt_language_paths.items():
    #     prepare_wmt(
    #         wmt_inputs_path=paths["inputs"],
    #         wmt_outputs_path=paths["outputs"],
    #         wmt_annotations_path=wmt_annotations_path,
    #         output_dir=wmt_output_dir
    #     )

    # multigec_data_path = Path("/home/semin/personal_work_ms/span_labeling_data/multigec/write-and-improve-corpus-2024-v2/multigec-2025-files/local_eval/ref/en-writeandimprove2024-ref1-dev.m2")
    # multigec_output_dir = Path("/home/semin/personal_work_ms/span_labeling/data/multigec")

    # prepare_multigec(
    #     file_path=multigec_data_path,
    #     output_dir=multigec_output_dir
    # )

    # universal_ner_data_path = Path("/home/semin/personal_work_ms/span_labeling_data/uner-20231114-092426")
    # universal_ner_output_dir = Path("/home/semin/personal_work_ms/span_labeling/data/universal_ner")

    # for uner_directory in universal_ner_data_path.glob("*"):
    #     if uner_directory.is_dir():
    #         for uner_file in uner_directory.glob("*.iob2"):
    #             if "test" not in uner_file.stem:
    #                 continue

    #             prepare_universal_ner(
    #                 file_path=uner_file,
    #                 output_dir=universal_ner_output_dir
    #             )

    synthetic_english_character_data_path = Path(
        "/home/semin/personal_work_ms/input-labeling/character_dataset.json"
    )
    prepare_synthetic(
        path=synthetic_english_character_data_path.as_posix(),
        output_path="/home/semin/personal_work_ms/span_labeling/data/synthetic/english_character_synthetic_data.json",
    )

    synthetic_german_word_data_path = Path(
        "/home/semin/personal_work_ms/input-labeling/german_word_dataset.json"
    )
    prepare_synthetic(
        path=synthetic_german_word_data_path.as_posix(),
        output_path="/home/semin/personal_work_ms/span_labeling/data/synthetic/german_word_synthetic_data.json",
    )

    synthetic_english_word_data_path = Path(
        "/home/semin/personal_work_ms/input-labeling/english_word_dataset.json"
    )
    prepare_synthetic(
        path=synthetic_english_word_data_path.as_posix(),
        output_path="/home/semin/personal_work_ms/span_labeling/data/synthetic/english_word_synthetic_data.json",
    )
