"""Systematized data parsers for converting various datasets to span labeling format

All parsers convert data to a standardized format:
{
    "text": str,
    "spans": [{"text": str, "label": str, "start": int, "end": int}],
    ...additional fields
}
"""

import json
import re
from pathlib import Path
from typing import Any


def save_data(path: str | Path, data: list[dict[str, Any]]) -> None:
    """Save data to JSON file with pretty formatting"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(data)} examples to {path}")


def load_json(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON data from file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL data from file"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class SyntheticParser:
    """Parser for synthetic span labeling datasets"""

    @staticmethod
    def parse(input_path: str, output_path: str) -> None:
        """Convert synthetic dataset to span labeling format

        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
        """
        data = load_json(input_path)
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
                            {
                                "text": span_text,
                                "label": "",
                                "start": start,
                                "end": end,
                            }
                        )

                formatted_data.append(
                    {
                        "text": text,
                        "instruction": task_instruction,
                        "spans": spans,
                    }
                )

        save_data(output_path, formatted_data)


class UniversalNERParser:
    """Parser for Universal NER datasets in IOB2 format"""

    @staticmethod
    def parse(input_path: str, output_path: str) -> None:
        """Convert Universal NER IOB2 file to span labeling format

        Args:
            input_path: Path to .iob2 file
            output_path: Path to output JSON file
        """
        input_path = Path(input_path)
        raw_text = input_path.read_text(encoding="utf-8").strip()
        raw_docs = re.split(r"\n\t?\n", raw_text)

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

                # Normalize tags
                if "OTH" in tag or tag == "B-O":
                    tag = "O"
                tags.append(tag)

            # Reconstruct text with spaces
            text = " ".join(tokens)

            # Convert BIO tags to spans
            spans = UniversalNERParser._bio_to_spans(tokens, tags)

            if text and spans:
                examples.append({"text": text, "spans": spans})

        save_data(output_path, examples)

    @staticmethod
    def _bio_to_spans(tokens: list[str], tags: list[str]) -> list[dict[str, Any]]:
        """Convert BIO tags to character-level spans"""
        spans = []
        current_span = None
        char_pos = 0

        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                # Start new span
                if current_span:
                    spans.append(current_span)

                label = tag[2:]  # Remove 'B-' prefix
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

        # Add final span if exists
        if current_span:
            spans.append(current_span)

        return spans


class MultiGECParser:
    """Parser for MultiGEC grammatical error correction datasets"""

    @staticmethod
    def parse(input_path: str, output_path: str) -> None:
        """Convert MultiGEC M2 file to span labeling format

        Args:
            input_path: Path to .m2 file
            output_path: Path to output JSON file
        """
        input_path = Path(input_path)
        raw_text = input_path.read_text(encoding="utf-8").strip()
        raw_docs = re.split(r"\n\t?\n", raw_text)

        examples = []
        for doc in raw_docs:
            text = ""
            spans = []

            for line in doc.split("\n"):
                if line.startswith("S "):
                    text = line[2:].strip()

                elif line.startswith("A "):
                    if not text:
                        continue

                    line = line.lstrip("A ").strip()
                    parts = line.split("|||")
                    if len(parts) < 6:
                        continue

                    indices, error_type, correction, _, _, annotator_id = parts
                    error_action = error_type.split(":")[0]

                    if error_action == "noop":
                        continue

                    start_token, end_token = map(int, indices.split())
                    tokens = text.split()

                    # Calculate character positions
                    start_char = sum(len(t) + 1 for t in tokens[:start_token])
                    end_char = max(0, sum(len(t) + 1 for t in tokens[:end_token]) - 1)

                    # Handle missing tokens (insertion errors)
                    if error_action == "M":
                        end_char = start_char

                    span = {
                        "text": text[start_char:end_char],
                        "label": error_action,
                        "correction": correction,
                        "start": start_char,
                        "end": end_char,
                    }
                    spans.append(span)

            if text and spans:
                examples.append({"text": text, "spans": spans})

        save_data(output_path, examples)


class WMTParser:
    """Parser for WMT translation evaluation datasets"""

    @staticmethod
    def parse(
        inputs_path: str,
        outputs_path: str,
        annotations_path: str,
        output_path: str,
        annotator_group: int = 0,
    ) -> None:
        """Convert WMT evaluation data to span labeling format

        Args:
            inputs_path: Path to inputs JSONL file
            outputs_path: Path to outputs JSONL file
            annotations_path: Path to annotations JSONL file
            output_path: Path to output JSON file
            annotator_group: Which annotator group to use (default: 0)
        """
        # Extract language pair from outputs filename
        outputs_path = Path(outputs_path)
        languages = outputs_path.stem

        # Load all data
        inputs = load_jsonl(inputs_path)
        outputs = load_jsonl(outputs_path)
        annotations = load_jsonl(annotations_path)

        # Index by keys for fast lookup
        inputs_dict = {item["orig_example_idx"]: item for item in inputs}
        outputs_dict = {
            (
                item["orig_example_idx"],
                item["setup_id"],
                item["dataset"],
                item["split"],
            ): item
            for item in outputs
        }

        # Filter and combine annotations
        examples = []
        for annotation in annotations:
            idx = annotation["orig_example_idx"]
            setup_id = annotation["setup_id"]
            dataset = annotation["dataset"]
            split = annotation["split"]

            # Filter by annotator group
            if annotation["metadata"]["annotator_group"] != annotator_group:
                continue

            # Check all required data exists
            key = (idx, setup_id, dataset, split)
            if (
                idx not in inputs_dict
                or key not in outputs_dict
                or not annotation["annotations"]
            ):
                continue

            input_text = inputs_dict[idx]["src"]
            output_text = outputs_dict[key]["output"]

            # Convert annotations to spans
            spans = []
            for ann in annotation["annotations"]:
                label = str(ann.get("type", ""))
                if label:
                    label = "MAJOR" if label == "1" else "MINOR"

                spans.append(
                    {
                        "text": ann["text"],
                        "label": label,
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

        save_data(output_path, examples)


if __name__ == "__main__":
    # Process WMT datasets
    print("\n=== Processing WMT datasets ===")
    wmt_output_dir = Path("<your_path>/span_labeling/data/wmt")
    wmt_inputs_paths = {
        "news": Path(
            "<your_path>/span_labeling_data/mt-eval/inputs-mt-eval_wmt24-news/"
        ),
        "literary": Path(
            "<your_path>/span_labeling_data/mt-eval/inputs-wmt24-literary"
        ),
        "social": Path("<your_path>/span_labeling_data/mt-eval/inputs-wmt24-social/"),
    }
    wmt_outputs_paths = {
        "news": Path(
            "<your_path>/span_labeling_data/mt-eval/outputs-mt-eval_wmt24-news/"
        ),
        "literary": Path(
            "<your_path>/span_labeling_data/mt-eval/outputs-wmt24-literary/"
        ),
        "social": Path("<your_path>/span_labeling_data/mt-eval/outputs-wmt24-social/"),
    }

    wmt_annotations_path = "<your_path>/span_labeling_data/mt-eval/llm-span-annotators span-annotation main annotations-human_mt-eval/annotations.jsonl"

    # Match input and output files by language pair
    wmt_datasets = {}
    for domain in wmt_inputs_paths.keys():
        wmt_inputs_path = wmt_inputs_paths[domain]
        wmt_outputs_path = wmt_outputs_paths[domain]
        wmt_datasets[domain] = {}

        for file_path in wmt_inputs_path.glob("*.jsonl"):
            languages = file_path.stem
            wmt_datasets[domain][languages] = {"inputs": file_path, "outputs": None}
        for file_path in wmt_outputs_path.glob("*.jsonl"):
            languages = file_path.stem
            if languages in wmt_datasets[domain]:
                wmt_datasets[domain][languages]["outputs"] = file_path

    for domain, wmt_language_pairs in wmt_datasets.items():
        for languages, paths in wmt_language_pairs.items():
            if paths["inputs"] and paths["outputs"]:
                WMTParser.parse(
                    inputs_path=paths["inputs"],
                    outputs_path=paths["outputs"],
                    annotations_path=wmt_annotations_path,
                    output_path=wmt_output_dir / f"wmt-{languages}-{domain}.json",
                )

    # # Process MultiGEC dataset
    # print("\n=== Processing MultiGEC dataset ===")
    # multigec_data_path = Path("<your_path>/span_labeling_data/multigec/write-and-improve-corpus-2024-v2/multigec-2025-files/local_eval/ref/en-writeandimprove2024-ref1-dev.m2")
    # multigec_output_dir = Path("<your_path>/span_labeling/data/multigec")
    # MultiGECParser.parse(
    #     input_path=multigec_data_path,
    #     output_path=multigec_output_dir / "multigec_en.json"
    # )

    # # Process Universal NER datasets
    # print("\n=== Processing Universal NER datasets ===")
    # universal_ner_data_path = Path("<your_path>/span_labeling_data/uner-20231114-092426")
    # universal_ner_output_dir = Path("<your_path>/span_labeling/data/universal_ner")

    # for uner_directory in universal_ner_data_path.glob("*"):
    #     if uner_directory.is_dir():
    #         for uner_file in uner_directory.glob("*.iob2"):
    #             if "test" in uner_file.stem:
    #                 UniversalNERParser.parse(
    #                     input_path=uner_file,
    #                     output_path=universal_ner_output_dir / f"uner_{uner_directory.stem}_{uner_file.stem.split('-')[0]}.json"
    #                 )

    # # Process synthetic datasets
    # print("\n=== Processing synthetic datasets ===")
    # synthetic_datasets = [
    #     ("character_dataset.json", "english_character_synthetic_data.json"),
    #     ("german_word_dataset.json", "german_word_synthetic_data.json"),
    #     ("english_word_dataset.json", "english_word_synthetic_data.json"),
    #     ("non_overlapping_english_word_dataset.json", "english_non_overlapping_word_synthetic_data.json"),
    # ]

    # for input_file, output_file in synthetic_datasets:
    #     input_path = Path(f"<your_path>/input-labeling/{input_file}")
    #     output_path = Path(f"<your_path>/span_labeling/data/synthetic/{output_file}")
    #     if input_path.exists():
    #         SyntheticParser.parse(
    #             input_path=input_path,
    #             output_path=output_path
    #         )

    # print("\n=== All datasets processed ===\n")
