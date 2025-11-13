import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).absolute().parent.parent.as_posix()


def load_prompts(prompts_dir: str = "span_labeling/prompts") -> dict:
    """Load all YAML files from prompts directory"""
    prompts = {}
    prompts_path = Path(PROJECT_ROOT) / prompts_dir

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    for yaml_file in prompts_path.glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        for prompt in data["prompts"]:
            prompts[prompt["method"]] = prompts.get(prompt["method"], {})
            prompts[prompt["method"]][prompt["dataset"]] = prompt

    return prompts


PROMPT_REGISTRY = load_prompts()


def get_prompt_config(method: str, dataset: str):
    return PROMPT_REGISTRY.get(method, {}).get(dataset, {})


def build_prompt(method: str, dataset: str, entry: dict) -> str:
    prompt_data = get_prompt_config(method, dataset)
    if not prompt_data:
        raise ValueError(f"No prompt found for {method}/{dataset}")

    sections = []

    sections.append(prompt_data["task"])
    sections.append("")

    if "format" in prompt_data:
        sections.append(f"Tag format: {prompt_data['format']}")

    if "labels" in prompt_data:
        sections.append(
            f"Use only the following Labels.\nLabels: {', '.join(prompt_data['labels'])}"
        )
    elif "label_dict" in prompt_data:
        labels_section = "Use only the following Labels.\nLabels:\n"
        for k, v in prompt_data["label_dict"].items():
            labels_section += f"{k} : {v}\n"
        sections.append(labels_section)

    if "examples" in prompt_data:
        examples_section = "\nExamples:\n"
        for i, example in enumerate(prompt_data["examples"], 1):
            examples_section += f"{i}. {example}\n"
        sections.append(examples_section)

    if "note" in prompt_data:
        sections.append(f"\n{prompt_data['note']}")

    if "instruction" in prompt_data:
        sections.append(f"\n{prompt_data['instruction']}")

    sections.append(f"{entry['model_input']}")
    # sections.append(f"\n{prompt_data['last_line']}")

    return "\n".join(sections)


def build_json_schema(method: str, dataset: str) -> dict:
    """Build a JSON Schema for structured outputs from the YAML prompt format.

    - Always requires the "text" field (string)
    - Adds additional fields present in the format example object for the dataset
      (e.g., label, correction). "label" accepts string or number to support
      datasets like WMT with numeric labels. If the prompt config provides a
      fixed set of labels (via "labels" or "label_dict"), constrain the
      "label" field with an enum of allowed values and a precise JSON type.
    - Disallows additional properties to keep outputs clean and predictable.
    """
    prompt_cfg = get_prompt_config(method, dataset) or {}
    fmt = prompt_cfg.get("format", "")

    properties: dict[str, dict] = {"text": {"type": "string"}}
    required = ["text"]

    # Determine label value constraints from prompt config, if available
    label_enum = None
    label_type: str | list[str] | None = None
    if "labels" in prompt_cfg and isinstance(prompt_cfg["labels"], list):
        # Simple list of string labels
        label_enum = list(prompt_cfg["labels"])  # keep order
        label_type = "string"
    elif "label_dict" in prompt_cfg and isinstance(prompt_cfg["label_dict"], dict):
        # Keys can be strings (e.g., R/M/U) or numbers (e.g., 0/1)
        keys = list(prompt_cfg["label_dict"].keys())
        # Preserve original types from YAML (ints remain ints)
        all_ints = all(isinstance(k, int) for k in keys)
        all_strs = all(isinstance(k, str) for k in keys)
        if all_ints:
            label_type = "integer"
        elif all_strs:
            label_type = "string"
        else:
            # Mixed types (unlikely) — allow both
            label_type = ["string", "integer"]
        label_enum = keys

    if fmt:
        try:
            fmt_json = json.loads(fmt)
            if (
                isinstance(fmt_json, list)
                and fmt_json
                and isinstance(fmt_json[0], dict)
            ):
                for k in fmt_json[0].keys():
                    if k == "text":
                        continue
                    if k == "label":
                        # Apply enum/type constraints if we have them, else keep broad type
                        if label_enum is not None and label_type is not None:
                            properties[k] = {"type": label_type, "enum": label_enum}
                        else:
                            # Default: accept string or number (to cover WMT and others)
                            properties[k] = {"type": ["string", "number"]}
                    else:
                        properties[k] = {"type": "string"}
        except Exception:
            # If the format isn't valid JSON, fall back to just {text}
            pass

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }
