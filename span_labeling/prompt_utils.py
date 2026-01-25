import yaml
from pathlib import Path


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


def build_prompt(
    method: str, dataset: str, entry: dict, note_extra: str = "", example_n: int = 100
) -> str:
    prompt_data = get_prompt_config(method, dataset)
    if not prompt_data:
        raise ValueError(f"No prompt found for {method}/{dataset}")

    sections = []

    sections.append(prompt_data["task"])
    sections.append("")

    if "format" in prompt_data:
        sections.append(f"Output Format: {prompt_data['format']}")

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
            if i > example_n:
                break
            examples_section += f"{i}. {example}\n"
        sections.append(examples_section)

    note_section = ""
    if "note" in prompt_data:
        note_section += f"\n{prompt_data['note']}"
    if note_extra:
        note_section += f"{note_extra}"
    if note_section:
        sections.append(f"{note_section}\n")

    if "instruction" in prompt_data:
        sections.append(f"\n{prompt_data['instruction']}")

    sections.append(f"{entry['model_input']}")

    return "\n".join(sections)
