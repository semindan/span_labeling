import fire
from span_labeling.base import MethodRegistry, DatasetRegistry
from tqdm import tqdm
import span_labeling.methods  # noqa: F401
import span_labeling.dataset  # noqa: F401
from copy import deepcopy
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).absolute().parent.parent.as_posix()


def run(
    model_name, method_name, dataset_name, dataset_path, comment=None, output_path=None
):
    MethodClass = MethodRegistry.get(method_name)
    DatasetClass = DatasetRegistry.get(dataset_name)

    if not output_path:
        dataset_file_name = Path(dataset_path).stem
        comment = "" if not comment else f"{comment}_"
        output_path = (
            Path(PROJECT_ROOT)
            / f"results/{comment}{model_name}_{method_name}_{dataset_file_name}_results.json"
        )

    method = MethodClass(model_name)
    dataset = DatasetClass(path=dataset_path).load()

    results = []
    for i, entry in enumerate(tqdm(deepcopy(dataset), desc="Processing entries")):
        entry = method.predict(entry)
        entry["metadata"] = {
            "method": method_name,
            "dataset": dataset_name,
            "model": model_name,
            "dataset_path": dataset_path,
        }
        results.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(run)
