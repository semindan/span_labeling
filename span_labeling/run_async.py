import asyncio
from span_labeling.base import (
    SpanLabelerBase,
    DatasetBase,
)
from tqdm import tqdm
import span_labeling.methods  # noqa: F401
import span_labeling.dataset  # noqa: F401
from copy import deepcopy
from pathlib import Path
import json
from typing import Optional

PROJECT_ROOT = Path(__file__).absolute().parent.parent.as_posix()


def run_experiment_async(
    method: SpanLabelerBase,
    dataset: DatasetBase,
    method_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    max_concurrent_requests: int = 8,
    comment: Optional[str] = None,
):
    """
    Simplified experiment runner that accepts initialized method and dataset instances.

    Args:
        method: Initialized method instance (e.g., IndexSpanLabeler(...))
        dataset: Initialized dataset instance (e.g., SyntheticDataset(...))
        method_name: Optional custom name for method (used in auto-generated filename)
        dataset_name: Optional custom name for dataset (used in auto-generated filename)
        max_concurrent_requests: Max number of concurrent API requests
        comment: Optional comment prefix for output filename
    """

    async def async_run_with_instances():
        model_clean = (
            method.model_name.replace(":", "_").replace(".", "_").replace("/", "__")
        )

        method_part = method_name if method_name else method.key

        dataset_part = dataset_name

        generated_name = f"{model_clean}_{method_part}_{dataset_part}"

        comment_prefix = "" if not comment else f"{comment}_"
        filename = f"{comment_prefix}{generated_name}_results.json"
        output_path = Path(PROJECT_ROOT) / "results_2026" / filename
        print(f"Output path: {output_path}")

        dataset_loaded = dataset.load()

        semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def bounded_predict(entry):
            async with semaphore:
                return await method.async_predict(entry)

        tasks = [bounded_predict(deepcopy(entry)) for entry in dataset_loaded]

        results = []
        results_iterator = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Processing entries (Async API)",
        )

        for future in results_iterator:
            result = await future

            # 7. Add metadata
            result["metadata"] = {
                "method": method.key,
                "dataset": dataset.key,
                "model": method.model_name,
                "dataset_path": str(dataset.path),
                "dataset_key": getattr(dataset, "key", None),
                "dataset_name": getattr(dataset, "name", None),
                "custom_method_name": method_name,
                "custom_dataset_name": dataset_name,
                "experiment_name": generated_name,
                "method_config": {
                    k: v
                    for k, v in vars(method).items()
                    if not k.startswith("_") and k not in ["client", "async_client"]
                },
            }
            results.append(result)

        # 8. Save results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving results to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Run the async function
    asyncio.run(async_run_with_instances())
