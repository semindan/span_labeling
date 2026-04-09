import asyncio
import json
from copy import deepcopy
from pathlib import Path
from tqdm.asyncio import tqdm as tqdm_asyncio
import datetime
from itertools import product
from fire import Fire
import pandas as pd
from span_labeling.config import PROJECT_ROOT, CONFIGS_DIR, Settings
from span_labeling.methods.index_method import IndexSpanLabeler
from span_labeling.methods.xml_method import XMLSpanLabeler
from span_labeling.methods.json_method import JSONSpanLabeler
from span_labeling.methods.occurrence_method import JSONOccurrenceSpanLabeler
from span_labeling.dataset import (
    NerDataset,
    MultigecDataset,
    SyntheticDataset,
    WMTDataset,
)
from span_labeling.modeling.openai_model import OpenAIModel
from span_labeling.modeling.vllm_model import VLLMModel


MODEL_CLASSES = {
    "openai": OpenAIModel,
    "vllm": VLLMModel,
}

METHOD_CLASSES = {
    "index": IndexSpanLabeler,
    "xml": XMLSpanLabeler,
    "json": JSONSpanLabeler,
    "json_occurrence": JSONOccurrenceSpanLabeler,
}

DATASET_CLASSES = {
    "ner": NerDataset,
    "multigec": MultigecDataset,
    "synthetic": SyntheticDataset,
    "wmt": WMTDataset,
}


def run_experiment_async(config: Settings):
    experiment_name = config.experiment.name
    method_name = config.method.name
    model_name = config.model.name
    seed = config.seed
    clean_model_name = model_name.replace(":", "_").replace(".", "_").replace("/", "__")
    dataset_name = config.dataset.name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_name = f"{experiment_name}_{clean_model_name}_{method_name}_{dataset_name}_{seed}_{timestamp}"
    output_file_name = f"{generated_name}_results.json"
    output_path = Path(PROJECT_ROOT) / config.experiment.output_dir / output_file_name

    model = MODEL_CLASSES[config.model.mode](config)
    method = METHOD_CLASSES[config.method.type](model, config)
    dataset = DATASET_CLASSES[config.dataset.type](config)

    dataset_loaded = dataset.load()

    async def async_run_with_instances():
        print(f"Output path: {output_path}")

        semaphore = asyncio.Semaphore(config.project.max_concurrent_requests)

        async def bounded_predict(entry):
            async with semaphore:
                return await method.async_predict(entry)

        tasks = []
        for i_entry, entry in enumerate(dataset_loaded):
            # if i_entry > 1:
            #     break
            tasks.append(bounded_predict(deepcopy(entry)))

        raw_results = await tqdm_asyncio.gather(
            *tasks, total=len(tasks), desc="Processing entries"
        )

        results = []
        for result in raw_results:
            result["metadata"] = {
                "config_path": config.config_path,
                "model": config.model.model_dump(mode="json"),
                "method": config.method.model_dump(mode="json"),
                "dataset": config.dataset.model_dump(mode="json"),
                "seed": config.seed,
                "experiment_name": config.experiment.name,
                "experiment_timestamp": timestamp,
            }
            results.append(result)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving results to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        model.client.close()
        await model.async_client.close()

    asyncio.run(async_run_with_instances())


# %%


def main(config_path: str, port: int = 8057):
    # config_path = Path(PROJECT_ROOT) / "span_labeling" / "configs" / "mistral_constrained.yaml"
    config_path = Path(config_path)

    if config_path.parent == Path("."):
        config_path = CONFIGS_DIR / config_path

    if not config_path.exists():
        print(f"Config file {config_path} does not exist.")
        return

    settings = Settings.from_yaml(config_path)
    print(f"Experiment Name: {settings.experiment.name}")

    models = [settings.model] if settings.model else settings.experiment.models

    if settings.dataset and settings.experiment.datasets:
        raise ValueError(
            "Both dataset and experiment.datasets are defined. Please specify only one of those parameters."
        )

    datasets = []
    if settings.dataset:
        datasets = [settings.dataset]
    elif settings.experiment.datasets:
        datasets = settings.experiment.datasets

    print(f"Datasets: {[dataset.name for dataset in datasets]}")

    # if settings.experiment.dataset_groups:
    #     for group_name in settings.experiment.dataset_groups:
    #         if group_name not in DATASET_GROUPS:
    #             raise ValueError(f"Dataset group {group_name} not found in DATASET_GROUPS.")
    #         datasets.extend(DATASET_GROUPS[group_name])

    # print(f" AFTER Datasets: {[dataset.name for dataset in datasets]}")

    methods = [settings.method] if settings.method else settings.experiment.methods
    seeds = [settings.seed] if settings.seed else settings.experiment.seeds

    # print(f"Models: {[model.name for model in models]}")
    # print(f"Methods: {[method.name for method in methods]}")
    # print(f"Datasets: {[dataset.name for dataset in datasets]}")
    # print(f"Seeds: {seeds}")

    # print("dataset groups: ", settings.experiment.dataset_groups)
    for model, method, dataset, seed in product(models, methods, datasets, seeds):
        print(
            f"Model: {model.name}, Method: {method.name}, Dataset: {dataset.name}, Seed: {seed}"
        )

        instance_config = settings.model_copy(
            update={
                "model": model,
                "dataset": dataset,
                "method": method,
                "seed": seed,
            },
            deep=True,
        )
        if instance_config.project.skip_experiment_if_exists:
            output_path = Path(PROJECT_ROOT) / settings.experiment.output_dir
            df = pd.read_csv(output_path / "results.csv")
            experiment_name = instance_config.experiment.name
            method_name = instance_config.method.name
            model_name = instance_config.model.name
            dataset_name = instance_config.dataset.name
            constrained = instance_config.method.constrained
            thinking = instance_config.model.enable_thinking
            structured = instance_config.method.use_structured_outputs

            filtered_df = df[
                (df["experiment_name"] == experiment_name)
                & (df["model"] == model_name)
                & (df["method_name"] == method_name)
                & (df["dataset_name"] == dataset_name)
                & (df["seed"] == seed)
                & (df["constrained"] == constrained)
                & (df["thinking"] == thinking)
                & (df["structured"] == structured)
            ]

            if not filtered_df.empty:
                print(
                    f"SKIP: Experiment already exists for config: {instance_config.config_path}"
                )
                print(f"experiment: {experiment_name}")
                print(f"model: {model_name}")
                print(f"method: {method_name}")
                print(f"dataset: {dataset_name}")
                print(f"seed: {seed}")
                print(f"constrained: {constrained}")
                print(f"thinking: {thinking}")
                print(f"structured: {structured}")
                continue

        if settings.experiment.mode == "vllm":
            instance_config.model.set_port(port)

        run_experiment_async(instance_config)


# %%
if __name__ == "__main__":
    Fire(main)
