"""
Simple grid-based experiment runner.

Define grids of models, datasets, and method parameters.
All combinations will be run automatically.
"""

from pathlib import Path
from span_labeling.run_async import run_experiment_async
from span_labeling.methods.index_method import IndexSpanLabeler
from span_labeling.methods.xml_method import XMLSpanLabeler
from span_labeling.methods.json_method import JSONSpanLabeler
from span_labeling.methods.occurrence_method import JSONOccurrenceSpanLabeler
from span_labeling.dataset import (
    SyntheticDataset,
    NerDataset,
    MultigecDataset,
    WMTDataset,
)

# Configuration
BASE_DIR = Path.home() / "personal_work_ms" / "span_labeling"
DATA_DIR = BASE_DIR / "data"

# Define your grids
MODELS = [
    # "qwen3:8b",
    # "llama3.1:8b",
    # "mistral:7b",
    # "gpt-oss:20b",
    # "Qwen/Qwen3-8B",
    "gpt-5-mini",
    # "meta-llama/Llama-3.3-70B-Instruct",
    # "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "NousResearch/Hermes-3-Llama-3.1-8B",
    # "openai/gpt-oss-20b",
    # "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    # "mistralai/Mistral-Small-24B-Instruct-2501",
    # "Qwen/Qwen3-32B"
]

DATASETS = [
    (NerDataset, {"path": DATA_DIR / "universal_ner" / "uner_en_ewt.json"}, None),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_ceb_gja.json"},
        "uner_ceb_gja",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_pt_bosque.json"},
        "uner_pt_bosque",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_sk_snk.json"},
        "uner_sk_snk",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_sv_talbanken.json"},
        "uner_sv_talbanken",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_zh_gsd.json"},
        "uner_zh_gsd",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_da_ddt.json"},
        "uner_da_ddt",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_en_pud.json"},
        "uner_en_pud",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_pt_pud.json"},
        "uner_pt_pud",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_sr_set.json"},
        "uner_sr_set",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_tl_trg.json"},
        "uner_tl_trg",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_zh_gsdsimp.json"},
        "uner_zh_gsdsimp",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_de_pud.json"},
        "uner_de_pud",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_hr_set.json"},
        "uner_hr_set",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_ru_pud.json"},
        "uner_ru_pud",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_sv_pud.json"},
        "uner_sv_pud",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_tl_ugnayan.json"},
        "uner_tl_ugnayan",
    ),
    (
        NerDataset,
        {"path": DATA_DIR / "universal_ner" / "uner_zh_pud.json"},
        "uner_zh_pud",
    ),
    (MultigecDataset, {"pat": DATA_DIR / "multigec" / "multigec_en.json"}, None),
    (WMTDataset, {"path": DATA_DIR / "wmt" / "wmt-en-cs-news.json"}, "wmt-en-cs"),
    (WMTDataset, {"path": DATA_DIR / "wmt" / "wmt-en-ru-news.json"}, "wmt-en-ru"),
    (WMTDataset, {"path": DATA_DIR / "wmt" / "wmt-en-zh-news.json"}, "wmt-en-zh"),
    (WMTDataset, {"path": DATA_DIR / "wmt" / "wmt-en-es-news.json"}, "wmt-en-es"),
    (WMTDataset, {"path": DATA_DIR / "wmt" / "wmt-en-ja-news.json"}, "wmt-en-ja"),
    (WMTDataset, {"path": DATA_DIR / "wmt" / "wmt-en-hi-news.json"}, "wmt-en-hi"),
    (WMTDataset, {"path": DATA_DIR / "wmt" / "wmt-en-is-news.json"}, "wmt-en-is"),
    (WMTDataset, {"path": DATA_DIR / "wmt" / "wmt-en-uk-news.json"}, "wmt-en-uk"),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-cs-literary.json"},
        "wmt-en-cs-literary",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-ru-literary.json"},
        "wmt-en-ru-literary",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-zh-literary.json"},
        "wmt-en-zh-literary",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-es-literary.json"},
        "wmt-en-es-literary",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-ja-literary.json"},
        "wmt-en-ja-literary",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-hi-literary.json"},
        "wmt-en-hi-literary",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-is-literary.json"},
        "wmt-en-is-literary",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-uk-literary.json"},
        "wmt-en-uk-literary",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-cs-social.json"},
        "wmt-en-cs-social",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-ru-social.json"},
        "wmt-en-ru-social",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-zh-social.json"},
        "wmt-en-zh-social",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-es-social.json"},
        "wmt-en-es-social",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-ja-social.json"},
        "wmt-en-ja-social",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-hi-social.json"},
        "wmt-en-hi-social",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-is-social.json"},
        "wmt-en-is-social",
    ),
    (
        WMTDataset,
        {"path": DATA_DIR / "wmt" / "wmt-en-uk-social.json"},
        "wmt-en-uk-social",
    ),
    (
        SyntheticDataset,
        {"path": DATA_DIR / "synthetic" / "english_word_synthetic_data.json"},
        None,
    ),
    (
        SyntheticDataset,
        {
            "path": DATA_DIR
            / "synthetic"
            / "english_non_overlapping_word_synthetic_data.json"
        },
        None,
    ),
]

METHODS = [
    (IndexSpanLabeler, {"enrich_prompt": False}, None),  # Auto-generated name
    (IndexSpanLabeler, {"enrich_prompt": True}, "index_enriched"),  # Custom saved name
    (XMLSpanLabeler, {}, None),
    (JSONSpanLabeler, {}, None),
    (JSONOccurrenceSpanLabeler, {}, None),
    (JSONSpanLabeler, {"use_structured_outputs": True}, "json_structured"),
    (
        JSONOccurrenceSpanLabeler,
        {"use_structured_outputs": True},
        "json_occurrence_structured",
    ),
    # (JSONSpanLabeler, {"constrained" : True}, "json_constrained"),
    # (JSONOccurrenceSpanLabeler, {"constrained" : True}, "json_occurrence_constrained"),
    # (JSONSpanLabeler, {"constrained" : True, "use_structured_outputs": True}, "json_constrained_structured"),
    # (JSONOccurrenceSpanLabeler, {"constrained" : True, "use_structured_outputs": True}, "json_occurrence_constrained_structured"),
    # (IndexSpanLabeler, {"enrich_prompt": False, "enable_thinking": True}, "index_thinking"),  # Auto-generated name
    # (IndexSpanLabeler, {"enrich_prompt": True, "enable_thinking": True}, "index_enriched_thinking"),  # Custom saved name
    # (XMLSpanLabeler, {"enable_thinking": True}, "xml_thinking"),
    # (JSONSpanLabeler, {"enable_thinking": True}, "json_thinking"),
    # (JSONOccurrenceSpanLabeler, {"enable_thinking": True}, "json_occurrence_thinking"),
    # (JSONSpanLabeler, {"use_structured_outputs": True, "enable_thinking": True}, "json_structured_thinking"),
    # (JSONOccurrenceSpanLabeler, {"use_structured_outputs": True, "enable_thinking": True}, "json_occurrence_structured_thinking"),
    # (JSONSpanLabeler, {"constrained" : True, "enable_thinking": True}, "json_constrained_thinking"),
    # (JSONOccurrenceSpanLabeler, {"constrained" : True, "enable_thinking": True}, "json_occurrence_constrained_thinking"),
    # (JSONSpanLabeler, {"use_structured_outputs": True, "enable_thinking": True, "constrained": True}, "json_constrained_structured_thinking"),
    # (JSONOccurrenceSpanLabeler, {"use_structured_outputs": True, "enable_thinking": True, "constrained": True}, "json_occurrence_constrained_structured_thinking"),
]


if __name__ == "__main__":
    # Run grid: models × datasets × methods
    for model_name in MODELS:
        for dataset_spec in DATASETS:
            # Unpack dataset specification
            if len(dataset_spec) == 3:
                dataset_class, dataset_params, dataset_name = dataset_spec
                dataset_name = (
                    dataset_name
                    if dataset_name is not None
                    else dataset_params["path"].stem
                )
            else:
                dataset_class, dataset_params = dataset_spec
                dataset_name = dataset_params["path"].stem

            dataset = dataset_class(**dataset_params)

            for method_spec in METHODS:
                # Unpack method specification
                if len(method_spec) == 3:
                    method_class, method_params, method_name = method_spec
                else:
                    method_class, method_params = method_spec
                    method_name = None

                method = method_class(model_name, **method_params)

                run_experiment_async(
                    method=method,
                    dataset=dataset,
                    method_name=method_name,
                    dataset_name=dataset_name,
                    max_concurrent_requests=8,
                )
