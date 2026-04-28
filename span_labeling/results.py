"""Unified script for evaluating and exporting span labeling results.

This version uses **micro-averaging within a dataset**: TP/FP/FN counts
(here, character-overlap weights) are pooled across all examples in a
dataset, and a single F1 is computed from the pooled counts. This
matches standard practice in span labeling evaluation (e.g., seqeval,
CoNLL-2003, UNER).

Macro-averaging across datasets within a task is handled downstream in
the plotting / aggregation code.
"""

import json
from pathlib import Path
from typing import Any
from collections import defaultdict
from tqdm import tqdm

import pandas as pd

from span_labeling.metrics import compute_overlap_counts, f1_from_counts
from span_labeling.error_analysis import analyze_error, ErrorType
from span_labeling.experiments.run import METHOD_CLASSES


def _normalize_spans(spans: Any, text_length: int) -> list[dict[str, Any]]:
    """Keep only valid, bounded spans to avoid pathological metric loops."""
    if not isinstance(spans, list):
        return []

    normalized = []
    for span in spans:
        if not isinstance(span, dict):
            continue

        try:
            start = int(span.get("start"))
            end = int(span.get("end"))
        except (TypeError, ValueError):
            continue

        # Clamp bounds to text range and drop degenerate spans.
        start = max(0, min(start, text_length))
        end = max(0, min(end, text_length))
        if start >= end:
            continue

        normalized.append(
            {
                "start": start,
                "end": end,
                "label": span.get("label", ""),
            }
        )

    return normalized


def evaluate_file(results_path: str) -> dict[str, Any]:
    """Evaluate span labeling results from a JSON file.

    Aggregates overlap counts across all examples in the file (which
    corresponds to one dataset) and computes a single F1 from the
    pooled counts (micro-averaging).
    """
    with open(results_path) as f:
        results = json.load(f)

    metadata = results[0].get("metadata", {})

    completion_tokens_list = []
    total_entries = len(results)

    error_counts = defaultdict(int)
    for error_type in ErrorType:
        error_counts[error_type.value] = 0

    # Pooled counts across all examples in this dataset
    soft_overlap = 0
    soft_predicted = 0
    soft_gold = 0
    hard_overlap = 0
    hard_predicted = 0
    hard_gold = 0

    for res in results:
        output = res.get("output", {})
        text = res.get("text", "")
        text_length = len(text) if isinstance(text, str) else 0

        file_result_metadata = res.get("metadata", {})

        ct = res.get("completion_tokens")
        if ct is not None:
            completion_tokens_list.append(ct)

        model = file_result_metadata.get("model", {}).get("name", "unknown")
        file_result_metadata.get("model", {}).get("mode", "unknown")
        file_result_metadata.get("model", {}).get("enable_thinking", False)
        method = file_result_metadata.get("method", {}).get("name", None)
        method_type = file_result_metadata.get("method", {}).get("type", None)

        predicted_spans_raw = (
            output.get("spans", []) if isinstance(output, dict) else []
        )
        gold_spans_raw = res.get("spans", [])

        if "gemma" in model.lower() and "json" in method.lower():
            # Try parsing once again for Gemma+JSON cases
            _method = METHOD_CLASSES[method_type](None, None)
            staging_predicted_spans_raw = _method.parse_response(res)
            if len(staging_predicted_spans_raw) > len(predicted_spans_raw):
                res["output"]["spans"] = staging_predicted_spans_raw
                predicted_spans_raw = staging_predicted_spans_raw

        predicted_spans = _normalize_spans(predicted_spans_raw, text_length)
        gold_spans = _normalize_spans(gold_spans_raw, text_length)

        # Analyze error type
        error_type = analyze_error(res)
        error_counts[error_type.value] += 1

        # Soft counts (label-blind)
        soft_counts = compute_overlap_counts(
            predicted_spans, gold_spans, hard_matching=False
        )
        soft_overlap += soft_counts["overlap_chars"]
        soft_predicted += soft_counts["predicted_chars"]
        soft_gold += soft_counts["gold_chars"]

        # Hard counts (labels must match), except for synthetic where
        # hard == soft (no labels in the task)
        dataset_type = file_result_metadata.get("dataset", {}).get("type", "unknown")
        if dataset_type == "synthetic":
            hard_counts = soft_counts
        else:
            hard_counts = compute_overlap_counts(
                predicted_spans, gold_spans, hard_matching=True
            )
        hard_overlap += hard_counts["overlap_chars"]
        hard_predicted += hard_counts["predicted_chars"]
        hard_gold += hard_counts["gold_chars"]

    if completion_tokens_list:
        avg_completion_tokens = sum(completion_tokens_list) / len(
            completion_tokens_list
        )
        total_completion_tokens = sum(completion_tokens_list)
    else:
        avg_completion_tokens = 0
        total_completion_tokens = 0

    # Micro-averaged F1 from pooled counts
    soft = f1_from_counts(soft_overlap, soft_predicted, soft_gold)
    hard = f1_from_counts(hard_overlap, hard_predicted, hard_gold)

    print("Soft Matching Metrics (micro-averaged):")
    print(f"  Precision: {soft['precision']:.3f}")
    print(f"  Recall:    {soft['recall']:.3f}")
    print(f"  F1:        {soft['f1']:.3f}")
    print("Hard Matching Metrics (micro-averaged):")
    print(f"  Precision: {hard['precision']:.3f}")
    print(f"  Recall:    {hard['recall']:.3f}")
    print(f"  F1:        {hard['f1']:.3f}")
    print(f"Total Entries: {total_entries}")
    print("Error Counts:")
    for error_type, count in error_counts.items():
        print(f"  {error_type}: {count}")

    return {
        "average_metrics": {
            "soft_precision": soft["precision"],
            "soft_recall": soft["recall"],
            "soft_f1": soft["f1"],
            "hard_precision": hard["precision"],
            "hard_recall": hard["recall"],
            "hard_f1": hard["f1"],
        },
        "metadata": metadata,
        "statistics": {
            "total_entries": total_entries,
            "error_counts": dict(error_counts),
            "avg_completion_tokens": avg_completion_tokens,
            "total_completion_tokens": total_completion_tokens,
            # Pooled counts kept for downstream re-aggregation if needed
            "soft_overlap": soft_overlap,
            "soft_predicted": soft_predicted,
            "soft_gold": soft_gold,
            "hard_overlap": hard_overlap,
            "hard_predicted": hard_predicted,
            "hard_gold": hard_gold,
        },
    }


def evaluate_dir(results_dir: str) -> list[dict[str, Any]]:
    """Evaluate all result files in a directory."""
    all_results = []
    results_dir_path = Path(results_dir)
    result_files = sorted(results_dir_path.glob("*.json"))
    print(results_dir_path)
    print(len(result_files))

    for result_file in tqdm(result_files, desc="Evaluating result files"):
        if "constrained" in result_file.name and "fixed" not in result_file.name:
            print(
                f"Skipping {result_file.name} due to non-fixed constrained method",
                flush=True,
            )
            continue

        print(f"\nEvaluating {result_file.name}:", flush=True)
        file_results = evaluate_file(str(result_file))
        print("-" * 40)
        file_results["filepath"] = str(result_file)
        all_results.append(file_results)

    return all_results


def export_csv(results_dir: str, output_csv: str) -> None:
    """Export evaluation results to CSV."""
    all_results = evaluate_dir(results_dir)

    records = []
    for file_result in all_results:
        file_result_metadata = file_result["metadata"]
        avg_metrics = file_result["average_metrics"]
        statistics = file_result.get("statistics", {})

        experiment_name = file_result_metadata["experiment_name"]
        experiment_timestamp = file_result_metadata["experiment_timestamp"]
        model = file_result_metadata["model"]["name"]
        thinking = file_result_metadata["model"]["enable_thinking"]
        constrained = file_result_metadata["method"]["constrained"]
        structured = file_result_metadata["method"]["use_structured_outputs"]
        method_type = file_result_metadata["method"]["type"]
        method_name = file_result_metadata["method"]["name"]
        dataset_type = file_result_metadata["dataset"]["type"]
        dataset_name = file_result_metadata["dataset"]["name"]
        seed = file_result_metadata.get("seed", "unknown")

        record = {
            "result_path": file_result.get("filepath", "unknown"),
            "experiment_name": experiment_name,
            "experiment_timestamp": experiment_timestamp,
            "model": model,
            "dataset_type": dataset_type,
            "dataset_name": dataset_name,
            "method_type": method_type,
            "method_name": method_name,
            "seed": seed,
            "thinking": thinking,
            "constrained": constrained,
            "structured": structured,
            "soft_precision": avg_metrics["soft_precision"],
            "soft_recall": avg_metrics["soft_recall"],
            "soft_f1": avg_metrics["soft_f1"],
            "hard_precision": avg_metrics["hard_precision"],
            "hard_recall": avg_metrics["hard_recall"],
            "hard_f1": avg_metrics["hard_f1"],
            "total_entries": statistics.get("total_entries", 0),
            "avg_completion_tokens": statistics.get("avg_completion_tokens", 0),
            "total_completion_tokens": statistics.get("total_completion_tokens", 0),
            # Pooled counts: useful if you ever want to re-aggregate
            # (e.g. micro-averaging across datasets within a task)
            "soft_overlap": statistics.get("soft_overlap", 0),
            "soft_predicted": statistics.get("soft_predicted", 0),
            "soft_gold": statistics.get("soft_gold", 0),
            "hard_overlap": statistics.get("hard_overlap", 0),
            "hard_predicted": statistics.get("hard_predicted", 0),
            "hard_gold": statistics.get("hard_gold", 0),
        }

        # Add detailed error counts to the record
        error_counts = statistics.get("error_counts", {})
        for error_type, count in error_counts.items():
            col_name = f"error_{error_type.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"
            record[col_name] = count

        records.append(record)

    df = pd.DataFrame.from_records(records)
    df = df.fillna(0)
    df.to_csv(output_csv, index=False)
    print(f"\nExported results to {output_csv}")
    return df


if __name__ == "__main__":
    export_csv("results", "results/_results.csv")
