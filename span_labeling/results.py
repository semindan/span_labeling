"""Unified script for evaluating and exporting span labeling results"""

import json
from pathlib import Path
from typing import Any
from collections import defaultdict

import pandas as pd

from span_labeling.metrics import evaluate
from span_labeling.error_analysis import analyze_error


def evaluate_file(results_path: str) -> dict[str, Any]:
    """Evaluate span labeling results from a JSON file

    Args:
        results_path: Path to JSON file containing results

    Returns:
        Dict with average metrics and metadata
    """
    with open(results_path) as f:
        results = json.load(f)

    metadata = results[0].get("metadata", {})

    all_hard_metrics = []
    all_soft_metrics = []
    total_entries = len(results)

    error_counts = defaultdict(int)

    for res in results:
        output = res.get("output", {})

        file_result_metadata = res.get("metadata", {})

        method = file_result_metadata.get("custom_method_name", None)
        if not method:
            method = file_result_metadata.get("method", "unknown")

        if method == "occurrence":
            method = "json_occurrence"

        predicted_spans = output.get("spans", []) if isinstance(output, dict) else []

        gold_spans = res["spans"]
        if file_result_metadata.get("dataset") == "wmt":
            # res["allowed_labels"] = ["0", "1"]
            gold_spans_rewritten = []
            for span in gold_spans:
                if str(span.get("label")) == "0":
                    span["label"] = "MINOR"
                elif str(span.get("label")) == "1":
                    span["label"] = "MAJOR"
                gold_spans_rewritten.append(span)
            gold_spans = gold_spans_rewritten

            predicted_spans_rewritten = []
            for span in predicted_spans:
                if str(span.get("label")) == "0":
                    span["label"] = "MINOR"
                elif str(span.get("label")) == "1":
                    span["label"] = "MAJOR"
                predicted_spans_rewritten.append(span)
            predicted_spans = predicted_spans_rewritten

        # Analyze error type
        error_type = analyze_error(res, method)
        error_counts[error_type] += 1

        soft_metrics = evaluate(predicted_spans, gold_spans, hard_matching=False)
        hard_metrics = evaluate(predicted_spans, gold_spans, hard_matching=True)
        all_soft_metrics.append(soft_metrics)
        all_hard_metrics.append(hard_metrics)

    # Aggregate metrics
    soft_total_precision = sum(m["precision"] for m in all_soft_metrics) / len(
        all_soft_metrics
    )
    soft_total_recall = sum(m["recall"] for m in all_soft_metrics) / len(
        all_soft_metrics
    )
    soft_total_f1 = sum(m["f1"] for m in all_soft_metrics) / len(all_soft_metrics)

    hard_total_precision = sum(m["precision"] for m in all_hard_metrics) / len(
        all_hard_metrics
    )
    hard_total_recall = sum(m["recall"] for m in all_hard_metrics) / len(
        all_hard_metrics
    )
    hard_total_f1 = sum(m["f1"] for m in all_hard_metrics) / len(all_hard_metrics)

    print("Soft Matching Metrics:")
    print(f"Average Precision: {soft_total_precision:.3f}")
    print(f"Average Recall: {soft_total_recall:.3f}")
    print(f"Average F1: {soft_total_f1:.3f}")
    print("Hard Matching Metrics:")

    print(f"Average Precision: {hard_total_precision:.3f}")
    print(f"Average Recall: {hard_total_recall:.3f}")
    print(f"Average F1: {hard_total_f1:.3f}")

    print(f"Total Entries: {total_entries}")
    print("Error Counts:")
    for error_type, count in error_counts.items():
        print(f"  {error_type}: {count}")

    return {
        "average_metrics": {
            "soft_precision": soft_total_precision,
            "soft_recall": soft_total_recall,
            "soft_f1": soft_total_f1,
            "hard_precision": hard_total_precision,
            "hard_recall": hard_total_recall,
            "hard_f1": hard_total_f1,
        },
        "metadata": metadata,
        "statistics": {
            "total_entries": total_entries,
            "error_counts": dict(error_counts),
            "empty_raw_response": error_counts.get("Empty Response", 0),
            "parsing_errors_format": error_counts.get("JSON Syntax Error", 0)
            + error_counts.get("Invalid Content / Schema (Parsed JSON)", 0),
            "parsing_errors_invalid_spans": error_counts.get("All Invalid Labels", 0)
            + error_counts.get("Partial Invalid Labels", 0)
            + error_counts.get("Invalid Spans (Content Mismatch)", 0),
        },
    }


def evaluate_dir(results_dir: str) -> list[dict[str, Any]]:
    """Evaluate all result files in a directory

    Args:
        results_dir: Path to directory containing JSON result files

    Returns:
        List of evaluation results for each file
    """
    all_results = []
    results_dir_path = Path(results_dir)
    result_files = sorted(results_dir_path.glob("*.json"))
    print(results_dir_path)
    print(result_files)
    for result_file in result_files:
        print(f"\nEvaluating {result_file.name}:")
        file_results = evaluate_file(str(result_file))
        print("-" * 40)
        all_results.append(file_results)

    return all_results


def export_csv(results_dir: str, output_csv: str) -> None:
    """Export evaluation results from a directory to a CSV file

    Args:
        results_dir: Path to directory containing JSON result files
        output_csv: Path to output CSV file
    """
    all_results = evaluate_dir(results_dir)

    # Prepare data for DataFrame
    records = []
    for file_result in all_results:
        file_result_metadata = file_result["metadata"]
        avg_metrics = file_result["average_metrics"]
        statistics = file_result.get("statistics", {})
        method = file_result_metadata.get("custom_method_name", None)
        if not method:
            method = file_result_metadata.get("method", "unknown")

        if method == "occurrence":
            method = "json_occurrence"

        dataset = file_result_metadata.get("custom_dataset_name", None)
        if not dataset:
            dataset = Path(file_result_metadata.get("dataset_path", "")).stem

        if "wmt" in dataset and dataset.split("-")[-1] not in [
            "news",
            "social",
            "literary",
        ]:
            dataset = dataset + "-news"

        record = {
            "model": file_result_metadata.get("model", ""),
            "dataset": file_result_metadata.get("dataset", ""),
            "method": method,
            "dataset_name": dataset,
            "soft_precision": avg_metrics["soft_precision"],
            "soft_recall": avg_metrics["soft_recall"],
            "soft_f1": avg_metrics["soft_f1"],
            "hard_precision": avg_metrics["hard_precision"],
            "hard_recall": avg_metrics["hard_recall"],
            "hard_f1": avg_metrics["hard_f1"],
            "total_entries": statistics.get("total_entries", 0),
            "empty_raw_response": statistics.get("empty_raw_response", 0),
            "parsing_errors_format": statistics.get("parsing_errors_format", 0),
            "parsing_errors_invalid_spans": statistics.get(
                "parsing_errors_invalid_spans", 0
            ),
        }

        # Add detailed error counts to the record
        error_counts = statistics.get("error_counts", {})
        for error_type, count in error_counts.items():
            # Normalize column name
            col_name = f"error_{error_type.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"
            record[col_name] = count

        records.append(record)

    df = pd.DataFrame.from_records(records)
    # Fill NaNs for error columns with 0
    df = df.fillna(0)
    # print(df)
    df.to_csv(output_csv, index=False)
    print(f"\nExported results to {output_csv}")
    return df


def compare_methods(
    results_dir: str, methods: str = "xml,json,occurrence,index"
) -> None:
    """Compare results across different methods grouped by model

    Args:
        results_dir: Path to directory containing JSON result files
        methods: Comma-separated list of methods to compare
    """
    all_results = evaluate_dir(results_dir)

    # Prepare data for DataFrame
    records = []
    for file_result in all_results:
        file_result_metadata = file_result["metadata"]
        avg_metrics = file_result["average_metrics"]
        statistics = file_result.get("statistics", {})
        record = {
            "model": file_result_metadata.get("model", ""),
            "dataset": file_result_metadata.get("dataset", ""),
            "method": file_result_metadata.get("method", ""),
            "dataset_name": Path(file_result_metadata.get("dataset_path", "")).stem,
            "precision": avg_metrics["precision"],
            "recall": avg_metrics["recall"],
            "f1": avg_metrics["f1"],
            "total_entries": statistics.get("total_entries", 0),
            "empty_raw_response": statistics.get("empty_raw_response", 0),
            "parsing_errors": statistics.get("parsing_errors", 0),
        }
        records.append(record)

    df = pd.DataFrame.from_records(records)
    method_list = [m.strip() for m in methods.split(",")]

    print("\n" + "=" * 80)
    print("COMPARISON BY MODEL")
    print("=" * 80)

    for model, group in df.groupby("model"):
        print(f"\n{model}:")
        filtered = group[group["method"].isin(method_list)].sort_values("dataset")
        if not filtered.empty:
            print(filtered.to_string(index=False))
