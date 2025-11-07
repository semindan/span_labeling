from span_labeling.metrics import evaluate
import fire


def run(results_path: str):
    """Evaluate span labeling results from a JSON file"""
    import json

    with open(results_path) as f:
        results = json.load(f)

    metadata = results[0].get("metadata", {})

    all_metrics = []
    for res in results:
        predicted_spans = res["output"]["spans"]
        gold_spans = res["spans"]
        metrics = evaluate(predicted_spans, gold_spans, hard_matching=False)
        all_metrics.append(metrics)

    # Aggregate metrics
    total_precision = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    total_recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    total_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)

    print(f"Average Precision: {total_precision:.3f}")
    print(f"Average Recall: {total_recall:.3f}")
    print(f"Average F1: {total_f1:.3f}")

    return {
        "all_metrics": all_metrics,
        "average_metrics": {
            "precision": total_precision,
            "recall": total_recall,
            "f1": total_f1,
        },
        "metadata": metadata,
    }


def run_dir(results_dir: str):
    """Evaluate all result files in a directory"""
    from pathlib import Path

    all_results = []
    results_dir_path = Path(results_dir)
    result_files = list(results_dir_path.glob("*.json"))

    for result_file in result_files:
        print(f"Evaluating {result_file.name}:")
        file_results = run(str(result_file))
        print("-" * 40)
        all_results.append(file_results)

    return all_results


if __name__ == "__main__":
    fire.Fire(
        {
            "file": run,
            "dir": run_dir,
        }
    )
