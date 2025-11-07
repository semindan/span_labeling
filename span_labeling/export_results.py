# %%
from span_labeling.evaluate_results import run_dir
import pandas as pd
from pathlib import Path


def export_results(results_dir: str, output_csv: str):
    """Export evaluation results from a directory to a CSV file"""
    all_results = run_dir(results_dir)
    # Prepare data for DataFrame
    records = []
    for file_result in all_results:
        file_result_metadata = file_result["metadata"]
        avg_metrics = file_result["average_metrics"]
        record = {
            "model": file_result_metadata.get("model", ""),
            "dataset": file_result_metadata.get("dataset", ""),
            "method": file_result_metadata.get("method", ""),
            "dataset_name": Path(file_result_metadata.get("dataset_path", "")).stem,
            "precision": avg_metrics["precision"],
            "recall": avg_metrics["recall"],
            "f1": avg_metrics["f1"],
        }
        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df
    df.to_csv(output_csv, index=False)
    print(f"Exported results to {output_csv}")


# %%
if __name__ == "__main__":
    # %%
    results_dir = "/home/semin/personal_work_ms/span_labeling/results"
    df = export_results(
        results_dir="/home/semin/personal_work_ms/span_labeling/results",
        output_csv="span_labeling_evaluation_results.csv",
    )

    # %%
    df.to_csv("span_labeling_evaluation_results.csv", index=False)


# %%
df.groupby("model")

# %%
