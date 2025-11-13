import json
from copy import deepcopy
from pathlib import Path

from span_labeling.config import get_hard_matching
from span_labeling.dataset import (
    ErrorDataset,
    MultigecDataset,
    NerDataset,
    SyntheticDataset,
    WMTDataset,
)
from span_labeling.methods.constrained_json_method import ConstrainedJSONSpanLabeler
from span_labeling.methods.index_method import IndexSpanLabeler
from span_labeling.methods.json_method import JSONSpanLabeler
from span_labeling.methods.occurrence_method import JSONOccurrenceSpanLabeler
from span_labeling.methods.xml_method import XMLSpanLabeler
from span_labeling.metrics import evaluate

methods = {
    # All constructors now default to the central config for model/base_url
    # "JSON": JSONSpanLabeler(),
    "Constrained-JSON": ConstrainedJSONSpanLabeler(api_url="http://tdll-8gpu2:5456"),
    # "XML": XMLSpanLabeler(),
    # 'Index': IndexSpanLabeler(model_name=None),  # uses config via SpanLabeler
    # 'Occurrence': JSONOccurrenceSpanLabeler(),
}

tasks = {
    "error": ErrorDataset(path="data/custom/error_test.json"),
    "ner": NerDataset(path="data/custom/ner_en_test.json"),
    "multigec": MultigecDataset(path="data/custom/multigec.json"),
    # 'ner': NerDataset(path="data/custom/ner_test.json"),
    "synthetic": SyntheticDataset(path="data/custom/synthetic_test.json"),
    # "wmt": WMTDataset(
    # path="data/custom/wmt-cs.json",
    # ),
}

all_results = {}

for task_key, dataset_obj in tasks.items():
    print(f"\n{'#'*80}")
    print(f"Task: {task_key} - {dataset_obj.name}")
    print(f"{'#'*80}\n")

    dataset = dataset_obj.load()
    all_results[task_key] = {}

    for method_name, method in methods.items():
        print(f"\n{method_name}:")

        results = []
        for i, ex in enumerate(deepcopy(dataset)):
            # if i > 2:
            #     break
            ex = method.predict(ex)
            result = ex["output"]

            if result["success"]:
                print(f"Out: {ex['output']['raw_response']}")

                metrics = evaluate(
                    result["spans"], ex["spans"], hard_matching=get_hard_matching()
                )
                results.append({"metrics": metrics, "entry": ex})
                print(f"  {i}: F1={metrics['f1']:.3f}")
            else:
                results.append({"f1": 0, "precision": 0, "recall": 0, "entry": ex})
                print(f"  {i}: FAIL")

        # Average
        metrics = [r["metrics"] for r in results if r.get("metrics")]
        avg_f1 = sum(r["f1"] for r in metrics) / len(results)
        avg_p = sum(r["precision"] for r in metrics) / len(results)
        avg_r = sum(r["recall"] for r in metrics) / len(results)

        all_results[task_key][method_name] = {
            "f1": avg_f1,
            "precision": avg_p,
            "recall": avg_r,
            "per_example": results,
        }

        print(f"  → Avg F1: {avg_f1:.3f}")


# Print summary table
print(f"\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}\n")

print(f"{'Task':<12} {'Method':<12} {'F1':<8} {'Precision':<10} {'Recall':<8}")
print("-" * 50)

for task in all_results:
    for method in all_results[task]:
        r = all_results[task][method]
        print(
            f"{task:<12} {method:<12} {r['f1']:<8.3f} {r['precision']:<10.3f} {r['recall']:<8.3f}"
        )

# Save results
Path("results").mkdir(exist_ok=True)
with open("results/all_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved to results/all_results.json")
