import json
from pathlib import Path
from copy import deepcopy
from span_labeling.methods.json_method import JSONSpanLabeler
from span_labeling.methods.xml_method import XMLSpanLabeler
from span_labeling.methods.index_method import IndexSpanLabeler
from span_labeling.methods.occurrence_method import JSONOccurrenceSpanLabeler
from span_labeling.dataset import ErrorDataset, NerDataset, SyntheticDataset

def evaluate(pred, gold):
    pred_set = {(s['start'], s['end'], s['label']) for s in pred}
    gold_set = {(s['start'], s['end'], s['label']) for s in gold}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp/(tp+fp) if tp+fp > 0 else 0
    r = tp/(tp+fn) if tp+fn > 0 else 0
    f1 = 2*p*r/(p+r) if p+r > 0 else 0
    return {'f1': f1, 'p': p, 'r': r, 'tp': tp, 'fp': fp, 'fn': fn}

methods = {
    # 'JSON': JSONSpanLabeler(model_name="hermes3:8b"),
    # 'XML': XMLSpanLabeler(model_name="hermes3:8b"),
    # 'Index': IndexSpanLabeler(model_name="hermes3:8b"),
    'Occurrence': JSONOccurrenceSpanLabeler(model_name="hermes3:8b")
}

tasks = {
    # 'error': ErrorDataset(path="data/custom/error_test.json"),
    # 'ner': NerDataset(path="data/custom/ner_test.json"),
    'synthetic': SyntheticDataset(path="data/custom/synthetic_test.json")
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
            ex = method.predict(ex)

            result = ex["output"]            
            
            if result['success']:
                metrics = evaluate(result['spans'], ex['spans'])
                results.append({"metrics" : metrics, "entry": ex})
                print(f"  {i}: F1={metrics['f1']:.3f}")
            else:
                results.append({'f1': 0, 'p': 0, 'r': 0, "entry": ex})
                print(f"  {i}: FAIL")
        
        # Average
        metrics = [r['metrics'] for r in results if r.get('metrics')]
        avg_f1 = sum(r['f1'] for r in metrics) / len(results)
        avg_p = sum(r['p'] for r in metrics) / len(results)
        avg_r = sum(r['r'] for r in metrics) / len(results)
        
        all_results[task_key][method_name] = {
            'f1': avg_f1,
            'precision': avg_p,
            'recall': avg_r,
            'per_example': results,
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
        print(f"{task:<12} {method:<12} {r['f1']:<8.3f} {r['precision']:<10.3f} {r['recall']:<8.3f}")

# Save results
Path('results').mkdir(exist_ok=True)
with open('results/all_results.json', 'w', encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved to results/all_results.json")