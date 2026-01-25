# %%
import json
import re
from span_labeling.methods.json_method import JSONSpanLabeler
from span_labeling.methods.occurrence_method import JSONOccurrenceSpanLabeler


def analyze_error(entry, method_name):
    output = entry.get("output", {})
    raw_response = output.get("raw_response", "")

    if not isinstance(raw_response, str):
        raw_response = str(raw_response)

    # 1. Empty Response
    if not raw_response or not raw_response.strip():
        return "Empty Response"

    # 2. Runner Error
    if "error" in output:
        return f"Runner Error: {output['error']}"

    predicted_spans = output.get("spans", [])

    # 3. Check for Invalid Labels if spans exist
    if predicted_spans:
        allowed_labels = entry.get("allowed_labels")
        allowed_labels = (
            [str(label) for label in allowed_labels] if allowed_labels else None
        )

        if allowed_labels:
            invalid_count = 0
            for span in predicted_spans:
                label = str(span.get("label"))
                if label not in allowed_labels:
                    invalid_count += 1

            if invalid_count == len(predicted_spans):
                return "All Invalid Labels"
            elif invalid_count > 0:
                return "Partial Invalid Labels"

        return "Success"

    # 4. Empty Prediction (Valid)

    if raw_response.strip() == "[]":
        return "Empty Prediction (Valid)"

    stripped = raw_response.strip()

    # 5. Truncation / Length Limit
    is_truncated = False
    if "json" in method_name or "occurrence" in method_name:
        if not (stripped.endswith("]") or stripped.endswith("}")):
            is_truncated = True
    # XML truncation check removed as it can end with text

    if is_truncated:
        return "Likely Truncated (Length Limit)"

    # 6. Invalid Spans (Content Mismatch)
    # If we are here, predicted_spans is empty, but raw_response is not empty/[]
    # Check if we can parse it but the spans were invalid (e.g. text not found)
    if "json" in method_name or "occurrence" in method_name:
        try:
            # Prepare entry for the method (needs text and response)
            # entry passed here might be the full result entry which has 'text'

            invalid_spans = []
            if "occurrence" in method_name:
                labeler = JSONOccurrenceSpanLabeler()
                # We need to reconstruct the entry format expected by parse_response_invalid
                # It expects 'response' and 'text'
                check_entry = {"response": raw_response, "text": entry.get("text", "")}
                invalid_spans = labeler.parse_response_invalid(check_entry)
            elif "json" in method_name:
                labeler = JSONSpanLabeler()
                check_entry = {"response": raw_response, "text": entry.get("text", "")}
                invalid_spans = labeler.parse_response_invalid(check_entry)

            if invalid_spans:
                return "Invalid Spans (Content Mismatch)"

        except Exception:
            pass  # Fall through to syntax error check

    # 7. Syntax vs Schema Errors
    if "json" in method_name or "occurrence" in method_name:
        try:
            match = re.search(r"\[.*?\]", stripped, re.DOTALL)
            if match:
                json_str = match.group()
                json.loads(json_str)
                return "Invalid Content / Schema (Parsed JSON)"
            else:
                json.loads(stripped)
                return "Invalid Content / Schema (Parsed JSON)"
        except json.JSONDecodeError:
            return "JSON Syntax Error"

    return "Parsing Error (Other)"


# %%
if __name__ == "__main__":
    # %%
    import json
    from pathlib import Path

    # path = Path("/home/semin/personal_work_ms/span_labeling/results/Qwen__Qwen3-8B_json_occurrence_constrained_uner_en_ewt_results.json")
    # path = Path("/home/semin/personal_work_ms/span_labeling/results/Qwen__Qwen3-8B_json_occurrence_constrained_english_word_synthetic_data_results.json")
    # path = Path("/home/semin/personal_work_ms/span_labeling/results/Qwen__Qwen3-8B_json_occurrence_constrained_wmt-en-ru_results.json")
    # path = Path("/home/semin/personal_work_ms/span_labeling/results/Qwen__Qwen3-8B_json_constrained_wmt-en-is_results.json")
    path = Path(
        "/home/semin/personal_work_ms/span_labeling/results/Qwen__Qwen3-8B_json_constrained_multigec_en_results.json"
    )

    results = json.loads(path.read_text())
    empty = 0
    for i, entry in enumerate(results):
        error_type = analyze_error(entry, "json")
        if error_type == "Success":
            #     print(entry["output"]["spans"])
            continue
        if error_type.startswith("Empty Prediction"):
            print("Skipping empty prediction")
            continue
        if error_type.startswith("Invalid spans"):
            # print(f"Entry {i}:")
            # print(f"Error Type: {error_type}")

            # print(entry["output"]["raw_response"])
            continue
        if error_type.startswith("All Invalid Labels"):
            continue

        if error_type.startswith("Likely Truncated"):
            continue

        print(f"Entry {i}:")
        print(f"Error Type: {error_type}")

        print(entry["output"]["raw_response"])


#     # analyze_error(results[0], "json")
# # %%
# entry
# # %%
# empty

# # %%
# len(results)
# # %%

# %%
