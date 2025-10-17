from typing import Any


def compute_overlap_f1(
    predicted_spans: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
    hard_matching: bool = True,
) -> dict[str, float | int]:
    """
    Compute precision, recall, F1 with character-level overlap matching.
    Based on factgenie's overlap F1 implementation.

    Args:
        predicted_spans: List of dicts with 'start', 'end', 'label'
        gold_spans: List of dicts with 'start', 'end', 'label'

    Returns:
        Dict with precision, recall, f1, and breakdown
    """
    if not predicted_spans and not gold_spans:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not predicted_spans or not gold_spans:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Track matched positions to avoid double counting
    matched_pairs = set()  # (hyp_pos, gold_idx)

    # Calculate total lengths
    hyp_length = sum(s["end"] - s["start"] for s in predicted_spans)
    ref_length = sum(s["end"] - s["start"] for s in gold_spans)
    overlap_length = 0

    # For each predicted span
    for hyp_span in predicted_spans:
        hyp_start = hyp_span["start"]
        hyp_end = hyp_span["end"]
        hyp_label = hyp_span["label"]

        # For each character position in this predicted span
        for hyp_pos in range(hyp_start, hyp_end):
            # Check all gold spans for matches at this position
            for gold_idx, gold_span in enumerate(gold_spans):
                gold_start = gold_span["start"]
                gold_end = gold_span["end"]
                gold_label = gold_span["label"]

                # Does this gold span cover this position?
                if gold_start <= hyp_pos < gold_end:
                    # Labels must match (hard matching)
                    if hard_matching and hyp_label != gold_label:
                        continue

                    # If not already matched, count it
                    if (hyp_pos, gold_idx) not in matched_pairs:
                        matched_pairs.add((hyp_pos, gold_idx))
                        overlap_length += 1
                        break  # Found match for this position

    # Calculate metrics
    precision = overlap_length / hyp_length if hyp_length > 0 else 0
    recall = overlap_length / ref_length if ref_length > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "overlap_chars": overlap_length,
        "predicted_chars": hyp_length,
        "gold_chars": ref_length,
    }


def evaluate(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    hard_matching: bool = True,
) -> dict[str, float | int]:
    """Evaluate with overlap F1"""
    return compute_overlap_f1(predicted, gold, hard_matching=hard_matching)
