from typing import Any


def compute_overlap_counts(
    predicted_spans: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
    hard_matching: bool = True,
) -> dict[str, int]:
    """
    Compute character-level overlap counts for one example.

    Returns the three quantities needed to compute precision, recall,
    and F1 at any aggregation level (per-example, per-dataset, per-task):
        - overlap_chars: total matched character weight
        - predicted_chars: total predicted span weight (precision denominator)
        - gold_chars:      total gold span weight     (recall denominator)

    Each zero-length span (e.g., GEC missing-span insertion points)
    contributes a weight of 1. A zero-length predicted span matches a
    zero-length gold span iff their positions are equal and (under hard
    matching) their labels match. Normal-length spans contribute their
    character length and overlap is matched character-by-character.

    Args:
        predicted_spans: list of dicts with 'start', 'end', 'label'
        gold_spans:      list of dicts with 'start', 'end', 'label'
        hard_matching:   if True, labels must match for a span to count

    Returns:
        Dict with keys overlap_chars, predicted_chars, gold_chars.
    """
    if not predicted_spans and not gold_spans:
        # Nothing to predict, nothing to find. Treated as empty contribution.
        return {"overlap_chars": 0, "predicted_chars": 0, "gold_chars": 0}

    def span_weight(s):
        return max(1, s["end"] - s["start"])

    predicted_chars = sum(span_weight(s) for s in predicted_spans)
    gold_chars = sum(span_weight(s) for s in gold_spans)

    if not predicted_spans or not gold_spans:
        return {
            "overlap_chars": 0,
            "predicted_chars": predicted_chars,
            "gold_chars": gold_chars,
        }

    matched_pairs: set = set()
    overlap_chars = 0

    for hyp_idx, hyp_span in enumerate(predicted_spans):
        hyp_start = hyp_span["start"]
        hyp_end = hyp_span["end"]
        hyp_label = str(hyp_span["label"])

        # Zero-length predicted span: only matches zero-length gold spans
        # at the same position with matching label.
        if hyp_start == hyp_end:
            for gold_idx, gold_span in enumerate(gold_spans):
                if gold_span["start"] != gold_span["end"]:
                    continue
                if gold_span["start"] != hyp_start:
                    continue
                if hard_matching and str(gold_span["label"]) != hyp_label:
                    continue
                key = ("zero", hyp_idx, gold_idx)
                if key not in matched_pairs:
                    matched_pairs.add(key)
                    overlap_chars += 1
                    break
            continue

        # Normal-length predicted span: character-by-character overlap
        # against non-zero-length gold spans.
        for hyp_pos in range(hyp_start, hyp_end):
            for gold_idx, gold_span in enumerate(gold_spans):
                gold_start = gold_span["start"]
                gold_end = gold_span["end"]
                gold_label = str(gold_span["label"])

                # Zero-length gold spans only match zero-length predictions
                # (handled above).
                if gold_start == gold_end:
                    continue

                if gold_start <= hyp_pos < gold_end:
                    if hard_matching and gold_label != hyp_label:
                        continue
                    key = (hyp_pos, gold_idx)
                    if key not in matched_pairs:
                        matched_pairs.add(key)
                        overlap_chars += 1
                        break

    return {
        "overlap_chars": overlap_chars,
        "predicted_chars": predicted_chars,
        "gold_chars": gold_chars,
    }


def f1_from_counts(
    overlap_chars: int,
    predicted_chars: int,
    gold_chars: int,
) -> dict[str, float]:
    """
    Compute precision, recall, F1 from pooled overlap counts.

    Convention for empty cases:
      - both predicted and gold empty -> P=R=F1=1.0 (vacuous match)
      - predicted empty, gold non-empty -> P=R=F1=0.0
      - gold empty, predicted non-empty -> P=R=F1=0.0
    """
    if predicted_chars == 0 and gold_chars == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    precision = overlap_chars / predicted_chars if predicted_chars > 0 else 0.0
    recall = overlap_chars / gold_chars if gold_chars > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_overlap_f1(
    predicted_spans: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
    hard_matching: bool = True,
) -> dict[str, Any]:
    """
    Compute precision, recall, F1 with character-level overlap matching
    for a single example. Kept for backward compatibility; new code
    should use `compute_overlap_counts` and pool across examples before
    calling `f1_from_counts`.
    """
    counts = compute_overlap_counts(
        predicted_spans, gold_spans, hard_matching=hard_matching
    )
    metrics = f1_from_counts(**counts)
    return {**metrics, **counts}


def evaluate(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    hard_matching: bool = True,
) -> dict[str, Any]:
    """Per-example evaluation. Returns counts and F1 for one example.
    Kept for backward compatibility."""
    return compute_overlap_f1(predicted, gold, hard_matching=hard_matching)
