from __future__ import annotations

from collections import defaultdict


def candidate_recall(
    gt: dict[str, set[str]],
    candidates: dict[str, list[str]],
) -> dict[str, float]:
    """
    For each Source 1 entity with at least one true link,
    measure what fraction of its true S2/S3 links appear in its candidate list.

    candidates: {query_s2_or_s3_id: [s1_candidate_id, ...]}
    gt:         {s1_id: {s2_or_s3_ids}}

    Returns per-S1 recall and macro average.
    """
    # Invert gt: {s2_or_s3_id: s1_id}
    link_to_s1: dict[str, str] = {}
    for s1_id, matches in gt.items():
        for m in matches:
            link_to_s1[m] = s1_id

    # For each S1 entity, collect which true links were retrieved
    retrieved: dict[str, set[str]] = defaultdict(set)
    for query_id, s1_cands in candidates.items():
        true_s1 = link_to_s1.get(query_id)
        if true_s1 is None:
            continue
        if true_s1 in s1_cands:
            retrieved[true_s1].add(query_id)

    recalls: dict[str, float] = {}
    for s1_id, matches in gt.items():
        if not matches:
            continue  # skip singletons
        recalls[s1_id] = len(retrieved[s1_id]) / len(matches)

    return recalls


def recall_summary(recalls: dict[str, float]) -> dict[str, float]:
    if not recalls:
        return {"macro_recall": 0.0, "n_entities": 0}
    values = list(recalls.values())
    perfect = sum(1 for v in values if v == 1.0)
    zero    = sum(1 for v in values if v == 0.0)
    return {
        "macro_recall":       sum(values) / len(values),
        "perfect_recall_pct": 100 * perfect / len(values),
        "zero_recall_pct":    100 * zero   / len(values),
        "n_entities":         len(values),
    }
