from __future__ import annotations

from collections.abc import Mapping
from typing import AbstractSet


def entity_f05(
    truth: AbstractSet[str],
    predicted: AbstractSet[str],
) -> float:
    """Challenge F0.5 for one Source 1 entity."""
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0

    true_positives = len(truth & predicted)
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(predicted)
    recall = true_positives / len(truth)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def macro_f05(
    truth: Mapping[str, AbstractSet[str]],
    predicted: Mapping[str, AbstractSet[str]],
) -> float:
    """Average over every Source 1 entity, including singletons."""
    if not truth:
        raise ValueError("Ground truth has no Source 1 entities")

    missing = truth.keys() - predicted.keys()
    extra = predicted.keys() - truth.keys()
    if missing or extra:
        raise ValueError(
            f"Prediction IDs differ from truth: "
            f"{len(missing)} missing, {len(extra)} extra"
        )

    return sum(
        entity_f05(matches, predicted[entity_id])
        for entity_id, matches in truth.items()
    ) / len(truth)
