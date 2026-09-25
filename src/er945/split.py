from __future__ import annotations

import hashlib


def _fold(entity_id: str, n_folds: int = 5) -> int:
    """Deterministic fold assignment from entity_id."""
    digest = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
    return digest % n_folds


def make_split(
    ground_truth: dict[str, set[str]],
    val_folds: frozenset[int] = frozenset({0}),
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    Split by Source 1 entity_id.
    All S2/S3 records belonging to a given S1 entity go into the same partition.
    Returns (train_gt, val_gt).
    """
    train_gt: dict[str, set[str]] = {}
    val_gt: dict[str, set[str]] = {}

    for s1_id, matches in ground_truth.items():
        if _fold(s1_id) in val_folds:
            val_gt[s1_id] = matches
        else:
            train_gt[s1_id] = matches

    return train_gt, val_gt
