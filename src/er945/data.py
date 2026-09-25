from __future__ import annotations

from pathlib import Path

import pandas as pd


_COLS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: Path | str) -> pd.DataFrame:
    """Load one source TSV; keep only the four canonical columns."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in _COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[_COLS].copy()


def load_ground_truth(path: Path | str) -> dict[str, set[str]]:
    """Return {source1_entity_id: set_of_matched_ids}. Empty set = singleton."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "")
        matches = {m.strip() for m in raw.split(",") if m.strip()}
        gt[s1_id] = matches
    return gt
