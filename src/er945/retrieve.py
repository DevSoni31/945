from __future__ import annotations

from collections import defaultdict
from math import log
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from er945.normalize import combined_text, normalize_name, normalize_address

# Tokens appearing in more than this fraction of S1 records are stop-tokens
_MAX_DF_FRAC = 0.01


def _tokenize(text: str) -> list[str]:
    return text.split() if text else []


class InvertedRetriever:
    """
    Token-overlap inverted index with IDF weighting.
    For each S2/S3 query, scores S1 candidates by sum of IDF weights
    of shared tokens, then returns top-k.
    """

    def __init__(self) -> None:
        self._posting: dict[str, list[int]] = {}   # token -> [s1 row indices]
        self._idf: dict[str, float] = {}
        self._index_ids: list[str] = []
        self._n_docs: int = 0

    def fit(self, s1: pd.DataFrame) -> None:
        self._index_ids = s1["entity_id"].tolist()
        self._n_docs = len(s1)
        max_df = int(_MAX_DF_FRAC * self._n_docs)

        # Build raw postings
        raw: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(tqdm(
            s1.itertuples(index=False),
            total=len(s1),
            desc="Building index",
            unit="rec",
        )):
            text = combined_text(row.business_name, row.business_address)
            for tok in set(_tokenize(text)):  # set: one posting per doc
                raw[tok].append(idx)

        # Drop stop-tokens, compute IDF
        self._posting = {}
        self._idf = {}
        for tok, docs in raw.items():
            if len(docs) > max_df:
                continue
            self._posting[tok] = docs
            self._idf[tok] = log(self._n_docs / len(docs))

    def query_df(
        self,
        df: pd.DataFrame,
        top_k: int = 10,
    ) -> dict[str, list[str]]:
        results = {}
        for row in df.itertuples(index=False):
            text = combined_text(row.business_name, row.business_address)
            tokens = set(_tokenize(text))

            scores: dict[int, float] = defaultdict(float)
            for tok in tokens:
                if tok not in self._posting:
                    continue
                idf = self._idf[tok]
                for doc_idx in self._posting[tok]:
                    scores[doc_idx] += idf

            if not scores:
                results[row.entity_id] = []
                continue

            top = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
            results[row.entity_id] = [self._index_ids[i] for i in top]

        return results
