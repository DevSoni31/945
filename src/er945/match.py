from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "rank",
    "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jaccard",
    "addr_ratio", "addr_partial", "addr_token_sort", "addr_jaccard",
    "combined_ratio",
    "q_missing_addr", "s_missing_addr", "country_match",
]


class PairMatcher:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._scaler = StandardScaler()
        self._clf    = LogisticRegression(max_iter=1000, C=1.0)

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        X = features[FEATURE_COLS].values.astype(np.float32)
        X = self._scaler.fit_transform(X)
        self._clf.fit(X, labels.values)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = features[FEATURE_COLS].values.astype(np.float32)
        X = self._scaler.transform(X)
        return self._clf.predict_proba(X)[:, 1]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(int)
