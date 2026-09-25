from __future__ import annotations

import pandas as pd
import numpy as np
from rapidfuzz import fuzz

from er945.normalize import normalize_name, normalize_address, combined_text


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_features(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> pd.DataFrame:
    """
    pairs columns: query_id, s1_candidate_id, rank
    Returns a feature DataFrame aligned to pairs, with no labels.
    """
    s1_map = {r.entity_id: r for r in s1.itertuples(index=False)}
    s2_map = {r.entity_id: r for r in s2.itertuples(index=False)}
    s3_map = {r.entity_id: r for r in s3.itertuples(index=False)}
    query_map = {**s2_map, **s3_map}

    rows = []
    for p in pairs.itertuples(index=False):
        qr = query_map.get(p.query_id)
        sr = s1_map.get(p.s1_candidate_id)
        if qr is None or sr is None:
            rows.append(None)
            continue

        qn = normalize_name(qr.business_name)
        sn = normalize_name(sr.business_name)
        qa = normalize_address(qr.business_address)
        sa = normalize_address(sr.business_address)

        name_ratio      = fuzz.ratio(qn, sn) / 100
        name_partial    = fuzz.partial_ratio(qn, sn) / 100
        name_token_sort = fuzz.token_sort_ratio(qn, sn) / 100
        name_token_set  = fuzz.token_set_ratio(qn, sn) / 100
        name_jaccard    = _token_jaccard(qn, sn)

        addr_ratio      = fuzz.ratio(qa, sa) / 100          if qa and sa else 0.0
        addr_partial    = fuzz.partial_ratio(qa, sa) / 100  if qa and sa else 0.0
        addr_token_sort = fuzz.token_sort_ratio(qa, sa) / 100 if qa and sa else 0.0
        addr_jaccard    = _token_jaccard(qa, sa)             if qa and sa else 0.0

        q_missing_addr  = 1 if not qa else 0
        s_missing_addr  = 1 if not sa else 0
        country_match   = 1 if qr.country == sr.country else 0

        combined_q = combined_text(qr.business_name, qr.business_address)
        combined_s = combined_text(sr.business_name, sr.business_address)
        combined_ratio = fuzz.ratio(combined_q, combined_s) / 100

        rows.append({
            "query_id":         p.query_id,
            "s1_candidate_id":  p.s1_candidate_id,
            "rank":             p.rank,
            "name_ratio":       name_ratio,
            "name_partial":     name_partial,
            "name_token_sort":  name_token_sort,
            "name_token_set":   name_token_set,
            "name_jaccard":     name_jaccard,
            "addr_ratio":       addr_ratio,
            "addr_partial":     addr_partial,
            "addr_token_sort":  addr_token_sort,
            "addr_jaccard":     addr_jaccard,
            "combined_ratio":   combined_ratio,
            "q_missing_addr":   q_missing_addr,
            "s_missing_addr":   s_missing_addr,
            "country_match":    country_match,
        })

    return pd.DataFrame([r for r in rows if r is not None])
