"""
Generate test set predictions using the trained matcher.
Usage: PYTHONPATH=src python scripts/predict_test.py
"""
import pickle, time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from er945.data import load_source
from er945.features import build_features
from er945.retrieve import InvertedRetriever

TRAIN   = Path("dataset/train")
TEST    = Path("dataset/test")
REPORTS = Path("reports")
OUTPUT  = Path("output")
OUTPUT.mkdir(exist_ok=True)

TOP_K  = 5
CHUNK  = 50_000

print("Loading data...")
s1_train = load_source(TRAIN / "train_source1.tsv")
s1_test  = load_source(TEST  / "test_source1.tsv")
s2_test  = load_source(TEST  / "test_source2.tsv")
s3_test  = load_source(TEST  / "test_source3.tsv")

print(f"Test S1: {len(s1_test):,} | S2: {len(s2_test):,} | S3: {len(s3_test):,}")

# ── Stage 1: Build test index + retrieve candidates ───────────────────────────
test_cands_cache = REPORTS / f"cands_test_top{TOP_K}.parquet"
if test_cands_cache.exists():
    print(f"Loading cached test candidates from {test_cands_cache} ...")
    test_pairs = pd.read_parquet(test_cands_cache)
else:
    # Fit index on ALL train S1 (full dataset, no split needed for test inference)
    index_cache = REPORTS / "inverted_index_full.pkl"
    if index_cache.exists():
        print("Loading cached full S1 index...")
        with open(index_cache, "rb") as f:
            retriever = pickle.load(f)
    else:
        print("Fitting inverted index on all train S1...")
        retriever = InvertedRetriever()
        retriever.fit(s1_train)
        with open(index_cache, "wb") as f:
            pickle.dump(retriever, f)
        print(f"Index cached → {index_cache}")

    def query_all(retriever, df, label):
        rows = []
        pbar = tqdm(total=len(df), desc=label, unit="rec")
        for start in range(0, len(df), CHUNK):
            chunk = df.iloc[start:start + CHUNK]
            batch = retriever.query_df(chunk, top_k=TOP_K)
            for qid, cands in batch.items():
                for rank, s1id in enumerate(cands):
                    rows.append({"query_id": qid, "s1_candidate_id": s1id, "rank": rank})
            pbar.update(len(chunk))
        pbar.close()
        return pd.DataFrame(rows)

    print("Querying S2 test...")
    df_s2 = query_all(retriever, s2_test, "S2-test")
    print("Querying S3 test...")
    df_s3 = query_all(retriever, s3_test, "S3-test")
    test_pairs = pd.concat([df_s2, df_s3], ignore_index=True)
    test_pairs.to_parquet(test_cands_cache, index=False)
    print(f"Test candidates saved → {test_cands_cache}  ({len(test_pairs):,} rows)")

print(f"Test candidate pairs: {len(test_pairs):,}")

# ── Stage 2: Build features ───────────────────────────────────────────────────
feat_cache = REPORTS / f"test_feats_top{TOP_K}.parquet"

def build_features_chunked(pairs_df, s1, s2, s3, cache_path, label):
    if cache_path.exists():
        print(f"Loading cached {label} features from {cache_path} ...")
        return pd.read_parquet(cache_path)
    all_feats = []
    partial   = cache_path.with_suffix(".partial.parquet")
    pbar      = tqdm(total=len(pairs_df), desc=f"Features [{label}]", unit="pair")
    for i in range((len(pairs_df) + CHUNK - 1) // CHUNK):
        chunk = pairs_df.iloc[i * CHUNK:(i + 1) * CHUNK]
        all_feats.append(build_features(chunk, s1, s2, s3))
        pbar.update(len(chunk))
        if (i + 1) % 10 == 0:
            pd.concat(all_feats, ignore_index=True).to_parquet(partial, index=False)
    pbar.close()
    result = pd.concat(all_feats, ignore_index=True)
    result.to_parquet(cache_path, index=False)
    if partial.exists(): partial.unlink()
    print(f"{label} features → {cache_path}  shape={result.shape}")
    return result

print("\nBuilding test features...")
test_feats = build_features_chunked(
    test_pairs, s1_train, s2_test, s3_test, feat_cache, "test"
)

# ── Stage 3: Score + predict ──────────────────────────────────────────────────
print("\nLoading matcher...")
with open(REPORTS / "matcher.pkl", "rb") as f:
    matcher = pickle.load(f)
print(f"Threshold: {matcher.threshold}")

print("Scoring test pairs...")
test_pairs = test_pairs.iloc[:len(test_feats)].copy()
test_pairs["score"] = matcher.predict_proba(test_feats)

# ── Stage 4: Build predictions for all test S1 entities ───────────────────────
print("Building predictions...")
s1_test_ids = s1_test["entity_id"].tolist()
preds: dict[str, set[str]] = {s1id: set() for s1id in s1_test_ids}

accepted = test_pairs[test_pairs["score"] >= matcher.threshold]
for row in tqdm(accepted.itertuples(index=False), desc="Writing preds", total=len(accepted)):
    if row.s1_candidate_id in preds:
        preds[row.s1_candidate_id].add(row.query_id)

matched   = sum(1 for v in preds.values() if v)
singleton = sum(1 for v in preds.values() if not v)
print(f"S1 entities with matches : {matched:,}")
print(f"S1 singletons            : {singleton:,}")

# ── Stage 5: Write output TSVs ────────────────────────────────────────────────
matching_path = OUTPUT / "matching_results.tsv"
pd.DataFrame([
    {"source1_entity_id": s1id,
     "matched_entity_ids": ",".join(sorted(m)) if m else ""}
    for s1id, m in preds.items()
]).to_csv(matching_path, sep="\t", index=False)
print(f"Matching results → {matching_path}")

# Candidate pairs TSV
(
    accepted
    .groupby("s1_candidate_id")["query_id"]
    .apply(lambda x: ",".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"s1_candidate_id": "source1_entity_id",
                     "query_id": "candidate_entity_ids"})
    .to_csv(OUTPUT / "candidate_pairs.tsv", sep="\t", index=False)
)
print(f"Candidate pairs   → output/candidate_pairs.tsv")
print("\nDone. Ready to validate and submit.")
