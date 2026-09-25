import pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from er945.data import load_source, load_ground_truth
from er945.split import make_split
from er945.features import build_features
from er945.match import PairMatcher
from er945.score import macro_f05

TRAIN   = Path("dataset/train")
REPORTS = Path("reports")
OUTPUT  = Path("output")
OUTPUT.mkdir(exist_ok=True)
TOP_K   = 5
CHUNK   = 50_000

print("Loading data...")
s1 = load_source(TRAIN / "train_source1.tsv")
s2 = load_source(TRAIN / "train_source2.tsv")
s3 = load_source(TRAIN / "train_source3.tsv")
gt = load_ground_truth(TRAIN / "train_ground_truth.tsv")
_, val_gt = make_split(gt)
link_to_s1 = {m: s1id for s1id, matches in gt.items() for m in matches}

# ── Stage 1: Build pairs ──────────────────────────────────────────────────────
pairs_cache = REPORTS / f"pairs_top{TOP_K}.parquet"
if pairs_cache.exists():
    print(f"Loading cached pairs from {pairs_cache} ...")
    pairs = pd.read_parquet(pairs_cache)
    print(f"Loaded {len(pairs):,} pairs")
else:
    print("Loading cached candidates...")
    with open(REPORTS / f"cands_S2_top{TOP_K}.pkl", "rb") as f:
        cands_s2 = pickle.load(f)
    with open(REPORTS / f"cands_S3_top{TOP_K}.pkl", "rb") as f:
        cands_s3 = pickle.load(f)
    merged = {**cands_s2, **cands_s3}

    print(f"Building pairs from {len(merged):,} queries...")
    rows = []
    for qid, s1_cands in tqdm(merged.items(), desc="Expanding pairs", unit="query"):
        true_s1 = link_to_s1.get(qid)
        for rank, s1id in enumerate(s1_cands):
            rows.append({
                "query_id":        qid,
                "s1_candidate_id": s1id,
                "rank":            rank,
                "true_s1":         true_s1,
                "label":           1 if s1id == true_s1 else 0,
            })
    pairs = pd.DataFrame(rows)
    pairs.to_parquet(pairs_cache, index=False)
    print(f"Pairs saved → {pairs_cache}  ({len(pairs):,} rows)")

pos = int(pairs["label"].sum())
neg = len(pairs) - pos
print(f"Positives: {pos:,} | Negatives: {neg:,} | Ratio 1:{neg//max(pos,1)}")

# ── Stage 2: Split 80/20 by query_id hash (entity-safe, reproducible) ────────
# We have only val-linked queries in our candidate cache.
# Split them 80/20 by S1 entity to train and evaluate the pair scorer.
val_s1_list  = list(val_gt.keys())
np.random.seed(42)
np.random.shuffle(val_s1_list)
cut          = int(0.8 * len(val_s1_list))
pair_train_s1 = set(val_s1_list[:cut])
pair_val_s1   = set(val_s1_list[cut:])

in_train     = pairs["true_s1"].isin(pair_train_s1)
train_pairs  = pairs[in_train].reset_index(drop=True)
val_pairs    = pairs[~in_train].reset_index(drop=True)
print(f"Pair-train: {len(train_pairs):,} | Pair-val: {len(val_pairs):,}")

# The final F0.5 is evaluated against the FULL val_gt at the end.
# pair_val_s1 is only used for threshold tuning.
pair_val_gt = {s1id: val_gt[s1id] for s1id in pair_val_s1}

# ── Stage 3: Sample train pairs ───────────────────────────────────────────────
MAX_TRAIN = 2_000_000
if len(train_pairs) > MAX_TRAIN:
    train_pairs = train_pairs.sample(MAX_TRAIN, random_state=42).reset_index(drop=True)
    print(f"Sampled: {len(train_pairs):,} "
          f"({train_pairs['label'].sum():,} pos, "
          f"{(train_pairs['label']==0).sum():,} neg)")
    print(f"Sampled: {len(train_pairs):,} "
          f"({train_pairs['label'].sum():,} pos, "
          f"{(train_pairs['label']==0).sum():,} neg)")

# ── Stage 4: Feature building ─────────────────────────────────────────────────
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

print("\nBuilding train features...")
train_feats = build_features_chunked(
    train_pairs, s1, s2, s3,
    REPORTS / f"train_feats_top{TOP_K}.parquet", "train"
)

print("\nBuilding val features...")
val_feats = build_features_chunked(
    val_pairs, s1, s2, s3,
    REPORTS / f"val_feats_top{TOP_K}.parquet", "val"
)

# ── Stage 5: Train matcher ────────────────────────────────────────────────────
matcher_cache = REPORTS / "matcher.pkl"
if matcher_cache.exists():
    print(f"\nLoading cached matcher from {matcher_cache} ...")
    with open(matcher_cache, "rb") as f:
        matcher = pickle.load(f)
else:
    print("\nTraining matcher...")
    t0      = time.time()
    matcher = PairMatcher(threshold=0.5)
    matcher.fit(train_feats, train_pairs["label"].iloc[:len(train_feats)])
    print(f"Trained in {time.time()-t0:.1f}s")
    with open(matcher_cache, "wb") as f:
        pickle.dump(matcher, f)
    print(f"Matcher saved → {matcher_cache}")

# ── Stage 6: Threshold sweep on pair-val F0.5 ────────────────────────────────
print("\nScoring val pairs...")
val_pairs = val_pairs.iloc[:len(val_feats)].copy()
val_pairs["score"] = matcher.predict_proba(val_feats)

print("Tuning threshold on pair-val F0.5...")
best_thresh, best_f05 = 0.5, 0.0
sweep_rows = []
for thresh in np.arange(0.1, 0.96, 0.05):
    preds: dict[str, set[str]] = {s1id: set() for s1id in pair_val_gt}
    for row in val_pairs[val_pairs["score"] >= thresh].itertuples(index=False):
        if row.s1_candidate_id in preds:
            preds[row.s1_candidate_id].add(row.query_id)
    f05    = macro_f05(pair_val_gt, preds)
    marker = " ◀ best" if f05 > best_f05 else ""
    print(f"  thresh={thresh:.2f}  F0.5={f05:.4f}{marker}")
    sweep_rows.append({"threshold": round(float(thresh), 2), "val_f05": round(f05, 4)})
    if f05 > best_f05:
        best_f05, best_thresh = f05, float(thresh)

pd.DataFrame(sweep_rows).to_csv(REPORTS / "threshold_sweep.csv", index=False)
print(f"\nBest threshold : {best_thresh:.2f}")
print(f"Best val F0.5  : {best_f05:.4f}")

# ── Stage 7: Save val predictions ─────────────────────────────────────────────
matcher.threshold = best_thresh
with open(matcher_cache, "wb") as f:
    pickle.dump(matcher, f)

# Score ALL val pairs (train+val split) for final F0.5 estimate
print("\nBuilding full val predictions...")
all_val_feats = build_features_chunked(
    pairs, s1, s2, s3,
    REPORTS / f"all_feats_top{TOP_K}.parquet", "all"
)
pairs["score"] = matcher.predict_proba(all_val_feats)

preds_final: dict[str, set[str]] = {s1id: set() for s1id in val_gt}
for row in tqdm(
    pairs[pairs["score"] >= best_thresh].itertuples(index=False),
    desc="Writing predictions", unit="pair"
):
    if row.s1_candidate_id in preds_final:
        preds_final[row.s1_candidate_id].add(row.query_id)

full_f05 = macro_f05(val_gt, preds_final)
print(f"\nFull val F0.5 (all 442K entities) : {full_f05:.4f}")

pd.DataFrame([
    {"source1_entity_id": s1id,
     "matched_entity_ids": ",".join(sorted(m)) if m else ""}
    for s1id, m in preds_final.items()
]).to_csv(OUTPUT / "val_matching_results.tsv", sep="\t", index=False)
print("Val predictions → output/val_matching_results.tsv")
print("\nDone.")
