from pathlib import Path
import time
import json
import pickle
from tqdm import tqdm

from er945.data import load_source, load_ground_truth
from er945.split import make_split
from er945.retrieve import InvertedRetriever
from er945.evaluate import candidate_recall, recall_summary

TRAIN       = Path("dataset/train")
OUTER_CHUNK = 10_000
REPORTS     = Path("reports")
REPORTS.mkdir(exist_ok=True)

print("Loading data...")
s1 = load_source(TRAIN / "train_source1.tsv")
s2 = load_source(TRAIN / "train_source2.tsv")
s3 = load_source(TRAIN / "train_source3.tsv")
gt = load_ground_truth(TRAIN / "train_ground_truth.tsv")

_, val_gt = make_split(gt)

s1_full = s1.reset_index(drop=True)
val_linked_ids = {m for matches in val_gt.values() for m in matches}
s2_val = s2[s2["entity_id"].isin(val_linked_ids)].reset_index(drop=True)
s3_val = s3[s3["entity_id"].isin(val_linked_ids)].reset_index(drop=True)

print(f"S1 index size (full) : {len(s1_full):,}")
print(f"Val S2 queries       : {len(s2_val):,}")
print(f"Val S3 queries       : {len(s3_val):,}")
print(f"Val S1 entities      : {len(val_gt):,}")

# --- Index: build once, cache to disk ---
index_cache = REPORTS / "inverted_index.pkl"
if index_cache.exists():
    print(f"\nLoading cached index from {index_cache} ...")
    with open(index_cache, "rb") as f:
        retriever = pickle.load(f)
    print(f"Loaded. Vocab={len(retriever._posting):,}")
else:
    print("\nFitting inverted index on ALL S1...")
    t0 = time.time()
    retriever = InvertedRetriever()
    retriever.fit(s1_full)
    print(f"Index built in {time.time()-t0:.1f}s | vocab={len(retriever._posting):,}")
    with open(index_cache, "wb") as f:
        pickle.dump(retriever, f)
    print(f"Index cached → {index_cache}")


def run_queries(retriever, df, top_k, label):
    """Run queries with per-chunk progress. Returns {query_id: [s1_ids]}."""
    cache = REPORTS / f"cands_{label}_top{top_k}.pkl"
    if cache.exists():
        print(f"  Loading cached {label} k={top_k} from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    results = {}
    pbar = tqdm(total=len(df), desc=f"{label} k={top_k}", unit="rec")
    for start in range(0, len(df), OUTER_CHUNK):
        chunk = df.iloc[start:start + OUTER_CHUNK]
        results.update(retriever.query_df(chunk, top_k=top_k))
        pbar.update(len(chunk))
    pbar.close()

    with open(cache, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved {label} k={top_k} → {cache}")
    return results


all_results = {}

for top_k in [5, 10, 20]:
    # Skip if this top_k was already fully evaluated
    report_file = REPORTS / f"retrieval_benchmark_top{top_k}.json"
    if report_file.exists():
        print(f"\n--- top_k = {top_k} (cached) ---")
        row = json.loads(report_file.read_text())
        all_results[str(top_k)] = row
        for k, v in row.items():
            print(f"  {k:<25}: {v:,}" if isinstance(v, int) else f"  {k:<25}: {v}")
        continue

    print(f"\n--- top_k = {top_k} ---")
    t0 = time.time()
    cands_s2 = run_queries(retriever, s2_val, top_k, "S2")
    cands_s3 = run_queries(retriever, s3_val, top_k, "S3")
    merged   = {**cands_s2, **cands_s3}
    elapsed  = time.time() - t0

    recalls = candidate_recall(val_gt, merged)
    summary = recall_summary(recalls)
    total_cands = sum(len(v) for v in merged.values())

    row = {
        "top_k":               top_k,
        "query_time_s":        round(elapsed, 1),
        "macro_recall":        round(summary["macro_recall"], 4),
        "perfect_recall_pct":  round(summary["perfect_recall_pct"], 2),
        "zero_recall_pct":     round(summary["zero_recall_pct"], 2),
        "total_candidates":    total_cands,
        "avg_cands_per_query": round(total_cands / max(len(merged), 1), 1),
    }
    all_results[str(top_k)] = row

    for k, v in row.items():
        print(f"  {k:<25}: {v:,}" if isinstance(v, int) else f"  {k:<25}: {v}")

    report_file.write_text(json.dumps(row, indent=2))
    print(f"  Saved → {report_file}")

(REPORTS / "retrieval_benchmark_summary.json").write_text(
    json.dumps(all_results, indent=2)
)
print("\nFull summary → reports/retrieval_benchmark_summary.json")
print("Done.")
