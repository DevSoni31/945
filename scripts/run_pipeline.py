"""
Full pipeline: retrieval → (embedding scoring placeholder) → decisions → output TSVs.
Usage: PYTHONPATH=src python scripts/run_pipeline.py --mode val
       PYTHONPATH=src python scripts/run_pipeline.py --mode test
"""

import argparse
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from er945.data import load_source, load_ground_truth
from er945.split import make_split
from er945.retrieve import InvertedRetriever
from er945.score import macro_f05

REPORTS = Path("reports")
OUTPUT  = Path("output")
REPORTS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

OUTER_CHUNK = 10_000
TOP_K       = 20       # candidates per S2/S3 record


def query_all(retriever, df, label, top_k):
    results = []
    pbar = tqdm(total=len(df), desc=label, unit="rec")
    for start in range(0, len(df), OUTER_CHUNK):
        chunk = df.iloc[start:start + OUTER_CHUNK]
        batch = retriever.query_df(chunk, top_k=top_k)
        for qid, cands in batch.items():
            for rank, s1id in enumerate(cands):
                results.append({"query_id": qid, "s1_candidate_id": s1id, "rank": rank})
        pbar.update(len(chunk))
    pbar.close()
    return pd.DataFrame(results)


def make_decisions(candidates: pd.DataFrame, s1_ids: list[str]) -> pd.DataFrame:
    """
    Placeholder decision rule: accept all candidates as matches.
    This will be replaced by the embedding scorer.
    Groups by S1 candidate and collects all S2/S3 query IDs that nominated it.
    """
    # For each S1 entity, collect all S2/S3 records that listed it as a candidate
    grouped = (
        candidates
        .groupby("s1_candidate_id")["query_id"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"s1_candidate_id": "source1_entity_id",
                         "query_id": "matched_entity_ids"})
    )
    # Ensure every S1 entity has a row, even singletons
    all_s1 = pd.DataFrame({"source1_entity_id": s1_ids})
    result = all_s1.merge(grouped, on="source1_entity_id", how="left")
    result["matched_entity_ids"] = result["matched_entity_ids"].fillna("")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    TRAIN = Path("dataset/train")
    TEST  = Path("dataset/test")

    print(f"Mode: {args.mode}")
    print("Loading data...")

    s1_train = load_source(TRAIN / "train_source1.tsv")
    s2_train = load_source(TRAIN / "train_source2.tsv")
    s3_train = load_source(TRAIN / "train_source3.tsv")
    gt       = load_ground_truth(TRAIN / "train_ground_truth.tsv")
    train_gt, val_gt = make_split(gt)

    if args.mode == "val":
        # Index: train S1 only (no leakage)
        train_s1_ids = set(train_gt.keys())
        s1_index = s1_train[s1_train["entity_id"].isin(train_s1_ids)].reset_index(drop=True)
        # Query: ALL S2/S3 records (not just linked ones)
        s2_query = s2_train.reset_index(drop=True)
        s3_query = s3_train.reset_index(drop=True)
        output_s1_ids = list(val_gt.keys())
        gt_for_score  = val_gt

    else:  # test
        # Index: ALL train S1 records
        s1_index = s1_train.reset_index(drop=True)
        s1_test  = load_source(TEST / "test_source1.tsv")
        s2_query = load_source(TEST / "test_source2.tsv")
        s3_query = load_source(TEST / "test_source3.tsv")
        output_s1_ids = s1_test["entity_id"].tolist()
        gt_for_score  = None

    print(f"S1 index size : {len(s1_index):,}")
    print(f"S2 queries    : {len(s2_query):,}")
    print(f"S3 queries    : {len(s3_query):,}")

    # Stage 1: Retrieval
    cand_path = REPORTS / f"candidates_{args.mode}_top{TOP_K}.parquet"
    if cand_path.exists():
        print(f"\nLoading cached candidates from {cand_path}")
        candidates = pd.read_parquet(cand_path)
    else:
        print("\nFitting inverted index...")
        t0 = time.time()
        retriever = InvertedRetriever()
        retriever.fit(s1_index)
        print(f"Index built in {time.time()-t0:.1f}s | vocab={len(retriever._posting):,}")

        print("Running queries...")
        df_s2 = query_all(retriever, s2_query, "S2", TOP_K)
        df_s3 = query_all(retriever, s3_query, "S3", TOP_K)
        candidates = pd.concat([df_s2, df_s3], ignore_index=True)
        candidates.to_parquet(cand_path, index=False)
        print(f"Candidates saved → {cand_path}  ({len(candidates):,} rows)")

    # Stage 2: Scoring (placeholder — embedding scorer goes here)
    print("\nStage 2: Scoring (placeholder — accepting all candidates)")

    # Stage 3: Decisions → filter candidates to only those nominated for output S1s
    print("Making decisions...")
    output_s1_set = set(output_s1_ids)
    relevant = candidates[candidates["s1_candidate_id"].isin(output_s1_set)]
    matching = make_decisions(relevant, output_s1_ids)

    # Write candidate_pairs.tsv
    cand_pairs_path = OUTPUT / "candidate_pairs.tsv"
    (
        relevant
        .groupby("s1_candidate_id")["query_id"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"s1_candidate_id": "source1_entity_id",
                         "query_id": "candidate_entity_ids"})
        .to_csv(cand_pairs_path, sep="\t", index=False)
    )

    # Write matching_results.tsv
    matching_path = OUTPUT / "matching_results.tsv"
    matching.to_csv(matching_path, sep="\t", index=False)
    print(f"Outputs written → {matching_path}, {cand_pairs_path}")

    # Stage 4: Score (val mode only)
    if gt_for_score:
        predicted = {
            row.source1_entity_id: (
                set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
            )
            for row in matching.itertuples(index=False)
        }
        score = macro_f05(gt_for_score, predicted)
        print(f"\nVal macro F0.5 = {score:.4f}")
        (REPORTS / f"score_{args.mode}.txt").write_text(f"macro_f05={score:.4f}\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
