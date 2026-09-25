# ER-945 Handover Document

**From:** Dev
**To:** Shashwat
**Target:** 0.99+ Macro F0.5

---

## 1. What this challenge is

**Task:** Given three sources of business entity records (S1, S2, S3), find which S2 and S3 records refer to the same real-world business as each S1 record.

**Metric:** Macro-averaged F0.5 across all S1 entities. F0.5 weights precision twice as heavily as recall: attaching a wrong match costs far more than missing a true one.

**Submission requires two files. Both are mandatory:**

1. `output/matching_results.tsv` - columns: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs, empty string if true singleton)
2. `output/candidate_pairs.tsv` - columns: `source1_entity_id`, `candidate_entity_ids` (all candidates before filtering)

## 2. Repository structure

```text
945/
├── src/er945/              # Core library
│   ├── data.py             # load_source(), load_ground_truth()
│   ├── normalize.py        # normalize_name(), normalize_address(), combined_text()
│   ├── retrieve.py         # InvertedRetriever - IDF-weighted token index
│   ├── features.py         # build_features() - 16 string-similarity features per pair
│   ├── match.py             # PairMatcher - logistic regression wrapper
│   ├── score.py             # macro_f05() - exact competition metric
│   ├── evaluate.py          # candidate_recall(), recall_summary()
│   └── split.py             # make_split() - reproducible 80/20 train/val split
│
├── scripts/
│   ├── benchmark_retrieval.py   # Benchmarks retriever recall at top_k=5,10,20
│   ├── train_matcher.py         # Trains matcher, outputs val F0.5
│   ├── predict_test.py           # Generates test submission (~12h locally)
│   └── run_pipeline.py           # Older end-to-end pipeline (superseded)
│
├── reports/                # ALL cached computation - GET FROM DEV, DO NOT RERUN
├── output/                 # Submission files
├── dataset/                # Raw data - NOT in git
└── tests/                  # Unit tests (pytest)
```

## 3. Files Dev must share with Shashwat - critical

These represent approximately 15 hours of CPU computation. Get them from Dev via Google Drive or USB:

| File | Size | Description | Compute time |
|---|---:|---|---:|
| `reports/inverted_index.pkl` | ~800 MB | IDF-weighted index on all 2.2M S1 records | 17 sec |
| `reports/cands_S2_top5.pkl` | ~300 MB | Top-5 S1 candidates for 740K validation S2 queries | 51 min |
| `reports/cands_S3_top5.pkl` | ~320 MB | Top-5 S1 candidates for 790K validation S3 queries | 82 min |
| `reports/pairs_top5.parquet` | ~600 MB | 7.6M labelled candidate pairs | 2 min |
| `reports/train_feats_top5.parquet` | ~400 MB | 2M training feature rows (16 features) | 29 min |
| `reports/val_feats_top5.parquet` | ~300 MB | 1.5M validation feature rows | 15 min |
| `reports/all_feats_top5.parquet` | ~1.5 GB | All 7.6M validation pair features | 81 min |
| `reports/matcher.pkl` | <1 MB | Trained logistic regression, threshold=0.15 | <2 sec |
| `output/val_matching_results.tsv` | ~50 MB | Validation predictions (F0.5 = 0.9557) | - |

Without these files, rerunning approximately 15 hours of computation is required before improving anything.

## 4. Key results achieved

### Retrieval recall @ top-5

- **95.08%** - Inverted index finds the true S1 in five candidates for 95% of links.
- **86.0% perfect-recall entities** - 86% of S1 entities had all true links retrieved.
- **0.61% zero-recall entities** - Only 0.61% were completely missed by the retriever.

### Validation baseline

- **F0.5:** 0.9557
- **Best threshold:** 0.15
- The low threshold indicates that the classifier is high-confidence.

## 5. Check the overnight run first

When you sit down, check whether `predict_test.py` is still running:

```sh
tail -f logs/predict_test.log
# or
ps aux | grep predict_test
```

This script runs for approximately 12 hours. The test set contains 9.97M S2/S3 records. It produces both submission files. If it crashed, resume it; it checkpoints automatically:

```sh
nohup PYTHONPATH=src python scripts/predict_test.py > logs/predict_test.log 2>&1 &
```

## 6. Critical bug to verify in `predict_test.py`

The script must use `train_source1.tsv` as the retrieval index and feature source because the matcher was trained on train S1 features. However, output predictions must be keyed by test S1 entity IDs from `test_source1.tsv`.

If `output/matching_results.tsv` contains S1 IDs that do not exist in `test_source1.tsv`, this bug is present and needs fixing.

## 7. Full improvement plan to 0.99+

### Step 1 - First leaderboard submission

```sh
PYTHONPATH=src python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Then upload the files.

### Step 2 - Failure analysis (approximately 30 minutes)

```sh
PYTHONPATH=src python scripts/analyse_failures.py
```

This reveals singleton false-positive counts, false negatives, and the worst entities. Use it to decide where to focus.

### Step 3 - Upgrade to LightGBM (approximately 1 hour, estimated +0.02-0.04 F0.5)

```sh
pip install lightgbm
```

Edit `src/er945/match.py`:

```python
from lightgbm import LGBMClassifier

class PairMatcher:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self._clf = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )

    def fit(self, features, labels):
        self._clf.fit(features[FEATURE_COLS].values.astype("float32"), labels.values)

    def predict_proba(self, features):
        return self._clf.predict_proba(
            features[FEATURE_COLS].values.astype("float32")
        )[:, 1]

    def predict(self, features):
        return (self.predict_proba(features) >= self.threshold).astype(int)
```

Retrain using the cached features. This should take approximately two minutes:

```sh
rm reports/matcher.pkl
PYTHONPATH=src python scripts/train_matcher.py
```

### Step 4 - Embedding features (approximately 3-4 hours CPU, estimated +0.03-0.05 F0.5)

This is the biggest remaining lever.

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

- Apache 2.0 license.
- Covers English, Hindi, French, Arabic, Bengali, Chinese, German, Japanese, Korean, Portuguese, Russian, Spanish, Turkish, Urdu, and 40+ more languages.
- 384-dimensional vectors.
- CPU-compatible.

Install the dependency:

```sh
pip install sentence-transformers
```

Create `scripts/embed_names.py`:

```python
import sys
sys.path.insert(0, "src")

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from er945.data import load_source
from er945.normalize import normalize_name

TRAIN = Path("dataset/train")
REPORTS = Path("reports")

s1 = load_source(TRAIN / "train_source1.tsv")
s2 = load_source(TRAIN / "train_source2.tsv")
s3 = load_source(TRAIN / "train_source3.tsv")

all_names = set()
for df in [s1, s2, s3]:
    all_names.update(df["business_name"].dropna().map(normalize_name).unique())
all_names = sorted(all_names)
print(f"Unique names: {len(all_names):,}")

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(
    all_names,
    batch_size=512,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

np.save(REPORTS / "name_embeddings.npy", embeddings)
with open(REPORTS / "name_to_idx.pkl", "wb") as f:
    pickle.dump({name: i for i, name in enumerate(all_names)}, f)
print(f"Saved: {embeddings.shape}")
```

Then add `name_cosine` to `build_features()` in `features.py`:

```python
# Load at script top:
import numpy as np
import pickle

_embs = np.load("reports/name_embeddings.npy")
with open("reports/name_to_idx.pkl", "rb") as f:
    _idx = pickle.load(f)


def _cosine(a, b):
    ia, ib = _idx.get(a), _idx.get(b)
    if ia is None or ib is None:
        return 0.0
    return float(np.dot(_embs[ia], _embs[ib]))  # already unit-normed

# Inside the build_features() per-pair loop:
"name_cosine": _cosine(qn, sn),
```

Add `name_cosine` to `FEATURE_COLS` in `match.py`. Then rebuild the derived feature caches and matcher:

```sh
rm reports/train_feats_top5.parquet reports/val_feats_top5.parquet \
   reports/all_feats_top5.parquet reports/matcher.pkl
PYTHONPATH=src python scripts/train_matcher.py
```

### Step 5 - Singleton false-positive fix (estimated +0.01-0.02 F0.5)

After failure analysis, if the singleton false-positive count is high, raise the threshold for entities where all candidate scores are low (< 0.6). Add a per-entity confidence filter in `predict_test.py`:

```python
# After scoring, for each S1 entity:
# if max(scores for this entity) < 0.6: use threshold 0.5 instead of 0.15
```

### Step 6 - Final threshold sweep and submission

Rescore the test set. With features cached, this should take approximately five minutes:

```sh
PYTHONPATH=src python scripts/predict_test.py
```

Validate both files:

```sh
PYTHONPATH=src python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Then submit.

## 8. F0.5 trajectory

| Stage | Expected result |
|---|---:|
| Logistic baseline | 0.9557 |
| First submission | ~0.93-0.96 |
| LightGBM | ~0.97 |
| Embedding features | ~0.98-0.99 |
| Singleton fix | 0.99+ |

## 9. Setup from scratch

```sh
git clone https://github.com/DevSoni31/945.git
cd 945
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place `dataset/` and `reports/` from Dev, then verify the setup:

```sh
PYTHONPATH=src pytest tests/ -v
PYTHONPATH=src python scripts/train_matcher.py  # should print F0.5: 0.9557
```

## 10. The pipeline in one diagram

```text
S2/S3 query records
        |
        v
[RETRIEVE] Inverted IDF index on S1
        |  top_k=5 candidates per query
        |  recall ceiling = 95.08%
        v
7.6M candidate (S2/S3, S1) pairs
        |
        v
[SCORE] 16 features: fuzzy name/address similarity,
        | token jaccard, rank, country match,
        | + name embedding cosine (Step 4)
        | -> LightGBM probability 0-1
        v
[DECIDE] Accept if score >= threshold
        |  Build S1 -> {S2/S3} predictions
        v
matching_results.tsv + candidate_pairs.tsv
```

## 11. Do not

- Rerun `benchmark_retrieval.py`; it is already complete with recall=0.9508.
- Delete any `.parquet` or `.pkl` in `reports/` without understanding the recomputation cost.
- Change the `make_split()` random seed; this invalidates all cached features.
- Forget `candidate_pairs.tsv`; the submission is invalid without it.
- Use `test_source1.tsv` as the retrieval index; use `train_source1.tsv`.

---

**Last updated:** 25 Sep 2026, 23:45 IST
**Validation F0.5:** 0.9557
**Test pipeline:** running overnight
