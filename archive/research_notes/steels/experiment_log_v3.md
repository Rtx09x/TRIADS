# TRM-MatSci V3 — Experiment Log
## Magpie Features + MLP-TRM | matbench_steels | 2026-03-01

---

## 1. Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Dataset** | matbench_steels (312 samples, yield strength prediction) |
| **CV Strategy** | 5-fold (KFold, shuffle=True, random_state=18012019) — official MatBench splits |
| **Input** | 132 Magpie compositional descriptors (22 properties × 6 statistics) |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max=300, eta_min=1e-5) |
| **Epochs** | 300 per fold |
| **Batch size** | 32 |
| **Recursion steps** | 16 (shared weights) |
| **Dropout** | 0.2 |
| **Early stopping** | patience=60 |
| **Val split** | Stratified 15% (quartile bins) |
| **Hardware** | Kaggle P100 |
| **Total time** | 5.2 minutes |

### Why Magpie?
V1 and V2 proved the input representation was the bottleneck (MLP stuck at 184 MPa regardless of model size). Magpie descriptors encode element interaction information through precomputed statistics (mean, avg_dev, min, max, range, mode of elemental properties like electronegativity, atomic radius, etc.), bypassing the need for the model to discover these interactions from scratch.

---

## 2. Results Summary (Test MAE — 5-Fold Average)

| Config | Hidden | ff_dim | Params | Test MAE (MPa) | ±Std |
|--------|:------:|:------:|:------:|:--------------:|:----:|
| **MLP-Magpie-L** | 128 | 256 | 248,065 | **130.33** | 12.93 |
| MLP-Magpie-S | 64 | 128 | 66,689 | 138.40 | 18.25 |

### Per-Fold Breakdown

**MLP-Magpie-S (66,689 params):**

| Fold | Test MAE | Val MAE | Early Stop Epoch | Val-Test Gap |
|:----:|:--------:|:-------:|:----------------:|:------------:|
| 1 | 128.69 | 134.72 | 300 (no ES) | -6.0 |
| 2 | 128.79 | 108.68 | 217 | +20.1 |
| 3 | 132.85 | 110.65 | 225 | +22.2 |
| **4** | **174.69** | **119.43** | **241** | **+55.3** |
| 5 | 126.99 | 126.78 | 260 | +0.2 |

**MLP-Magpie-L (248,065 params):**

| Fold | Test MAE | Val MAE | Early Stop Epoch | Val-Test Gap |
|:----:|:--------:|:-------:|:----------------:|:------------:|
| 1 | 131.12 | 113.87 | 269 | +17.2 |
| 2 | 117.24 | 121.03 | 132 | -3.8 |
| 3 | 125.95 | 95.82 | 152 | +30.1 |
| **4** | **154.56** | **117.11** | **151** | **+37.5** |
| 5 | 122.77 | 118.16 | 237 | +4.6 |

---

## 3. Key Findings

### F1: Magpie Input Breaks the 184 MPa Ceiling ✅
| Version | Input | Best MAE | Change |
|---------|-------|:--------:|:------:|
| V1 | Mat2Vec weighted sum (200d) | 184.38 | baseline |
| V2 | Same for MLP / element tokens for Trans | 184.57 | ±0 |
| **V3** | **Magpie descriptors (132d)** | **130.33** | **−54 MPa** |

The input representation was confirmed as the bottleneck. Magpie features provide element interaction information that the weighted-sum destroyed.

### F2: MLP-L Outperforms MLP-S (Reversed from V1)
In V1, h64 beat h128 (184.4 vs 189.5) because narrow = better regularization with weak input. In V3 with richer Magpie input, h128 beats h64 (130.3 vs 138.4) — the model now has enough information to benefit from extra capacity.

### F3: Fold 4 Is Consistently the Hardest
Fold 4 is an outlier in both configs (174.7 for S, 154.6 for L). Without fold 4, averages would be ~129 (S) and ~124 (L). This fold's test split likely contains compositions with unusual element combinations or extreme yield strengths.

### F4: Val-Test Gap Still Present but Smaller
V2 MLP had ~77 MPa val-test gap. V3 gaps are smaller but still significant (avg ~20-30 MPa). The 37-sample validation set remains too small for reliable early stopping.

### F5: MLP-L Early Stops Much Earlier
MLP-L (248K params) early stops at epochs 132-269 (avg ~188), while MLP-S (67K params) goes to 217-300 (avg ~249). The larger model finds and overfits its optimum faster, but early stopping successfully captures the best checkpoint.

---

## 4. Progress Across All Versions

| Version | Best Model | MAE (MPa) | Params | Input |
|---------|-----------|:---------:|:------:|-------|
| V1 | MLP-TRM-100K-h64 | 184.38 | 99,841 | Mat2Vec weighted sum |
| V2 | MLP-L | 184.57 | ~115K | Same (MLP unchanged) |
| **V3** | **MLP-Magpie-L** | **130.33** | **248,065** | **Magpie 132 descriptors** |
| V3.1 | MLP-Magpie-XS | 160.43 | 31,153 | Magpie 132d (too small) |
| V4 | MLP-Combined-S | 131.63 | 66,889 | Magpie+Mat2Vec 332d |
| — | CrabNet (target) | 107.32 | — | Element attention (pretrained) |
| — | TPOT (leaderboard best) | 79.95 | — | AutoML trees |

**Current leaderboard position: #10** (between Darwin 123.29 and gptchem 143.00)

---

## 5. V3.1 — XS Model Results ✅

| Config | Params | Test MAE (MPa) | ±Std |
|--------|:------:|:--------------:|:----:|
| MLP-Magpie-XS (h=48, ff=72) | 31,153 | 160.43 | 22.97 |

**Verdict:** Too small — underfits. 48-dim bottleneck compresses 132-dim input too aggressively. Confirmed ~65K+ params is the minimum for Magpie input. Full data in `experiment_log_v4.md`.
