# TRM-MatSci V2 — Experiment Log
## 6-Model Element-Token Benchmark | matbench_steels | 2026-03-01

---

## 1. Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Dataset** | matbench_steels (312 samples, yield strength prediction) |
| **CV Strategy** | 5-fold (KFold, shuffle=True, random_state=18012019) — official MatBench splits |
| **Input (MLP)** | Mat2Vec 200-dim fraction-weighted sum (same as V1) |
| **Input (Transformer)** | Element-as-token: 205 dims per element (Mat2Vec + radius + EN + group + period + fraction) |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max=300, eta_min=1e-5) |
| **Epochs** | 300 per fold |
| **Batch size** | 32 |
| **Recursion steps** | 16 (shared weights) |
| **Dropout** | 0.2 |
| **Early stopping** | patience=60 |
| **Val split** | Stratified 15% (quartile bins) |
| **Hardware** | Kaggle P100 |

### Architecture Variants
- **MLP-S/L**: Same MLP-TRM from V1, fraction-weighted pooled input (200-dim)
- **Trans-Normal-S/L**: Pool element tokens → attention over synthetic [x_proj, y, z] triplet
- **Trans-Novel-S/L**: Dual-reference architecture — E0 (fixed) + Et (evolving) element tokens, z cross-attends to both

---

## 2. Results Summary (Test MAE — 5-Fold Average)

| Config | Hidden | ff_dim | Params | Test MAE (MPa) | ±Std |
|--------|:------:|:------:|:------:|:--------------:|:----:|
| **MLP-S** | 64 | 128 | ~30K | **186.60** | 18.46 |
| **MLP-L** | 128 | 256 | ~115K | **184.57** | 11.21 |
| Trans-Normal-S | 256 | 256 | ~710K | 389.44 | 22.84 |
| Trans-Novel-S | 256 | 256 | ~800K | 388.58 | 23.22 |
| Trans-Normal-L | 256 | 512 | ~970K | 391.30 | — |
| Trans-Novel-L | 256 | 512 | ~1.1M | 390.30 | — |

**Best V2 model: MLP-L at 184.57 ± 11.21 MPa** — unchanged from V1 (184.38 MPa).

### Key Comparison

| Model | MAE | Verdict |
|-------|:---:|---------|
| V1 best (MLP-100K-h64) | 184.38 | V1 baseline |
| **V2 best (MLP-L)** | **184.57** | **No improvement** — MLP still uses weighted-sum input |
| Dummy | 229.74 | — |
| All V2 Transformers | 388–391 | **1.7× worse than dummy** |
| CrabNet | 107.32 | Leaderboard |

---

## 3. Key Findings

### F1: Element-Token Input Did NOT Help MLP
MLP models use `featurize_pooled()` (fraction-weighted sum), not the new element tokens. Both V1 and V2 MLP results are identical — the "food fix" was never applied to MLP.

### F2: Element-Token Input Did NOT Help Transformer
Despite feeding real element tokens to the Transformer, results are **worse than dummy** (389 vs 229 MPa). The attention mechanism cannot learn element interactions from 312 samples.

### F3: Trans-Novel ≈ Trans-Normal (No Benefit)
The dual-reference architecture (E0 fixed + Et evolving) provided zero improvement:
- Trans-Normal-S: 389.44 vs Trans-Novel-S: 388.58 (within noise)
- Trans-Normal-L: 391.30 vs Trans-Novel-L: 390.30 (within noise)

### F4: Transformer Recursion Is Non-Functional
Panel 4 (recursion convergence) shows MLP lines descending smoothly from ~1400→185 across 16 steps. **Transformer lines are flat** at ~390. The recursive loop does nothing for the Transformer — it computes one pass and returns it 16 times.

### F5: Transformer Training Curves Show S-Curve Convergence
Training curves are sigmoidal: plateau (0–50 epochs) → steep drop (50–200) → flattening (200–300). The Transformer IS finding a loss basin, but the basin's floor (~390 MPa test) is worse than predicting the mean. This is a structured local minimum, not random memorization.

### F6: Val-Test Gap Is Extreme for MLP
MLP-L best val: ~107 MPa, test: ~184 MPa — gap of ~77 MPa. The small stratified val set (37 samples) is being overfitted by early stopping.

---

## 4. Conclusions → V3 Direction

The core problem identified: **312 samples is too few for attention to discover element interactions from scratch.** Leaderboard models that beat 100 MPa either use:
1. Tree/ensemble methods (TPOT, AutoML) that handle small N natively
2. Heavy feature engineering (RF-SCM/Magpie: 132 hand-crafted descriptors)
3. Pretraining on 300K+ materials (CrabNet)

**V3 strategy: Use Magpie descriptors (132 features capturing compositional statistics) as input to MLP-TRM.** This encodes element interaction information through engineered statistics rather than asking attention to discover it from scratch.
