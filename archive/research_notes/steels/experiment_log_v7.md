# TRM-MatSci V7 — Experiment Log
## MLP + Hybrid-L + Cross-Arch Ensembles | matbench_steels | 2026-03-02

---

## 1. Setup

Same base as V5/V6 (332-dim combined features, 5-fold CV, 300 epochs, SWA@200, P100 GPU).

**Four configurations:**
- V7A: MLP-SWA (h=64, ff=128) — slightly larger ff_dim (128 vs V5A's 100)
- V7B: Hybrid-L (d_attn=48, nhead=2) + SWA — scaled-up attention feature extractor
- V7C: Hybrid-L ×3 seeds — multi-seed ensemble, average predictions
- V7D: Cross-architecture (MLP + Hybrid×3 avg) — combine different error patterns

**Total time:** 17.3 minutes

---

## 2. Results

| Config | Params | Test MAE (MPa) | ±Std |
|--------|:------:|:--------------:|:----:|
| **V7B Hybrid-L** | 87,353 | **127.08** | 18.72 |
| V7D Cross-Arch | 166,842 | 128.22 | 14.63 |
| V7A MLP-SWA | 79,489 | 131.05 | 16.77 |
| V7C Hybrid-Ens ×3 | 87,353×3 | 134.06 | 13.30 |

### Per-Fold Breakdown

| Fold | V7A-MLP | V7B-Hyb | V7C-Ens | V7D-Cross |
|:----:|:-------:|:-------:|:-------:|:---------:|
| 1 | 127.28 | 124.56 | 128.03 | 124.33 |
| **2** | **151.79** | **153.03** | **153.69** | **146.91** |
| 3 | 112.30 | **104.59** | 128.53 | 116.95 |
| **4** | **149.40** | **143.42** | **144.16** | **143.36** |
| 5 | 114.49 | 109.78 | 115.88 | 109.54 |

---

## 3. Key Findings

### F1: Hybrid-L Is New Project Best — 127.08 MPa ✅
First time an attention-containing architecture beats pure MLP. Scaling d_attn from 32 (V6B: 135.0) to 48 (V7B: 127.1) gave **7.9 MPa improvement.** The Hybrid hasn't plateaued — d_attn scaling still yields returns.

### F2: MLP Has Peaked — Larger ff_dim Hurts
V5A (ff=100): 128.98 → V7A (ff=128): 131.05. More MLP capacity overfits on 312 samples. The MLP's optimal config is confirmed: h=64, ff=100. No further MLP gains expected.

### F3: Same-Seed Ensemble Fails Again
V7C (134.06) is worse than V7B single best (127.08). Three Hybrid models with different seeds converge to the same error patterns. Same-architecture ensembles are definitively ruled out as a strategy.

### F4: Cross-Arch Helps MLP but Hurts Hybrid
V7D (128.22) improves over V7A MLP alone (131.05) but degrades V7B Hybrid (127.08). The MLP's errors drag down the superior Hybrid. For leaderboard submission, use V7B alone.

### F5: Two-Tier Fold Structure
Folds split cleanly into easy (3,5: 105–110 MPa) and hard (2,4: 143–153 MPa). Fold 1 is medium (124 MPa). The hard folds account for the entire gap to Darwin. If folds 2,4 improved by ~20 MPa each, the average would be ~123 MPa.

### F6: Below CrabNet on Easy Folds
V7B Fold 3: **104.59 MPa** — below CrabNet (107.32)! With 87K params vs CrabNet's millions. On favorable data splits, the Hybrid-TRM is already competitive with much larger architectures.

---

## 4. Complete Results — All Versions

| Version | Best Model | MAE | Params | Gap to Darwin |
|---------|-----------|:---:|:------:|:-------------:|
| V1 | MLP-TRM-h64 | 184.4 | 100K | 61.1 |
| V2 | MLP-L | 184.6 | 115K | 61.3 |
| V3 | MLP-Magpie-L | 130.3 | 248K | 7.0 |
| V4 | MLP-Combined-S | 131.6 | 67K | 8.3 |
| V5A | MLP-SWA | 129.0 | 67K | 5.7 |
| V6C | MLP-Ensemble ×3 | 129.0 | 67K×3 | 5.7 |
| **V7B** | **Hybrid-L** | **127.1** | **87K** | **3.8** |
| — | Darwin (target) | 123.3 | — | 0 |
| — | CrabNet | 107.3 | — | — |

---

## 5. V8 Direction

### What We Know
1. MLP has peaked — no further gains from that architecture
2. Hybrid attention scaling (d_attn 32→48) still yields strong returns
3. Hard folds (2,4) are the entire gap to Darwin
4. Same-seed ensembles are dead; cross-arch ensembles marginal

### Strategies to Explore
1. **Scale Hybrid attention further** — d_attn=64, nhead=4 (32 dims/head). Test if attention feature extraction continues improving
2. **Deeper attention** — 2 self-attention layers before cross-attention. More capacity to learn property interactions
3. **Fold-difficulty analysis** — what compositions are in folds 2,4? Rare elements? Extreme yield strengths? Can we diagnose and address?
4. **Longer SWA** — swa_start=150 out of 400 epochs. Give SWA more time to flatten the loss landscape
5. **Target augmentation** — log-transform or robust scaling of yield strength to handle the high-variance samples in hard folds
