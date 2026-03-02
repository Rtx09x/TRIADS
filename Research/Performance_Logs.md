# TRIADS Performance Logs

*Complete experimental results across all 15 major versions. All results use strict MatBench nested 5-fold cross-validation on `matbench_steels` (312 samples).*

> **See also:** [Architecture Evolution](./Architecture_Evolution.md) · [Hyperparameter Studies](./Hyperparameter_Studies.md) · [README](../README.md)
>
> **Training JSON data:** [V1](../Training%20Json/summary.json) · [V2](../Training%20Json/summary_v2.json) · [V3](../Training%20Json/summary_v3.json) · [V4](../Training%20Json/summary_v4.json) · [V5](../Training%20Json/summary_v5.json) · [V6](../Training%20Json/summary_v6.json) · [V7](../Training%20Json/summary_v7.json) · [V8](../Training%20Json/summary_v8.json) · [V9](../Training%20Json/summary_v9.json) · [V10](../Training%20Json/summary_v10.json) · [V11](../Training%20Json/summary_v11.json) · [V12](../Training%20Json/summary_v12.json) · [V13](../Training%20Json/summary_v13.json) · [V14](../Training%20Json/summary_v14.json) · [V15](../Training%20Json/summary_v15.json)

---

## Master Leaderboard

### TRIADS Models (Ranked by MAE)

| Rank | Version | Configuration | Parameters | MAE (MPa) | ±Fold Std | Key Innovation |
|:----:|:--------|:-------------|:----------:|:---------:|:---------:|:---------------|
| 1 | **V13A** | **2×SA + StdDS, 5-seed ensemble** | **224,685** | **91.20** | **12.23** | **Multi-seed ensemble + 2-layer SA** |
| 2 | V13B | 2×SA + ConfDS, 5-seed ensemble | 229,390 | 93.04 | 13.01 | Confidence-weighted DS ensemble |
| 3 | V14A | Mega-Features + 2×SA (single seed) | 238,509 | 94.94 | 14.21 | Single-seed SOTA, 670d features |
| 4 | V12A | Scaled + Expanded + StdDS | 191,213 | 95.99 | 10.56 | First sub-100 MPa model |
| 5 | V14B | Tokenized Mega-Features | 195,917 | 96.15 | 12.14 | 58-token 2-pass TRM attention |
| 6 | V12B | Scaled + Expanded + ConfDS | 195,918 | 97.59 | 16.21 | Single fold hit 74.55 MPa |
| 7 | V11B | Scaled + Deep Supervision | 172,013 | 102.30 | 8.61 | DS unlocks d_attn=64, lowest ±std |
| 8 | V10A | Fixed-20 + Deep Supervision | 87,353 | 103.28 | 10.49 | Core breakthrough—beat RF-SCM |
| 9 | V10B | Adaptive Halting + DS | 87,353 | 104.87 | 10.22 | 25% compute savings, −1.6 MPa |
| 10 | V10.1 | V10A multi-seed (grand mean) | 87,353 | 105.85 | 3.00* | Reproducibility validated |
| 11 | V11A | Feature Expansion (d_attn=48) | 100,153 | 107.98 | 11.06 | Extra features need more capacity |
| 12 | V7B | Hybrid-L (d_attn=48) | 87,353 | 127.08 | 18.72 | First attention > MLP result |
| 13 | V7D | Cross-Architecture ensemble | 166,842 | 128.22 | 14.63 | MLP + Hybrid combined |
| 14 | V5A | MLP-SWA | 66,889 | 128.98 | 17.42 | SWA introduced |
| 15 | V6C | MLP-SWA ×3 seeds | 66,889 ×3 | 129.04 | 8.99 | Halved variance, same mean |
| 16 | V3 | MLP-Magpie-L | 248,065 | 130.33 | 12.93 | Magpie breaks 184 MPa ceiling |
| 17 | V7A | MLP-SWA (ff=128) | 79,489 | 131.05 | 16.77 | Larger MLP overfits |
| 18 | V4 | MLP-Combined-S | 66,889 | 131.63 | 14.83 | Combined features, 4× fewer params |
| 19 | V4 | MLP-Combined-L | 117,281 | 132.76 | 21.45 | Larger model overfits |
| 20 | V11C | ACT Learned Halting | 89,466 | 132.59 | 13.33 | Ponder cost killed learning |
| 21 | V7C | Hybrid-TRM ×3 seeds | 87,353 ×3 | 134.06 | 13.30 | Same-seed ensemble fails |
| 22 | V6B | Hybrid-TRM (d_attn=32) | 67,305 | 134.97 | 23.15 | Val 84.1 MPa (MODNet level) |
| 23 | V9A | Hybrid-20 (no DS) | 87,353 | 134.59 | 10.43 | Over-refinement paradox |
| 24 | V3 | MLP-Magpie-S | 66,689 | 138.40 | 18.25 | Smaller Magpie model |
| 25 | V9B | Hybrid-20L (no DS) | 100,853 | 140.14 | 19.29 | Over-refinement + overfitting |
| 26 | V8A | Hybrid-M2V | 77,849 | 143.03 | 23.20 | Removing CA layer hurts |
| 27 | V6A | FeatGroup-L (d=48) | 80,929 | 153.96 | 8.84 | Most stable model ever (±8.84) |
| 28 | V8B | Hybrid-XL (d_attn=64, no DS) | 113,545 | 155.06 | 19.47 | Attention overfitting |
| 29 | V3.1 | MLP-Magpie-XS | 31,153 | 160.43 | 22.97 | Too small—underfits |
| 30 | V5B | FeatGroup (d=32) | 38,593 | 165.11 | 17.56 | Property tokens work |
| 31 | V1 | MLP-TRM-100K-h64 | 99,841 | 184.38 | 8.2 | V1 best |
| 32 | V2 | MLP-L | ~115,000 | 184.57 | 11.21 | MLP unchanged from V1 |
| 33 | V2 | Trans-Novel-S (element tokens) | ~800,000 | 388.58 | 23.22 | Attention fails on 312 samples |
| 34 | V15 | HTRM (Hierarchical TRM) | ~220,000 | 431.86 | 49.59 | Catastrophic failure |
| 35 | V1 | Trans-TRM-h128 | 167,713 | 583.55 | 24.48 | Attention barely functional |
| 36 | V1 | Trans-TRM-100K-h64 | 100,641 | 994.40 | 22.20 | Degenerate dims/head |

*\* V10.1 ±std is cross-seed standard deviation, not cross-fold.*

### External Baselines (MatBench Leaderboard)

| Model | MAE (MPa) | Type |
|:------|:---------:|:-----|
| AutoGluon | 77.03 | Stacked ensemble (AutoML) |
| TPOT-Mat | 79.95 | AutoML pipeline |
| MODNet v0.1.12 | 87.76 | Neural network |
| RF-Regex Steels | 90.58 | Random Forest + regex features |
| **TRIADS V13A (Ours)** | **91.20** | **Hybrid-TRM + Deep Supervision** |
| RF-SCM/Magpie | 103.51 | Random Forest + Magpie |
| CrabNet | 107.31 | Transformer (pretrained) |
| Darwin | 123.29 | Evolutionary algorithm |
| Dummy (mean prediction) | 229.74 | Baseline |

---

## Per-Version Detailed Results

### V1: Baseline Sweep ([code](../Training%20Code/trm.py) · [📊 results](../Images/trm_results_v2.png) · [📄 JSON](../Training%20Json/summary.json)) — 12 Models

**Date:** March 1, 2026 | **Hardware:** Kaggle P100 | **Runtime:** 119.5 minutes

| Config | Target Params | Actual Params | Hidden | MAE (MPa) | ±Std |
|:-------|:------------:|:-------------:|:------:|:---------:|:----:|
| MLP-TRM-10K-h64 | 10K | 20,641 | 64 | 191.6 | 11.9 |
| MLP-TRM-10K-h128 | 10K | 41,249 | 128 | 190.2 | 14.5 |
| MLP-TRM-50K-h64 | 50K | 49,441 | 64 | 188.2 | 10.5 |
| MLP-TRM-50K-h128 | 50K | 48,433 | 128 | 188.9 | 8.5 |
| **MLP-TRM-100K-h64** | **100K** | **99,841** | **64** | **184.4** | **8.2** |
| MLP-TRM-100K-h128 | 100K | 98,721 | 128 | 189.5 | 14.1 |
| Trans-TRM-10K-h64 | 10K | 51,105 | 64 | 992.2 | 23.9 |
| Trans-TRM-10K-h128 | 10K | 167,713 | 128 | 583.6 | 24.5 |
| Trans-TRM-50K-h64† | 50K | 51,105 | 64 | 992.2 | 23.9 |
| Trans-TRM-50K-h128† | 50K | 167,713 | 128 | 583.6 | 24.5 |
| Trans-TRM-100K-h64 | 100K | 100,641 | 64 | 994.4 | 22.2 |
| Trans-TRM-100K-h128† | 100K | 167,713 | 128 | 583.6 | 24.5 |

*† Identical model to previous row due to parameter floor exceeding target.*

---

### V2: Element-as-Token ([code](../Training%20Code/trm2.py) · [📊 results](../Images/trm_results_v2%20(1).png) · [📄 JSON](../Training%20Json/summary_v2.json)) — 6 Models

| Config | Hidden | ff_dim | Params | MAE (MPa) | ±Std |
|:-------|:------:|:------:|:------:|:---------:|:----:|
| MLP-S | 64 | 128 | ~30K | 186.60 | 18.46 |
| MLP-L | 128 | 256 | ~115K | 184.57 | 11.21 |
| Trans-Normal-S | 256 | 256 | ~710K | 389.44 | 22.84 |
| Trans-Novel-S | 256 | 256 | ~800K | 388.58 | 23.22 |
| Trans-Normal-L | 256 | 512 | ~970K | 391.30 | — |
| Trans-Novel-L | 256 | 512 | ~1.1M | 390.30 | — |

---

### V3–V4: Feature Engineering ([V3 code](../Training%20Code/trm3.py) · [V4 code](../Training%20Code/trm4.py) · [📊 V3](../Images/trm_results_v3.png) · [📊 V4](../Images/trm_results_v4.png) · [📄 V3 JSON](../Training%20Json/summary_v3.json) · [📄 V4 JSON](../Training%20Json/summary_v4.json))

| Config | Params | Input | MAE (MPa) | ±Std |
|:-------|:------:|:------|:---------:|:----:|
| MLP-Magpie-L | 248,065 | Magpie 132d | 130.33 | 12.93 |
| MLP-Magpie-S | 66,689 | Magpie 132d | 138.40 | 18.25 |
| MLP-Magpie-XS | 31,153 | Magpie 132d | 160.43 | 22.97 |
| MLP-Combined-S | 66,889 | Magpie+Mat2Vec 332d | 131.63 | 14.83 |
| MLP-Combined-L | 117,281 | Magpie+Mat2Vec 332d | 132.76 | 21.45 |

---

### V5–V7: SWA, Hybrid Architecture, and Ensembles ([V5 code](../Training%20Code/trm5.py) · [V6 code](../Training%20Code/trm6.py) · [V7 code](../Training%20Code/trm7.py) · [📊 V5](../Images/trm_results_v5.png) · [📊 V7](../Images/trm_results_v7.png))

| Config | Params | MAE (MPa) | ±Std | Notes |
|:-------|:------:|:---------:|:----:|:------|
| V5A MLP-SWA | 66,889 | 128.98 | 17.42 | SWA introduced |
| V5B FeatGroup | 38,593 | 165.11 | 17.56 | Property tokens, dual-reference |
| V6A FeatGroup-L | 80,929 | 153.96 | 8.84 | Lowest variance ever |
| V6B Hybrid-TRM | 67,305 | 134.97 | 23.15 | Val reached 84.1 MPa |
| V6C MLP-Ens ×3 | 66,889 ×3 | 129.04 | 8.99 | Variance halved |
| **V7B Hybrid-L** | **87,353** | **127.08** | **18.72** | **First attention > MLP** |
| V7A MLP-SWA | 79,489 | 131.05 | 16.77 | Larger ff overfits |
| V7C Hybrid-Ens ×3 | 87,353 ×3 | 134.06 | 13.30 | Same-seed fails |
| V7D Cross-Arch | 166,842 | 128.22 | 14.63 | MLP drags Hybrid |

---

### V8–V9: Scaling Wall and Over-Refinement ([V8 code](../Training%20Code/trm8.py) · [V9 code](../Training%20Code/trm9.py) · [📊 V8](../Images/trm_results_v8.png) · [📊 V9](../Images/trm_results_v9.png))

| Config | Params | MAE (MPa) | ±Std | Notes |
|:-------|:------:|:---------:|:----:|:------|
| V8A Hybrid-M2V | 77,849 | 143.03 | 23.20 | CA layer is load-bearing |
| V8B Hybrid-XL | 113,545 | 155.06 | 19.47 | d_attn=64 overfits without DS |
| V9A Hybrid-20 | 87,353 | 134.59 | 10.43 | Over-refinement paradox |
| V9B Hybrid-20L | 100,853 | 140.14 | 19.29 | Larger + deeper both hurt |

---

### V10: Deep Supervision Breakthrough ([code](../Training%20Code/trm10.py) · [📊 results](../Images/trm_results_v10.png) · [📄 JSON](../Training%20Json/summary_v10.json))

| Config | Params | MAE (MPa) | ±Std | Notes |
|:-------|:------:|:---------:|:----:|:------|
| **V10A Fixed-20 DS** | **87,353** | **103.28** | **10.49** | **Beat Darwin, CrabNet, RF-SCM** |
| V10B Adaptive Halt | 87,353 | 104.87 | 10.22 | Saves 25% compute |

**V10.1 Reproducibility (3 seeds):**

| Seed | MAE (MPa) | ±Fold Std |
|:----:|:---------:|:---------:|
| 42 (original) | 103.28 | 10.50 |
| 123 | 104.21 | 8.07 |
| 7 | 110.07 | 13.46 |
| **Grand Mean** | **105.85** | **(Seed Std: 3.00)** |

---

### V11: Pushing for #1 ([code](../Training%20Code/trm11.py) · [📊 results](../Images/trm_results_v11.png) · [📄 JSON](../Training%20Json/summary_v11.json))

| Config | Params | MAE (MPa) | ±Std | Notes |
|:-------|:------:|:---------:|:----:|:------|
| **V11B Scaled+DS** | **172,013** | **102.30** | **8.61** | **DS unlocks d_attn=64** |
| V11A FeatExp | 100,153 | 107.98 | 11.06 | Extra features need more capacity |
| V11C ACT | 89,466 | 132.59 | 13.33 | Ponder cost dominates |

**V11B Per-Fold:**

| Fold | MAE (MPa) |
|:----:|:---------:|
| 1 | 118.82 |
| 2 | 101.79 |
| 3 | 95.60 |
| 4 | 99.82 |
| 5 | 95.48 |

---

### V12: Breaking 100 MPa ([code](../Training%20Code/trm12.py) · [📊 results](../Images/trm_results_v12.png) · [📄 JSON](../Training%20Json/summary_v12.json))

| Config | Params | MAE (MPa) | ±Std | Notes |
|:-------|:------:|:---------:|:----:|:------|
| **V12A StdDS** | **191,213** | **95.99** | **10.56** | **First sub-100 model** |
| V12B ConfDS | 195,918 | 97.59 | 16.21 | Peak fold: 74.55 MPa |

**V12A Per-Fold:**

| Fold | MAE (MPa) |
|:----:|:---------:|
| 1 | 114.71 |
| 2 | 82.75 |
| 3 | 97.48 |
| 4 | 94.07 |
| 5 | 90.95 |

---

### V13: Final Architecture — TRIADS SOTA ([code](../Training%20Code/trm13.py) · [📊 results](../Images/trm_results_v13.png) · [📄 JSON](../Training%20Json/summary_v13.json))

| Config | Params | MAE (MPa) | ±Std | Notes |
|:-------|:------:|:---------:|:----:|:------|
| **V13A StdDS Ensemble** | **224,685** | **91.20** | **12.23** | **Project SOTA** |
| V13B ConfDS Ensemble | 229,390 | 93.04 | 13.01 | Standard DS wins again |

**V13A Per-Fold (5-Seed Ensemble):**

| Fold | MAE (MPa) |
|:----:|:---------:|
| 1 | 114.32 |
| 2 | 81.46 |
| 3 | 80.55 |
| 4 | 90.49 |
| 5 | 89.18 |

---

### V14: Mega-Features ([code](../Training%20Code/trm14.py) · [📊 results](../Images/trm_results_v14.png) · [📄 JSON](../Training%20Json/summary_v14.json))

| Config | Params | MAE (MPa) | ±Std | Notes |
|:-------|:------:|:---------:|:----:|:------|
| **V14A Flat** | **238,509** | **94.94** | **14.21** | **Single-seed SOTA** |
| V14B Tokenized | 195,917 | 96.15 | 12.14 | 58 tokens too sparse |

---

### V15: HTRM — Failure ([code](../Training%20Code/trm15.py) · [📊 results](../Images/trm_results_v15.png) · [📄 JSON](../Training%20Json/summary_v15.json))

| Config | Params | MAE (MPa) | ±Std | Notes |
|:-------|:------:|:---------:|:----:|:------|
| V15 HTRM | ~220K | 431.86 | 49.59 | Catastrophic collapse |

---

## Error Reduction Timeline

| Milestone | Version | MAE (MPa) | Reduction from V1 |
|:----------|:--------|:---------:|:------------------:|
| Initial baseline | [V1](../Training%20Code/trm.py) | 184.38 | — |
| Feature engineering (Magpie) | [V3](../Training%20Code/trm3.py) | 130.33 | −54.05 (−29.3%) |
| SWA generalization | [V5A](../Training%20Code/trm5.py) | 128.98 | −55.40 (−30.0%) |
| Hybrid attention architecture | [V7B](../Training%20Code/trm7.py) | 127.08 | −57.30 (−31.1%) |
| Deep Supervision breakthrough | [V10A](../Training%20Code/trm10.py) | 103.28 | −81.10 (−44.0%) |
| Architectural scaling + DS | [V11B](../Training%20Code/trm11.py) | 102.30 | −82.08 (−44.5%) |
| Feature expansion + scaling | [V12A](../Training%20Code/trm12.py) | 95.99 | −88.39 (−47.9%) |
| **5-seed ensemble (final)** | **[V13A](../Training%20Code/trm13.py)** | **91.20** | **−93.18 (−50.5%)** |

**Total improvement: 184.38 → 91.20 MPa (50.5% error reduction over 15 versions)**
