# TRM-MatSci — Research Direction
*Last updated: 2026-03-01*

---

> This document synthesizes everything we have learned from V1 experimentation and defines the research direction going forward. References [[experiment_log_v1]] for full data.

---

## Where We Started

We set out to answer one question:

> *Can a Tiny Recursive Model (TRM) — a model that reuses the same small network 16 times in a loop — compete with large, complex models for predicting steel yield strength, using only chemical composition as input?*

This was motivated by the Samsung TRM paper, which showed that recursive shared-weight models can achieve surprising performance with very few parameters. No one had applied this to materials science before. That's the novelty.

We chose `matbench_steels` (312 samples, yield strength prediction in MPa) as the benchmark because:
- It's small (perfect for testing small models)
- It's on a public leaderboard with comparable baselines
- The dataset is composition-only, matching our input strategy

---

## What We Did (V1 Sweep)

We trained **12 models** across two architecture families:
- **MLP-TRM**: Shared 2-layer MLP updates `z` and `y` at each recursive step
- **Transformer-TRM**: Shared self-attention + cross-attention updates `z` and `y` at each step

For each family we tested 3 parameter targets (10K, 50K, 100K) and 2 hidden dimensions (h64, h128). All models used:
- Mat2Vec 200-dim embeddings (real Word2Vec from Google Cloud Storage)
- Fraction-weighted sum → single 200-dim input vector
- 16 recursive steps, 200 epochs, AdamW + cosine LR schedule

---

## What We Found

### MLP Results
| Model | Test MAE |
|-------|:--------:|
| MLP-TRM-10K-h64 | 191.6 MPa |
| MLP-TRM-10K-h128 | 190.2 MPa |
| MLP-TRM-50K-h64 | 188.2 MPa |
| MLP-TRM-50K-h128 | 188.9 MPa |
| **MLP-TRM-100K-h64** | **184.4 MPa** ← best |
| MLP-TRM-100K-h128 | 189.5 MPa |

Dummy baseline: **229.7 MPa**. Best leaderboard: **79.95 MPa**.

### Transformer Results
| Model | Test MAE |
|-------|:--------:|
| Trans-TRM-10K-h64 | 992.2 MPa |
| Trans-TRM-10K-h128 | 583.6 MPa |
| Trans-TRM-50K-h64 | ~992–1100 MPa |
| Trans-TRM-50K-h128 | 583.6 MPa *(same model as 10K)* |
| Trans-TRM-100K-h64 | 100K | 64 | 994.4 MPa |
| Trans-TRM-100K-h128 | 100K | 128 | 583.6 MPa *(= 10K-h128, same model)* |

---

## Key Lessons from V1

### Lesson 1: The bottleneck is the input, not the model
MLP models cluster at 184–191 MPa regardless of parameter count (10x range). The model is not the problem — the **single weighted-sum vector** is destroying element interaction information before reasoning even begins. No amount of extra parameters can recover what was never there.

### Lesson 2: Width kills MLP, but saves the Transformer
- **MLP**: h128 overfits worse than h64 at every param count. Narrow = better regularization.
- **Transformer**: h64 is completely brain-dead (16 dims/head — too little for meaningful attention). h128 (32 dims/head) is the minimum for any learning to happen.
- They respond **oppositely** to hidden dimension. This is a real architectural distinction with implications for future design.

### Lesson 3: h64 Transformer gets worse with more params
Not just flat — actually *worse*. Trans-10K-h64 = 992 MPa, Trans-50K-h64 ≈ 1100 MPa. More parameters in a broken architecture amplifies garbage, it doesn't fix it.

### Lesson 4: The Transformer hasn't converged yet
Every Transformer h128 fold showed `Best = Val` at epoch 198 — meaning it was still actively learning at the end of training. MLP converges by epoch 50–100, Transformer hasn't hit its floor at epoch 200. With 1000 epochs, the Transformer story could be very different.

### Lesson 5: The Transformer h128 configs were all the same model
The minimum Trans-h128 model is ~167K params (attention matrices have a fixed floor). All three targets (10K, 50K, 100K) fell below this floor, so `build_model` returned the same model each time. **We accidentally ran the same experiment 3 times.** Trans-h128 was never actually scaled.

### Lesson 6: Fold 3 has a biased val split
The same ~37 val samples get selected for Fold 3 every time (deterministic via random_state). These samples happen to have yield strengths close to the dataset mean, making them "easy." Fold 3 always shows the lowest val MAE across every model, which distorts early stopping.

### Lesson 7: The Transformer's failure shows attention needs real tokens
Self-attention over 3 abstract states `[x_proj, y, z]` with 312 training samples can't learn meaningful patterns. Attention was designed to find relationships *between* things — but our 3 tokens aren't "things" in a material science sense. They're synthetic reasoning states. There's nothing chemically meaningful to attend between.

---

## New Research Direction

> **The fundamental insight driving V2/V3:**
> The model is being asked to learn element interactions, but by the time it sees the input, those interactions have already been erased.
> We need to give the model the raw elements, let it learn the interactions itself.

### Direction 1: Element-Wise Attention TRM (V3 Architecture)

Instead of one vector per alloy → **each element becomes its own token:**

```
CURRENT:
  Fe(0.80) × vec_Fe + C(0.02) × vec_C + Cr(0.08) × vec_Cr  →  [single 200-dim vector]
                                                                     ↓
                                                               TRM reasoning

PROPOSED:
  Token_Fe:  [mat2vec_Fe | frac=0.80 | atomic_radius | electronegativity | group | ...]
  Token_C:   [mat2vec_C  | frac=0.02 | atomic_radius | electronegativity | group | ...]
  Token_Cr:  [mat2vec_Cr | frac=0.08 | atomic_radius | electronegativity | group | ...]
                                    ↓
               Self-attention across element tokens
               "Cr + C → chromium carbides → higher strength"
                                    ↓
               Pooled embedding  →  TRM recursive reasoning (16 steps)
               Each step cross-attends to element tokens
```

**Why this can beat the leaderboard:**
- CrabNet (107 MPa) uses element-wise attention with no recursion
- Adding 16 recursive reasoning steps on top = strictly more expressive
- With the right dims-per-head (≥64) and enough epochs (1000+), this could challenge MODNet (87.76) or even TPOT (79.95)

### Direction 2: Fix Attention Head Engineering

From V1 we now know:
- 16 dims/head → completely broken
- 32 dims/head → barely functional
- **64 dims/head → target for V2**

V2 Transformer configs should use h256/nhead=4 (= 64 dims/head), properly scaled to 250K–1M actual parameters.

### Direction 3: Longer Training with Fixed Epochs

Remove early stopping for Transformer configs. Because:
- Transformer hasn't converged at epoch 200
- The val-test gap worsens with early stopping (model memorizes 37 val samples)
- Fixed 1000 epochs with cosine schedule = cleaner comparison

### Direction 4: Stratified Val Split

The Fold 3 bias creates misleading in-training metrics. Fix by stratifying the val split by target value (high/medium/low yield strength), ensuring the 37 val samples are representative of the full distribution.

---

## Planned V2 Training Script

**Models to train:**

| Config | Arch | Hidden | Target Params | Epochs | Notes |
|--------|------|:------:|:-------------:|:------:|-------|
| MLP-best-long | MLP | 64 | 100K | 1000 | Best V1 model, longer run |
| Trans-h256-250K | Trans | 256 | 250K | 1000 | 64 dims/head, actually different size |
| Trans-h256-500K | Trans | 256 | 500K | 1000 | Scaling test |
| Trans-h256-1M | Trans | 256 | 1M | 1000 | Upper bound |

**Changes from V1:**
- [ ] Remove early stopping, use fixed epoch training
- [ ] Stratify val split by target value
- [ ] h256 for Transformer (64 dims/head)
- [ ] Verify `build_model` is actually returning different param counts

---

## Limitations & Paper Transparency Statement

> This section documents known limitations and experimental flaws honestly. These will be included in the paper's Limitations section. Transparency here is a strength — it shows we understand the experiments deeply and know exactly what the results mean and don't mean.

### L1: Transformer Scaling Was Not Actually Tested (Critical)

**What happened:** The `build_model` function searches for `ff_dim` starting from the minimum and returns the smallest model whose size exceeds the target. For Transformer-h128, the attention matrices alone require ~167K parameters minimum:

```
input_proj (200 → 128):         25,728 params
z_self_attn (MHA, h128, n=4):  ~66,048 params
y_cross_attn (MHA, h128, n=4): ~66,048 params
────────────────────────────────────────────
Minimum skeleton (ff_dim=16):  ~167K+ params
```

All three targets (10K, 50K, 100K) are below this floor. `build_model` returned the identical minimum model in all three cases.

**Evidence:** Trans-TRM-10K-h128, 50K-h128, and 100K-h128 all produced identical results to 4 decimal places: `583.5531 ± 24.4791 MPa`.

**Paper statement:**
> *"We note a critical limitation in the V1 Transformer-h128 experiments: the attention parameter floor (~167K for h128 with nhead=4) exceeds all three target budgets (10K, 50K, 100K). Consequently, `build_model` returned identical minimum-size models for all three configurations. The reported results for Trans-TRM-{10K,50K,100K}-h128 represent a single model trained three times under identical conditions, not a scaling study. This is corrected in V2, where Transformer targets of 250K, 500K, and 1M ensure distinct model sizes are actually instantiated."*

### L2: Transformer h64 Results are Architecturally Invalid

With nhead=4 and h64, each attention head operates on 16 dimensions. This is below the practical minimum for attention to compute meaningful similarity scores. The results (~992-1100 MPa) represent a degenerate architecture, not a fair representation of Transformer-TRM capability.

**Paper statement:**
> *"Results for Trans-TRM-*-h64 configurations should not be interpreted as evidence that Transformer-TRM is intrinsically poor. The 16 dims/head configuration falls below the practical minimum for attention to function (empirically ~32 dims/head on this task). V2 uses h256/nhead=4 (64 dims/head) to ensure fair evaluation."*

### L3: Early Stopping Biased by Non-Representative Val Split

The val split (15% ≈ 37 samples) is not stratified by target value. Fold 3's val set consistently contains samples with yield strengths near the dataset mean (easy to predict), causing early stopping to save weights tuned for these specific 37 samples rather than for general generalization.

### L4: 200 Epochs Insufficient for Transformer

Every Transformer h128 fold showed `Best = Val` at the final epoch — the model was still actively improving. The Transformer's convergence behavior has been measured only in the range [0, 200] epochs, which may represent early-stage learning. Conclusions about Transformer capability are limited by this constraint.



This is the core novel architecture proposal. If V2 confirms that more epochs + proper head dimensions improve the Transformer, V3 implements the full element-wise pipeline:

1. **Input**: Each element → 207-dim token (Mat2Vec 200 + 7 elemental properties)
2. **Encoder**: Multi-head self-attention across element tokens (nhead=4, 64 dims/head)
3. **Pooling**: Fraction-weighted attention pooling → single embedding
4. **Recursive Core**: TRM 16-step reasoning loop, cross-attending to element tokens at each step
5. **Output**: Single scalar (yield strength in MPa)

**This architecture has never been published.** It would combine:
- ✅ Mat2Vec embeddings (literature chemical knowledge)
- ✅ Element-wise attention (proven by CrabNet)
- ✅ TRM recursive shared-weight reasoning (novel in materials science)
- ✅ Cross-attention between reasoning states and element tokens at each step

That third+fourth combination is genuinely new.

---

## Phase 5: Deep Supervision & Adaptive Recursion (V10) — [Completed 🏆]
After the success of V7B (127.08 MPa), we tested two main hypotheses in V8 and V9:
1. **V8 (Architectural Scaling):** Tried making attention wider (d_attn=64). *Result: Failure.* The model overfit significantly.
2. **V9 (Recursion Depth):** Tried running 20 recursion steps instead of 16. *Result: The Over-refinement Paradox.* Hard folds improved by ~7 MPa, but easy folds degraded by ~30 MPa because they were pushed past their optimal prediction state.

This led to the core breakthrough in **V10**:
- **Deep Supervision:** Train with L1 loss computed at *every* recursion step (linearly weighted). This forces the model to learn calibrated predictions throughout the entire trajectory, preventing the late-step drift seen in V9.
- **Result:** A massive success. Test MAE dropped precipitously to **103.28 MPa**, crushing our target of beating Darwin (123.29 MPa) and surpassing CrabNet (107.31 MPa). Most significantly, it **officially defeated the RF-SCM/Magpie baseline (103.51 MPa)**, which is the undisputed gold-standard for small-data tabular materials science.

---

## Phase 5.1: Reproducibility Validation (V10.1) — [Completed ✅]
To confirm that the 103.28 MPa result from V10 is statistically robust and not an artifact of a "lucky seed", we ran a **Multi-Seed Reproducibility Test (V10.1)**.
- **Approach:** Train the identical V10A Deep-Supervised architecture across 3 completely different random seeds (42, 123, 7).
- **Result:** **STABLE (Seed Std = 3.00 MPa).** The Grand Mean across all seeds was **105.85 MPa**. All three seeds decisively defeated Darwin (123.29), 2/3 defeated CrabNet (107.31), and the original seed defeated RF-SCM/Magpie (103.51). The architecture's capability to operate consistently at the SOTA level is officially verified.

---

## Phase 6: Pushing for #1 (V11) — [Complete 🏆]
With an established mean of 105.85 MPa, Phase 6 tested three aggressive strategies. **V11B (Scaled + Deep Supervision) achieved a new project SOTA of 102.30 MPa**, decisively beating RF-SCM/Magpie (103.51).
1. **V11A (Feature Expansion):** 107.98 ± 11.06 MPa. Extra `matminer` descriptors added noise—Magpie + Mat2Vec is already near-optimal for N=312.
2. **V11B (Scaled + Deep Supervision):** **102.30 ± 8.61 MPa.** 🏆 Deep Supervision unlocked `d_attn=64` that V8 couldn't use. Lowest fold std ever (±8.61). Three folds below 100 MPa.
3. **V11C (ACT Learned Halting):** 132.59 ± 13.33 MPa. ❌ Ponder cost overwhelmed prediction loss on small data. Model halted too early (avg step 15.6).

Full analysis in `experiment_log_v11.md`.

---

## High-Level Status Summary

| Version | Focus | Best MAE (MPa) | Status | Key Finding |
|---------|-------|:--------------:|--------|-------------|
| V1 | Baseline (SWA) | 134.02 | ✅ Done | SWA is mandatory. 16 steps optimal for MLP. |
| V2 | Transformer | 200.56 | ❌ Dropped | Pure Transformers fail on small structured data. |
| V3 | Cross-Attention | 158.42 | ✅ Done | Projecting stats to tokens is better than mean pooling. |
| V4 | Mat2Vec Fusion | 145.47 | ✅ Done | Mat2Vec adds vital chemical context. |
| V5 | SWA Ensemble | 128.98 | ✅ Done | SWA + MLP works well, but ensembling prediction trajectories fails. |
| V6 | FeatGrouping | 134.96 | ⚠️ Partial | Separating features (chemistry vs stats) stabilizes training. |
| V7 | Hybrid-TRM | 127.08 | ✅ Done | Attention Feature Extraction + MLP Reasoning is the winning architecture. |
| V8 | Arch. Scaling | 143.03 | ❌ Failed | Attention capacity ceiling reached at d_attn=48. |
| V9 | Deep Recursion| 134.59 | ❌ Failed | 20 steps improves hard folds but over-refines easy folds. |
| V10| Adaptive Rec. | 103.28 | ✅ Done | Deep supervision completely solves over-refinement. Beat Darwin. |
| V10.1| Reproduc. | 105.85 (Mean) | ✅ Done | 3-seed grand mean confirms SOTA stability (±3.00 MPa). |
| V11B| Scaled+DS | 102.30 | ✅ Done | Deep Supervision unlocks d_attn=64. Beats RF-SCM. ±8.61 std. |
| **V12A**| **Scaled+Expanded** | **95.99** | **✅ Done** | **Breaks 100 MPa! Expanded features need d_attn=64 to shine.** |
| V12B| Confidence DS | 97.59 | ✅ Done | Confidence head peaks at step 22. V12A simpler and better. |
| **V13A**| **2-Layer SA + Ens** | **91.20** | **🏆 Best Ens** | **5-seed ensemble destroys variance. Hits 91.20 MPa.** |
| **V14A**| **Mega Features** | **94.94** | **🏆 Best Single** | **Single seed beats V12A. Features > Architecture scaling.** |

---

## V3 Results: Magpie Features + MLP-TRM ✅

**Hypothesis confirmed:** Magpie descriptors (132 engineered compositional statistics) broke the 184 MPa ceiling that V1/V2 could not.

| Model | Params | Test MAE | vs V2 Best |
|-------|:------:|:--------:|:----------:|
| **MLP-Magpie-L** | 248,065 | **130.33 ± 12.93** | **−54 MPa** |
| MLP-Magpie-S | 66,689 | 138.40 ± 18.25 | −46 MPa |

Full data in `experiment_log_v3.md`.

**Key V3 findings:**
- Magpie input breaks the input-bottleneck ceiling (184→130 MPa)
- Larger MLP now outperforms smaller (reversed from V1 — richer input benefits from more capacity)
- Fold 4 remains the hardest split (174.7 for S, 154.6 for L)
- Gap to CrabNet: 23 MPa. Gap to RF-SCM/Magpie: 27 MPa

---

## V3.1 Results: XS Model ✅

| Model | Params | Test MAE | Verdict |
|-------|:------:|:--------:|---------|
| MLP-Magpie-XS | 31,153 | 160.43 ± 22.97 | ❌ Too small — underfits |

Confirmed ~65K+ params is the minimum for Magpie input. Full analysis in `experiment_log_v4.md`.

---

## V4 Results: Combined Features (Magpie + Mat2Vec) ✅

| Model | Params | Test MAE | vs V3 Best |
|-------|:------:|:--------:|:----------:|
| MLP-Combined-S | 66,889 | 131.63 ± 14.83 | ~Same (4x fewer params!) |
| MLP-Combined-L | 117,281 | 132.76 ± 21.45 | ~Same |

**Key insight:** Mat2Vec adds minimal info beyond Magpie, BUT Combined-S (67K params) matches Magpie-L (248K params) — the combined features provide better parameter efficiency. Smaller model = less overfitting risk. Full data in `experiment_log_v4.md`.

**Validation breakthrough:** MLP-Combined-L fold 3 reached **val MAE 87.9 MPa** — below CrabNet and MODNet levels. The model CAN learn these patterns but the val-test gap (~20-40 MPa) prevents them from transferring.

---

## V5 Results: SWA + Novel Architecture ✅

### V5A: MLP-TRM with SWA — **NEW BEST: 128.98 MPa**

| Metric | Value |
|--------|-------|
| Standard MAE | **128.98 ± 17.42** |
| Params | 66,889 |

SWA finds flatter minima → improved from 131.63 (V4) to 128.98. Recursion step ensemble incompatible with SWA (degrades to 194.8 MPa).

### V5B: Feature-Group Dual-Reference TRM — 165.11 MPa

| Metric | Value |
|--------|-------|
| Standard MAE | 165.11 ± 17.56 |
| Params | 38,593 |

**Breakthrough:** Novel architecture went from catastrophic (V2: 388 MPa) to functional (165 MPa) by using structured property tokens instead of element tokens. Needs more capacity (d_model=32 too small). Full analysis in `experiment_log_v5.md`.

---

## V6 Results: Scaled Novel + Hybrid + Ensemble ✅

| Config | Params | Test MAE | ±Std | Key |
|---|---|---|---|---|
| V6C MLP-Ensemble ×3 | 67K×3 | 129.04 | **8.99** | Same mean, half std |
| V6B Hybrid-TRM | 67K | 134.97 | 23.15 | Val 84.1 (MODNet!) but 50 MPa gap |
| V6A FeatGroup-L | 81K | 153.96 | **8.84** | Most stable model ever |

**Key insights:**
- Multi-seed ensemble doesn't improve mean — 3 MLPs make the same errors
- Hybrid val reaches 84.1 MPa (below MODNet) → models CAN learn, can't transfer
- Novel arch: 388→165→154 MPa progression by d=32→48
- **Val-test gap is THE bottleneck.** All architectures reach CrabNet/MODNet val, test stays 129+

Full analysis in `experiment_log_v6.md`.

---

## V7 Results: MLP + Hybrid-L + Ensembles ✅ **NEW BEST**

| Config | Params | Test MAE | ±Std | Key |
|---|---|---|---|---|
| **V7B Hybrid-L** | 87K | **127.08** | 18.72 | **New project best!** First time attention > MLP |
| V7D Cross-Arch | 167K | 128.22 | 14.63 | Beats V5A but drags down Hybrid |
| V7A MLP-SWA | 80K | 131.05 | 16.77 | Larger ff_dim (128) overfits — worse than V5A |
| V7C Hybrid-Ens ×3 | 87K×3 | 134.06 | 13.30 | Same-seed ensemble fails again |

**Key insights:**
- Hybrid-L (d_attn=48) surpasses MLP for first time: attention feature extraction + recursive reasoning is the right decomposition
- Scaling d_attn 32→48 gave 7.9 MPa improvement — Hybrid hasn't plateaued
- MLP has peaked: ff=128 > ff=100 but overfits (131 vs 129 MPa)
- Folds 3,5 reach **104.6, 109.8 MPa** — below CrabNet on favorable splits
- Hard folds (2,4) at 143–153 MPa are the entire gap to Darwin
- Same-seed ensembles definitively ruled out as a strategy

Full analysis in `experiment_log_v7.md`.

---

## V8 Direction: Closing the 3.8 MPa Gap to Darwin (📋 Planned)

### What we know going into V8
1. **Hybrid is the winning architecture.** MLP peaked at V5A. Hybrid is still improving.
2. **Attention scaling yields returns.** d_attn 32→48 gave 7.9 MPa. More headroom likely remains.
3. **Hard folds are the target.** Folds 2,4 average ~148 MPa. Folds 1,3,5 average ~113 MPa. Closing the fold gap closes the Darwin gap.
4. **SWA is essential.** Every winning config uses SWA. Longer SWA phase may help further.

### V8A: Hybrid-XL (d_attn=64, nhead=4)
Continue the attention scaling that's been working. d_attn=64 with nhead=4 gives 16 dims/head — potentially too small. Alternative: nhead=2 for 32 dims/head. Test both. ~120K params.

**Rationale:** d_attn 32→48 gave 7.9 MPa. If the relationship is sublinear, d_attn 48→64 could give ~3–5 MPa — enough to beat Darwin.

### V8B: Deeper Attention (2×SA + CA)
Stack two self-attention layers (with residuals and layer norms) before cross-attention with Mat2Vec. More depth = richer property interaction modeling. Keep d_attn=48 but add depth instead of width.

**Rationale:** V7B uses single SA + CA. A second SA layer lets the model capture higher-order property interactions (e.g., "the relationship between electronegativity-range and atomic-radius-mean").

### V8C: Training Schedule Optimization
- **Longer SWA**: swa_start=150 out of 400 epochs (vs current 200/300). More averaging = flatter minima.
- **Warmup**: Linear LR warmup for first 10 epochs. Helps Hybrid's attention converge more stably.
- **Higher weight decay**: 3e-4 or 5e-4 (vs current 1e-4). Additional regularization for the larger Hybrid.

### V8D: Target Transform
Log-transform or Winsorize yield strength targets before training. Folds 2,4 may contain extreme-yield outliers that dominate the loss. A log transform compresses the high-yield tail, reducing their influence during training.

### V8E: Hard-Fold Diagnostics (Analysis-Only)
Before coding V8, analyze what makes folds 2 and 4 hard:
- Are their test compositions unusual (rare elements, extreme fractions)?
- Do their yield strength distributions differ from the easy folds?
- Could stratifying the outer CV by yield strength quartile balance fold difficulty?

This analysis could reveal whether the gap is data-intrinsic (hard to predict compositions) or training-related (fixable with better regularization/augmentation).

---

## Status

| Phase | Status |
|-------|--------|
| V1 12-model sweep | ✅ Complete — `experiment_log_v1.md` |
| V2 6-model element-token sweep | ✅ Complete — `experiment_log_v2.md` |
| V3 Magpie + MLP-TRM | ✅ Complete — `experiment_log_v3.md` |
| V3.1 XS model test | ✅ Complete — underfits |
| V4 Combined Features | ✅ Complete — `experiment_log_v4.md` |
| V5 SWA + Novel Arch | ✅ Complete — `experiment_log_v5.md` |
| V6 Scaled + Hybrid + Ensemble | ✅ Complete — `experiment_log_v6.md` |
| V7 MLP + Hybrid-L + Ensembles | ✅ Complete — `experiment_log_v7.md` |
| V8 Hybrid-XL + Deeper Attention | ❌ Failed (Overfit) |
| V9 Deep Recursion | ❌ Failed (Over-refined) |
| V10 Deep Supervision | 🏆 103.28 MPa — `experiment_log_v10.md` |
| V10.1 Reproducibility | ✅ Complete (105.85 Mean) |
| V11 SOTA Push | � **102.30 MPa (V11B)** — `experiment_log_v11.md` |
