# TRIADS Hyperparameter Studies and Ablation Analysis

*Detailed technical analysis of hyperparameter sensitivity, architectural ablations, and design decisions derived from ~200 models across 15 major versions.*

> **See also:** [Architecture Evolution](./Architecture_Evolution.md) · [Performance Logs](./Performance_Logs.md) · [README](../README.md)
>
> **Result plots:** [V2](../Images/trm_results_v2.png) · [V3](../Images/trm_results_v3.png) · [V5](../Images/trm_results_v5.png) · [V7](../Images/trm_results_v7.png) · [V8](../Images/trm_results_v8.png) · [V9](../Images/trm_results_v9.png) · [V10](../Images/trm_results_v10.png) · [V11](../Images/trm_results_v11.png) · [V12](../Images/trm_results_v12.png) · [V13](../Images/trm_results_v13.png) · [V14](../Images/trm_results_v14.png)

---

## 1. Attention Dimension Scaling

The relationship between attention dimension (`d_attn`) and model performance is **non-monotonic** and critically dependent on the presence of Deep Supervision as a regularizer.

### Without Deep Supervision

| d_attn | Architecture | Version | MAE (MPa) | Observation |
|:------:|:------------|:--------|:---------:|:------------|
| 32 | Hybrid-TRM (1 SA layer) | [V6B](../Training%20Code/trm6.py) | 134.97 | Functional baseline |
| 48 | Hybrid-TRM (1 SA layer) | [V7B](../Training%20Code/trm7.py) | 127.08 | +7.9 MPa improvement |
| 64 | Hybrid-TRM (1 SA layer) | [V8B](../Training%20Code/trm8.py) ([📊](../Images/trm_results_v8.png)) | 155.06 | **Catastrophic overfitting** (+28 MPa) |

**Finding:** Without Deep Supervision, `d_attn=48` is the maximum usable capacity for N=312 samples. The 32→48 scaling trend does not extrapolate; 48→64 causes the model to memorize training patterns rather than learn generalizable representations.

### With Deep Supervision

| d_attn | Architecture | Version | MAE (MPa) | Observation |
|:------:|:------------|:--------|:---------:|:------------|
| 48 | Hybrid-TRM + DS (20 steps) | [V10A](../Training%20Code/trm10.py) | 103.28 | Deep Supervision baseline |
| 64 | Hybrid-TRM + DS (20 steps) | [V11B](../Training%20Code/trm11.py) | 102.30 | **Successful scaling** |
| 64 (2×SA) | Deep Hybrid-TRM + DS (20 steps) | [V13A](../Training%20Code/trm13.py) | ~96.77 (single) | Further improvement with depth |

**Principle:** Deep Supervision acts as a regularizer that unlocks architectural scaling on small datasets. The same `d_attn=64` that causes catastrophic overfitting at V8 becomes the project's best-performing configuration at V11B when regularized by step-wise loss computation.

---

## 2. Recursion Depth

### Fixed Steps Without Deep Supervision

| Steps | Version | MAE (MPa) | ±Std | Notes |
|:-----:|:--------|:---------:|:----:|:------|
| 16 | [V7B](../Training%20Code/trm7.py) | 127.08 | 18.72 | Optimal for SWA-calibrated weights |
| 20 | [V9A](../Training%20Code/trm9.py) ([📊](../Images/trm_results_v9.png)) | 134.59 | 10.43 | Over-refinement paradox |

**The Over-Refinement Paradox (V9):** Per-fold analysis revealed that extending recursion from 16 to 20 steps improved hard folds by ~7 MPa each but degraded easy folds by 22–38 MPa. The SWA-averaged weights were calibrated for 16-step trajectories; running them for 20 steps caused easy samples to drift past their optimal prediction state.

| Fold | V7B (16 steps) | V9A (20 steps) | Delta |
|:----:|:--------------:|:--------------:|:-----:|
| 1 | 124.56 | 116.32 | −8.2 ✅ |
| 2 | 153.03 | 146.07 | −7.0 ✅ |
| 3 | 104.59 | 142.55 | +38.0 ❌ |
| 4 | 143.42 | 136.70 | −6.7 ✅ |
| 5 | 109.78 | 131.31 | +21.5 ❌ |

### Fixed Steps With Deep Supervision

| Steps | Version | MAE (MPa) | ±Std | Notes |
|:-----:|:--------|:---------:|:----:|:------|
| 20 | [V10A](../Training%20Code/trm10.py) ([📊](../Images/trm_results_v10.png)) | 103.28 | 10.49 | Over-refinement completely solved |

Deep Supervision forces every step to produce a calibrated prediction, preventing the late-step drift that destroyed V9. The 20-step DS model improved **every fold** relative to the 16-step non-DS baseline, including the easy folds that V9 destroyed.

### Adaptive Halting Mechanisms

**V10B (Convergence-based halting):** Stop recursion per-sample when `|pred_t − pred_{t-1}| < 1.0 MPa`, minimum 12 steps. Result: 104.87 MPa—slightly worse than fixed-20 (103.28). Because Deep Supervision made step 20 universally robust, halting early forfeited marginal refinement gains. Adaptive halting saves ~25% compute but sacrifices ~1.6 MPa accuracy.

**V11C (ACT Learned Halting):** A neural `halt_head` learns when to stop per-sample using Adaptive Computation Time (Graves, 2016). Result: **132.59 MPa (failure).** The ponder cost penalty (`λ=0.01`) dominated the loss landscape on small data, teaching the model to halt as early as possible (average step 15.6). The halt head converged to a trivial solution rather than learning sample-dependent behavior.

---

## 3. Feature Space Ablation

### Input Representation Comparison

| Input | Dimensions | Best MAE (MPa) | Version | Key Insight |
|:------|:----------:|:--------------:|:--------|:------------|
| Mat2Vec weighted sum | 200 | 184.38 | [V1](../Training%20Code/trm.py) | Information bottleneck at input |
| Element-as-token (per-element) | 205/token | 388.58 | [V2](../Training%20Code/trm2.py) | Attention cannot discover interactions from 312 samples |
| Magpie descriptors only | 132 | 130.33 | [V3](../Training%20Code/trm3.py) | Engineered features break the ceiling |
| Magpie + Mat2Vec (Combined) | 332 | 131.63 (67K params) | [V4](../Training%20Code/trm4.py) | Same accuracy with 4× fewer parameters |
| Combined + SWA | 332 | 128.98 | [V5A](../Training%20Code/trm5.py) | SWA finds flatter minima |
| Combined + matminer extras | ~462 | 107.98 | [V11A](../Training%20Code/trm11.py) | Extra features add noise at d_attn=48 |
| Combined + matminer extras | ~462 | 95.99 | [V12A](../Training%20Code/trm12.py) | Same features excel at d_attn=64 |
| Mega-features (all matminer) | ~670 | 94.94 | [V14A](../Training%20Code/trm14.py) | Further gains from domain-specific descriptors |

**Coupling effect between features and architecture:** Expanded matminer features failed on smaller architectures (V11A: 107.98 MPa with `d_attn=48`) but excelled on larger ones (V12A: 95.99 MPa with `d_attn=64`). More chemical descriptors require more attention capacity to extract their signal. Neither scaling alone was sufficient—the breakthrough required both.

### Property Token Structuring

| Token Strategy | Version | MAE (MPa) | Notes |
|:--------------|:--------|:---------:|:------|
| Raw element tokens (attention) | [V2](../Training%20Code/trm2.py) | 388 | Catastrophic—attention learns nothing |
| 22 Magpie property tokens (d=32) | [V5B](../Training%20Code/trm5.py) | 165 | 223 MPa improvement from restructuring |
| 22 Magpie property tokens (d=48) | [V6A](../Training%20Code/trm6.py) | 154 | Scaling continues to help |
| 58 mega-feature tokens (2-pass SA) | [V14B](../Training%20Code/trm14.py) | 96.15 | Powerful but noisy |

**Finding:** The optimal tokenization strategy for small datasets is to group features by property type (22 Magpie properties), not by element. This provides structured tokens that attention can meaningfully compare, without creating a sparse token matrix that overfits.

---

## 4. Ensemble Strategies

### Same-Architecture Multi-Seed Ensembles

| Strategy | Version | MAE (MPa) | ±Std | Delta vs. Best Single |
|:---------|:--------|:---------:|:----:|:---------------------:|
| MLP-SWA ×3 seeds | [V6C](../Training%20Code/trm6.py) | 129.04 | 8.99 | +0.06 (no improvement) |
| Hybrid-TRM ×3 seeds | [V7C](../Training%20Code/trm7.py) | 134.06 | 13.30 | +6.98 (degradation) |
| **2-Layer Hybrid ×5 seeds** | **[V13A](../Training%20Code/trm13.py)** ([📊](../Images/trm_results_v13.png)) | **91.20** | **12.23** | **−5.57** |

**Critical distinction:** At early performance levels (V6–V7), same-architecture models made the same errors regardless of seed—ensembling reduced variance (±17 → ±9) but not mean. At V13-level performance (~96 MPa single seed), the remaining error became predominantly variance-driven, making multi-seed ensembling the optimal strategy for a 5.57 MPa improvement.

### Cross-Architecture Ensembles

| Strategy | Version | MAE (MPa) | Notes |
|:---------|:--------|:---------:|:------|
| MLP + Hybrid ×3 avg | [V7D](../Training%20Code/trm7.py) | 128.22 | MLP errors drag down superior Hybrid |

**Finding:** Cross-architecture ensembling is counterproductive when one architecture is strictly superior. The weaker model's errors contaminate the stronger model's predictions.

---

## 5. Training Schedule

### Stochastic Weight Averaging (SWA)

SWA is **mandatory** for generalization on this dataset. Every winning configuration from V5 onward uses SWA. The protocol: cosine annealing for 200 epochs (base training), then weight averaging for epochs 200–300 (SWA phase).

**SWA is incompatible with recursion step ensembling.** SWA shifts internal weight distributions, making intermediate recursion steps poorly calibrated. Averaging steps 13–16 at inference degrades MLP performance from 128.98 → 194.84 MPa.

### Epochs and Early Stopping

| Strategy | Applicable To | Notes |
|:---------|:-------------|:------|
| 300 epochs with SWA@200 | All winning models V5+ | Standard training protocol |
| Early stopping (patience=60) | MLP models | MLP converges by epoch 50–100 |
| No early stopping | Transformer variants | Transformers still learning at epoch 200 (V1 finding) |

### Validation Split

A **stratified 15% validation split** (stratified by yield strength quartiles) was adopted from V2 onward, after V1 discovered that the random split produced a Fold 3 bias where validation samples clustered near the dataset mean, distorting early stopping checkpoint selection.

---

## 6. Regularization Analysis

### Dropout

Standard dropout of 0.2 applied at every recursion step. Not extensively ablated—0.1 (V1) was found slightly insufficient, 0.2 (V2 onward) was adopted and retained.

### Weight Decay

AdamW with weight_decay=1e-4 used throughout. Not ablated beyond confirming its necessity.

### Deep Supervision as Regularization

Deep Supervision's regularization effect is its most important property for small datasets. By requiring the shared weights to simultaneously satisfy 20 loss objectives (one per step), the model cannot specialize to any single step's gradient signal. This distributes the learning pressure uniformly across the weight space, acting as an implicit multi-task regularizer that prevents the overfitting observed in V8 when the same architecture was trained with single-step loss only.

### Attention Head Configuration

| Dims/Head | Result | Status |
|:---------:|:-------|:-------|
| 16 | Attention completely non-functional (992 MPa) | ❌ Degenerate |
| 32 | Minimally functional (583 MPa) | ⚠️ Barely viable |
| 64 | Optimal sweet spot for this task | ✅ Used in final architecture |

**Empirical minimum:** 32 dimensions per attention head is the absolute floor for attention to compute meaningful similarity scores on compositional property data. Below this threshold, increasing model parameters paradoxically degrades performance.

---

## 7. Cross-Attention Layer Analysis

The cross-attention layer in the Hybrid-TRM serves as more than context injection—it provides a critical **second layer of computational depth**.

**Evidence ([V8A](../Training%20Code/trm8.py) · [📊 results](../Images/trm_results_v8.png)):** Removing the cross-attention layer and injecting Mat2Vec as a 23rd self-attention token regressed performance by 16 MPa (127.08 → 143.03), despite the attention being "cleaner" (homogeneous token space). The architecture's processing pipeline (`SA → FF → CA`) requires an explicit second computational layer after self-attention to refine representations before they enter the recursive loop.

---

## 8. The Fold Difficulty Structure

A persistent pattern across all versions is a **two-tier fold difficulty structure**:

| Fold Group | Typical MAE Range | Composition |
|:-----------|:-----------------:|:------------|
| Easy (Folds 3, 5) | 90–110 MPa | Common compositions, moderate yield strengths |
| Medium (Fold 1) | 115–125 MPa | Mixed difficulty |
| Hard (Folds 2, 4) | 95–155 MPa | Unusual compositions or extreme yield strengths |

Hard folds accounted for the entire gap to leaderboard baselines through V1–V9. Deep Supervision (V10+) dramatically reduced the hard fold penalty, bringing Folds 2 and 4 into the sub-100 MPa range for the first time.
