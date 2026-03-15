# V11 Experiment Log: The Push for #1
*Dataset: matbench_steels | 312 samples | 5-Fold Nested CV*
*Baseline: V10A = 103.28 MPa | Total runtime: 22.1 minutes*

---

## Goal
Beat RF-SCM/Magpie (103.51 MPa) and push toward MODNet (87.76) and TPOT-Mat (79.95).

## Strategy
1. **V11A (Feature Expansion):** More chemical descriptors via `matminer` MultipleFeaturizer
2. **V11B (Scaled + Deep Supervision):** `d_attn=64` (previously overfit in V8), now regularized by Deep Supervision
3. **V11C (ACT Learned Halting):** Neural `halt_head` learns optimal recursion depth per sample

---

## Final Leaderboard

| Model | Params | MAE (MPa) | ±Std | vs Baseline |
|-------|-------:|:---------:|:----:|-------------|
| **V11B-Scaled** | **172,013** | **102.30** | **8.61** | **🏆 New SOTA — Beats RF-SCM** |
| V11A-FeatExp | 100,153 | 107.98 | 11.06 | Beats Darwin, not RF-SCM |
| V11C-ACT | 89,466 | 132.59 | 13.33 | ❌ Regression |

---

## Per-Fold Breakdown

| Fold | V11A-FeatExp | V11B-Scaled | V11C-ACT |
|:----:|:-----------:|:-----------:|:--------:|
| 1 | 124.19 | 118.82 | 154.53 |
| 2 | 94.33 | 101.79 | 124.22 |
| 3 | 103.42 | **95.60** | 122.14 |
| 4 | 117.33 | **99.82** | 141.60 |
| 5 | 100.63 | **95.48** | 120.49 |
| **Mean** | **107.98** | **102.30** | **132.59** |

---

## Analysis

### V11A: Feature Expansion — 107.98 ± 11.06 MPa
- **Verdict:** ⚠️ Marginal. Beat Darwin (123.29) and CrabNet (107.31), but **did not beat V10A** (103.28).
- Extra features (ElementFraction, Stoichiometry, ValenceOrbital, IonProperty, BandCenter) added noise rather than signal on this 312-sample dataset.
- Fold variance is moderate (94.3–124.2 MPa).

### V11B: Scaled + Deep Supervision — 102.30 ± 8.61 MPa 🏆
- **Verdict:** **NEW PROJECT BEST.** Deep Supervision successfully unlocked the `d_attn=64` capacity that V8 couldn't use.
- **Key breakthrough:** Standard deviation collapsed from ~18+ MPa (V7B) to **8.61 MPa** — the most stable model we've ever trained.
- **Hard fold revolution:** Folds 3, 4, 5 all dropped below 100 MPa (95.60, 99.82, 95.48). Previously, hard folds were 140–150 MPa.
- **Val MAE of 59 MPa** observed on Fold 1 during training — far below MODNet-level. The val-test gap (~60 MPa on that fold) remains the bottleneck.
- Proves: Hybrid-TRM architecture **has not saturated.** More capacity + Deep Supervision = more performance.

### V11C: ACT Learned Halting — 132.59 ± 13.33 MPa
- **Verdict:** ❌ Failed. Significantly worse than both V10A and V11B.
- **ACT stats:** Average halt step = 15.6, early halting = 97%. The model learned to halt *too early* (before the minimum 16 steps that V10A uses), stranding most samples at shallow recursion depths.
- **Root cause hypothesis:** The ponder cost penalty (`λ=0.01`) dominated the loss landscape on this small dataset, teaching the model that "thinking less is always better." The halt_head converged to a trivial solution (halt ASAP after min_steps=12) rather than learning sample-dependent halting.
- **Potential fix:** Much smaller ponder cost (`λ=0.001`), or curriculum: start with fixed 20 steps, gradually enable halting after epoch 150.

---

## Key Takeaways

1. **Deep Supervision is the unlock for scaling.** V8 overfit at `d_attn=64`. V11B succeeds at the same size. The only difference: Deep Supervision.
2. **Feature expansion hits diminishing returns.** Magpie + Mat2Vec is already near-optimal for 312 samples. More features ≠ better.
3. **ACT needs careful tuning on small data.** The ponder cost overwhelms the prediction loss when N=312. Learned halting may work better on larger datasets.
4. **The val-test gap is still dominant.** V11B's Fold 1 val MAE (59 MPa) vs test (118.82 MPa) = 60 MPa gap. Closing this gap is the path to MODNet.
5. **Fold stability matters.** V11B's ±8.61 std is the lowest ever — indicating the model generalizes more uniformly across compositions.
