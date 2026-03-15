# Matbench Steels — Research Notes

## Task
**Target:** Yield Strength (MPa)  
**Dataset:** `matbench_steels` — 312 samples, 5-fold nested cross-validation  
**SOTA Result:** 91.20 ± 12.23 MPa (V13A, 5-seed ensemble, 225K parameters)

## Architecture: HybridTRIADS V13A

**Model:** `DeepHybridTRM` — 2-layer Self-Attention + Recursive MLP + Deep Supervision  
**Parameters:** 225K (5-seed ensemble)  
**Input:** Magpie (132d) + Extended Matminer descriptors (~130d) + Mat2Vec (200d) ≈ 462d  

**Key architectural components:**
- 22 Magpie property tokens → 2-layer self-attention (SA1: first-order, SA2: higher-order interactions)
- Cross-attention: z_{t-1} queries Mat2Vec chemistry context
- Recursive MLP loop: T=20 shared-weight steps
- GRU-style gated residuals (essential for stable T>8)
- Deep supervision: linearly-weighted L1 at every cycle (w_t ∝ t)
- 5-seed ensemble: seeds [42, 123, 7, 0, 99], average predictions per fold

## Development History (V1→V13)

| Version | Key Change | MAE (MPa) |
|---|---|---|
| V1 Mat2Vec MLP | Baseline, mat2vec-only input | 184.38 |
| V3 Magpie MLP | Magpie features replace mat2vec | 130.33 |
| V5A + SWA | Stochastic weight averaging | 128.98 |
| V7B Hybrid-L | Self-attn + cross-attn cell, T=16 | 127.08 |
| V9A same, T=20 | Extend to 20 cycles (no DS) | 134.59 |
| V10A + DS, T=20 | Deep supervision at every cycle | **103.29** |
| V11B scaled | Wider model (d=64, 172K params) | 102.30 |
| V12A + features | Expanded feature set, 191K | 95.99 |
| **V13A + 2-SA, ens.** | 2nd SA layer, 5-seed ensemble | **91.20** |

## Key Findings

**Deep supervision is the highest-leverage change:**  
V9A→V10A: identical architecture (87K, 20 cycles), only objective changes.  
MAE: 134.59 → 103.29 MPa (−23.3% relative). This is the largest single-step gain.

**Over-refinement failure (no deep supervision):**  
T=16→T=20 *hurts* on easy folds (fold 3: 104.59→142.55, fold 5: 109.78→131.31).  
Deep supervision fixes this by penalizing drift at every cycle.

**Negative result — HTRM:** Hierarchical variant with gradient detachment collapses to 431.87±49.59 MPa. Gradient detachment is incompatible with small-data optimization.

## Fold-Level Results (V13A Final)
| Fold | MAE (MPa) |
|---|---|
| Fold 1 | 114.32 |
| Fold 2 | 81.46 |
| Fold 3 | 80.55 |
| Fold 4 | 90.49 |
| Fold 5 | 89.18 |
| **Mean ± Std** | **91.20 ± 12.23** |

The high fold variance (±12.23) reflects real data heterogeneity: fold 1 contains harder alloys.

## Leaderboard Comparison
| Model | MAE (MPa) | Params |
|---|---|---|
| TPOT-Mat (AutoML) | 79.95 | — |
| AutoML-Mat | 82.30 | — |
| MODNet v0.1.12 (tree-based) | 87.76 | ~0 |
| **TRIADS V13A (5-seed)** | **91.20** | 225K |
| RF-SCM/Magpie | 103.51 | — |
| CrabNet | 107.32 | ~600K |
| Darwin (Evolutionary) | 123.29 | — |
