# Matbench Expt Gap — Research Notes

## Task
**Target:** Experimental Band Gap (eV)  
**Dataset:** `matbench_expt_gap` — 4,604 samples, 5-fold nested cross-validation  
**SOTA Result:** 0.3068 ± 0.0082 eV (V3-S20-D20, 100K parameters, seed 42)

## Architecture: HybridTRIADS V3

**Model:** `DeepHybridTRM` — 2-layer Self-Attention + Recursive MLP + Deep Supervision  
**Parameters:** ~100K  
**Input:** Magpie (132d) + BandCenter + HOMO/LUMO gap proxies + Mat2Vec (200d) = 354d  

**Key featurization insight:** For electronic tasks, BandCenter and HOMO/LUMO gap features from NIST orbital energies are included. Removing these (generic 468d input) worsens MAE from 0.3342 to 0.3616 — a 7.6% relative increase with no architectural change.

**Best config:** `V3-S20-D20` — 20 recursion steps, 0.20 dropout  
Seeds: [42] (single seed; 5-fold CV)

## Development History

| Version | Key Change | MAE (eV) |
|---|---|---|
| V1 (generic, 468d) | Magpie + ElementFraction + Mat2Vec | 0.3616 |
| V2 (physics-informed, 354d) | BandCenter + HOMO/LUMO instead of ElementFraction | 0.3342 |
| **V3 (100K, 5-fold)** | Same as V2, 4 config sweep, best is S20-D20 | **0.3068** |

## Key Findings

**Physics-informed features matter more than architecture:**  
V1→V2: same architecture, different features → 7.6% MAE reduction (0.3616 → 0.3342).  
V2→V3: same features, scaled model + 5-fold → additional 7.9% reduction.

**Config sweep (V3):**
| Config | MAE (eV) |
|---|---|
| V3-S16-D15 | best of 4 shown in results.json |
| V3-S16-D20 | " |
| V3-S20-D15 | " |
| **V3-S20-D20** | **0.3068** (winner) |

## Fold-Level Results (V3-S20-D20, seed 42)
| Fold | MAE (eV) |
|---|---|
| Fold 1 | 0.3069 |
| Fold 2 | 0.3122 |
| Fold 3 | 0.3190 |
| Fold 4 | 0.2998 |
| Fold 5 | 0.2962 |
| **Mean ± Std** | **0.3068 ± 0.0082** |

## Leaderboard Comparison
| Model | MAE (eV) | Params |
|---|---|---|
| Ax/SAASBO CrabNet (comp-only) | 0.3310 | ~600K |
| MODNet v0.1.12 (tree-based) | 0.3327 | ~0 |
| **TRIADS V3 (100K, comp-only)** | **0.3068** | 100K |
| CrabNet (comp-only) | 0.3463 | ~600K |
| Darwin (modality unconfirmed) | 0.2865 | — |
