# Matbench Classification — Research Notes

## Tasks Covered
This module covers **two classification benchmarks** with a single model file:

1. `matbench_expt_ismetal` — Metal vs. Non-metal (4,921 samples) → ROC-AUC
2. `matbench_glass` — Glass forming ability (5,680 samples) → ROC-AUC

## Architecture: HybridTRIADS (44K / 100K)

**Model:** `DeepHybridTRM` — 2-layer Self-Attention + BCEWithLogitsLoss + Deep Supervision  
**Training loss:** Deep supervision with linear-weighted BCE at every cycle

### matbench_expt_ismetal
**Result:** 0.9655 ± 0.0029 ROC-AUC (100K model, 1 seed)   
**Parameters:** ~100K  
**Featurizer:** `MetallicityFeaturizer` — includes BandCenter, HOMO/LUMO gap + TMetalFraction  
**Input dim:** 354d  

### matbench_glass
**Result:** 0.9285 ± 0.0063 ROC-AUC (44K model, 5-seed ensemble)  
**Parameters:** 44K  
**Featurizer:** `GlassFeaturizer` — excludes BandCenter and HOMO/LUMO (thermodynamics, not electronics)  
**Input dim:** ~351d  

**Note:** 44K outperforms 100K for glass forming (0.9285 vs 0.9259) — consistent with noise-limited targets. Smaller models generalize better when the label itself is noisy (thermodynamic kinetics not captured in static composition data).

## Key Findings

**Sensor design matters:**  
For metallicity, including BandCenter and HOMO/LUMO directly provides electronic-structure priors.  
For glass forming, these features are excluded because mixing thermodynamics (not electronic structure) is the governing mechanism.

**44K vs 100K:**  
The 44K metallicity model achieves 0.9644 vs 100K's 0.9655 — essentially identical within noise.  
This demonstrates that with proper physics-informed sensors, extra parameters add little.

## Fold-Level Results

### matbench_expt_ismetal (100K, seed 42)
| Fold | ROC-AUC |
|---|---|
| Fold 1 | 0.9674 |
| Fold 2 | 0.9648 |
| Fold 3 | 0.9602 |
| Fold 4 | 0.9683 |
| Fold 5 | 0.9669 |
| **Mean ± Std** | **0.9655 ± 0.0029** |

### matbench_glass (44K, 5-seed)
| Fold | ROC-AUC |
|---|---|
| Fold 1 | 0.9244 |
| Fold 2 | 0.9222 |
| Fold 3 | 0.9245 |
| Fold 4 | 0.9392 |
| Fold 5 | 0.9322 |
| **Mean ± Std** | **0.9285 ± 0.0063** |

## Leaderboard Comparison

### matbench_expt_ismetal
| Model | ROC-AUC | Params |
|---|---|---|
| **TRIADS 100K (comp-only)** | **0.9655** | 100K |
| TRIADS 44K (comp-only) | 0.9644 | 44K |
| Darwin | 0.9598 | — |
| AMMExpress v2020 | 0.9209 | — |
| GPTChem | 0.8965 | >1B |

### matbench_glass
| Model | ROC-AUC | Params |
|---|---|---|
| MODNet v0.1.12 (tree-based) | 0.9603 | ~0 |
| **TRIADS 44K (5-seed)** | **0.9285** | 44K |
| AMMExpress v2020 | 0.8607 | — |
| RF-SCM/Magpie | 0.8587 | — |
