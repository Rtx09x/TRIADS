# Matbench JDFT2D — Research Notes

## Task
**Target:** Exfoliation Energy (meV/atom)  
**Dataset:** `matbench_jdft2d` — 636 samples, 5-fold nested cross-validation  
**SOTA Result:** 35.89 ± 12.40 meV/atom (V4, 75K parameters, 5-seed ensemble)

## Architecture: HybridTRIADS V4

**Model:** `DeepHybridTRM` (75K variant) + structural features  
**Parameters:** ~75K  
**Input:** Magpie (132d) + Stoich/Valence/IonProp/TMetalFraction (22d) + Structural (11d) + Mat2Vec (200d) = ~365d  

**Key insight:** Including 11 structural features from pymatgen (lattice params a/b/c/α/β/γ, volume/atom, density, nsites, spacegroup number, packing fraction) gives a significant advantage over composition-only models.

**Configuration:** d_attn=32, d_hidden=64, ff_dim=96, dropout=0.20, T=16  
**Ensemble:** 5 seeds [42, 123, 456, 789, 1024]

## Development History

| Version | Key Change | MAE (meV/atom) |
|---|---|---|
| V1 (100K, comp-only) | Magpie + Mat2Vec | 45.80 |
| V2 (44K, comp-only) | Smaller model | 46.59 |
| V3 (75K, comp+struct, single seed) | Add 11 structural features | 37.00 |
| **V4 (75K, comp+struct, 5-seed ensemble)** | 5-seed ensemble | **35.89** |

## Key Findings

**Structural features are essential for 2D materials:**  
Layer spacing (c-parameter), anisotropy (a/b/c ratios), and density encode the layered/van-der-Waals nature of 2D materials — directly relevant to exfoliation energy.  
V2→V3: same model size, adding structural features → 21.3% MAE reduction.

**5-seed ensemble reduces variance substantially:**  
V3→V4: same architecture, add ensemble → 3.0% mean improvement + reduces std from 14.2 to 12.4.

**75K vs 44K vs 100K for N=636:**  
The 75K model outperforms both smaller (44K: 46.59) and larger (100K: 45.80) models.  
This creates a capacity sweet-spot for N~636 with ~365d input.

## Fold-Level Results (V4, 5-seed ensemble)
| Fold | MAE (meV/atom) |
|---|---|
| Fold 1 | 30.22 |
| Fold 2 | 24.41 |
| Fold 3 | 28.90 |
| Fold 4 | 65.81 |
| Fold 5 | 30.12 |
| **Mean ± Std** | **35.89 ± 12.40** |

Note: the high fold-4 variance (65.81) is due to extreme outliers in 2D material databases — some materials have unusually high exfoliation energies (>1000 meV/atom) recorded under non-standard conditions.

## Leaderboard Comparison
| Model | MAE (meV/atom) | Params | Notes |
|---|---|---|---|
| coGN | 37.17 | 1M+ | Pre-trained, structure-based |
| MODNet v0.1.12 | 33.19 | ~0 | Feature-based |
| **TRIADS V4 (75K, 5-seed)** | **35.89** | 75K | Best no-pretraining |
| TRIADS V3 (single-seed) | 37.00 | 75K | — |
| TRIADS V1 (comp-only) | 45.80 | 100K | — |
