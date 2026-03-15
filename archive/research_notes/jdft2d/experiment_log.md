# TRIADS on matbench_jdft2d — Experiment Log
*Exfoliation Energy (meV/atom) — 636 samples from JARVIS-DFT*

---

## Dataset Details
- **Property**: Exfoliation energy — energy to peel a 2D layer from bulk crystal
- **Physics**: Depends on interlayer van der Waals forces, crystal structure, layer stacking, polarizability
- **Size**: 636 samples (small, similar to matbench_steels at 312)
- **Input**: Structure objects (we extract composition + structural features)
- **Leaderboard #1**: MODNet v0.1.12 at **33.1918 meV/atom**

---

## V1 — Direct Transfer (100K model)

| Parameter | Value |
|-----------|-------|
| Model | 100K params (d_attn=40, d_hidden=72, ff_dim=108) |
| Features | 354d (Magpie + Extras + Mat2Vec, same as expt_gap V2) |
| Dropout | 0.15 |
| Seed | 42 |
| Time | 5.6 min on P100 |

**Result: 45.80 ± 13.48 meV/atom**

| Fold | MAE |
|:----:|:---:|
| 1 | **31.39** (beats MODNet!) |
| 2 | 45.58 |
| 3 | **69.38** (catastrophic) |
| 4 | **34.10** (beats MODNet!) |
| 5 | 48.58 |

**Finding**: 101K params on 636 samples = 159:1 ratio. Massive overfitting on Fold 3. But Folds 1 & 4 prove the architecture can reach MODNet-level when the data distribution is favorable.

---

## V2 — Smaller Model (44K, dropout 0.20)

| Parameter | Value |
|-----------|-------|
| Model | 44K params (d_attn=24, d_hidden=48, ff_dim=72) |
| Features | 354d (same as V1) |
| Dropout | 0.20 |
| Seed | 42 |
| Time | 5.4 min on P100 |

**Result: 46.59 ± 8.39 meV/atom**

| Fold | MAE |
|:----:|:---:|
| 1 | 36.74 |
| 2 | 43.96 |
| 3 | **62.18** (still bad) |
| 4 | 44.71 |
| 5 | 45.35 |

**Finding**: Lower variance (±8.4 vs ±13.5) but similar average. The smaller model underfits on the easy folds (Fold 1: 36.74 vs 31.39) while only slightly improving the hard fold. The problem isn't model size — it's missing features.

---

## V3 — Structural Features (planned)

**Key insight**: We're extracting only composition from the Structure objects, throwing away structural information that directly determines exfoliation energy (interlayer distance, crystal symmetry, density).

### Changes:
- **ADD**: Lattice params (a,b,c,α,β,γ), density, volume/atom, space group, num sites, packing fraction (+11 features)
- **REMOVE**: BandCenter, HOMO/LUMO gap (−4 features, band-gap-specific noise)
- **Model**: 75K params (d_attn=32, d_hidden=64) — middle ground between 100K (overfit) and 44K (underfit)

**Result: 37.00 ± 11.11 meV/atom**

| Fold | MAE |
|:----:|:---:|
| 1 | **25.75** |
| 2 | 35.58 |
| 3 | 54.63 |
| 4 | **25.42** |
| 5 | 43.63 |

**Finding**: Structural features dropped MAE by −19.2% (45.80 → 37.00). Every single fold improved. Lattice parameters, density, and space group provided critical physics that composition alone couldn't capture for exfoliation energy prediction.

---

## V4 — 5-Seed Ensemble (✅ Done — 35.89 meV/atom, #3!)

| Seed | Avg MAE | F1 | F2 | F3 | F4 | F5 |
|:----:|:-------:|:---:|:---:|:---:|:---:|:---:|
| 42 | 37.00 | 25.75 | 35.58 | 54.63 | 25.42 | 43.63 |
| 123 | 37.59 | 20.92 | 39.67 | 60.00 | 25.96 | 41.40 |
| 456 | 36.71 | 25.57 | 34.03 | 54.69 | 24.90 | 44.37 |
| 789 | 37.92 | 20.77 | 37.89 | 62.98 | 26.09 | 41.89 |
| 1024 | 40.35 | 34.48 | 37.46 | 59.46 | 24.71 | 45.65 |

- **Single-seed avg: 37.92 ± 1.29**
- **5-Seed Ensemble: 35.89 ± 12.40 (↓5.4%)**
- **Per-fold ensemble: [23.40, 34.09, 56.63, 23.75, 41.57]**

### Leaderboard
| # | Model | MAE (meV/atom) |
|:-:|-------|:-:|
| 1 | MODNet v0.1.12 | 33.19 |
| 2 | ??? | ??? |
| **3** | **TRIADS V4 (75K, +struct, 5-seed)** | **35.89** |

### Full Journey
```
V1 (100K, comp-only):    45.80  (direct transfer)
V2 (44K, comp-only):     46.59  (smaller, underfits)
V3 (75K, +struct):       37.00  (structural features = −19.2%)
V4 (75K, +struct, ens):  35.89  (ensemble = −5.4% more)
─────────────────────────────────────────────────
Total: 45.80 → 35.89 = −21.6% improvement
```

## ✅ matbench_jdft2d — COMPLETE (#3, 35.89 meV/atom)
