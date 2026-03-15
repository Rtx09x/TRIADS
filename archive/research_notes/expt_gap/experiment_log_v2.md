# TRM-MatSci V2 — Experiment Log | matbench_expt_gap
## Featurization Overhaul | 2026-03-04

---

## 1. Objective

**Hypothesis**: The V1 featurization pipeline (468d), designed for mechanical properties (steel yield strength), is suboptimal for electronic properties (band gaps). By removing noisy/sparse features and adding physics-informed descriptors, we can improve performance without changing the model architecture.

**Approach**: Featurization-first — fix the inputs, keep the architecture frozen, compare apples-to-apples with V1-82K.

---

## 2. Featurization Changes (V1 → V2)

### REMOVED
| Featurizer | Features | Reason |
|-----------|:--------:|--------|
| **ElementFraction** | ~80 | Extremely sparse — most features are zeros for any given composition when you have 60+ unique elements. Mat2Vec already captures element identity far better. |

### ADDED
| Featurizer | Features | Reason |
|-----------|:--------:|--------|
| **TMetalFraction** | 1 | Transition metal d-electrons dramatically narrow/close band gaps. Single scalar captures this. |
| **HOMO/LUMO gap** (manual) | 3 | HOMO energy, LUMO energy, and gap estimate from NIST atomic orbital energies. Directly analogous to band gap. Replaces matminer `AtomicOrbitals` which returns string values. |

### KEPT (unchanged from V1)
| Featurizer | Features | Notes |
|-----------|:--------:|-------|
| Magpie | 132 | 22 properties × 6 stats. Includes GSBandGap (legitimate signal), Electronegativity, NValence. |
| Stoichiometry | 6 | L-norms of composition vector (Kaggle matminer: 6, not 5 as documented). |
| ValenceOrbital | 8 | s/p/d/f valence electrons (Kaggle matminer: 8, not 32 as documented). |
| IonProperty | 3 | Ionic character, oxidation states (Kaggle matminer: 3, not 4). |
| BandCenter | 1 | Band center estimate from electronegativity. |
| Mat2Vec | 200 | Composition-weighted pretrained embeddings from 3.3M papers. |

### ATTEMPTED BUT FAILED
| Featurizer | Issue | Resolution |
|-----------|-------|------------|
| `AtomicOrbitals` | Returns string values (`'s'`, `'p'`, `'d'`) for HOMO/LUMO character — can't convert to float | Replaced with manual HOMO/LUMO gap computation using NIST orbital energies |
| `OxidationStates` | `add_charges_from_oxi_state_guesses()` hangs for minutes per composition on complex materials | Removed. IonProperty already provides oxidation state info. |
| `ElectronegativityDiff` | Same oxi-state dependency — hangs indefinitely | Removed. Magpie electronegativity stats + BandCenter already cover this signal. |

### Feature Budget
| Version | Magpie | Extra | Mat2Vec | Total |
|---------|:------:|:-----:|:-------:|:-----:|
| V1 | 132 | ~136 (80 sparse) | 200 | 468 |
| **V2** | **132** | **22** (all active) | **200** | **354** |

---

## 3. Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Dataset** | `matbench_expt_gap` (4,604 samples, experimental band gap in eV) |
| **Featurizer** | `BandGapFeaturizer` (354d: 132 Magpie + 22 Extra + 200 Mat2Vec) |
| **CV Strategy** | 5-fold (KFold, shuffle=True, random_state=18012019) |
| **Architecture** | DeepHybridTRM (frozen from V1-82K config) |
| **d_attn** | 32 |
| **nhead** | 4 |
| **d_hidden** | 64 |
| **ff_dim** | 96 |
| **max_steps** | 16 |
| **dropout** | 0.15 |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max=200, eta_min=1e-4) → SWA (lr=5e-4) |
| **Epochs** | 300 (SWA starts at 200) |
| **Batch size** | 64 |
| **Parameters** | 75,457 |
| **Hardware** | 1× Tesla P100 (16 GB) |
| **Seed** | 42 |
| **Total time** | 36.1 min (featurization: ~2 min, training: 34 min) |

### Architecture Data Flow (unchanged from V1)
```
Magpie (132) → reshape [22, 6] → tok_proj → SA1 → SA2 → CrossAttn(Mat2Vec) → MeanPool
                                                                                    ↓
Extra (22)   → ──────────────── raw flat concat ────────────────────────────────→ [pool_in]
                                                                                    ↓
Mat2Vec (200) → m2v_proj → [1, 64] context token → CrossAttn K,V                pool → xp
                                                                                    ↓
                                                                              MLP Reasoning
                                                                              (16 steps, shared weights)
```

Pool layer: `Linear(32 + 22 = 54 → 64)` ← was `Linear(32 + 136 = 168 → 64)` in V1.

---

## 4. Results

### V2 (This Experiment)
| Config | Params | Test MAE (eV) | ±Std | Folds |
|--------|:------:|:-------------:|:----:|:-----:|
| **V2-82K** | 75,457 | **0.3342** | 0.0131 | 0.3182, 0.3402, 0.3418, 0.3514, 0.3195 |

### Comparison: V1 → V2 (Same Architecture, Different Features)
| Model | Params | Features | MAE (eV) | ±Std |
|-------|:------:|:--------:|:--------:|:----:|
| V1-82K (generic) | 82,753 | 468d | 0.3616 | 0.0103 |
| **V2-82K (band-gap)** | **75,457** | **354d** | **0.3342** | **0.0131** |
| **Δ** | −7,296 | −114d | **−0.0274 (−7.6%)** | |

### Also beats the larger V1 model:
| Model | Params | MAE | vs V2-82K |
|-------|:------:|:---:|:---------:|
| V1-218K | 218,541 | 0.3510 | V2 wins by −0.0168 (−4.8%) |

### Leaderboard Context
| # | Model | MAE (eV) |
|:-:|-------|:--------:|
| 1 | Darwin | 0.2865 |
| 2 | Ax/SAASBO CrabNet | 0.3310 |
| 3 | MODNet v0.1.12 | 0.3327 |
| **4** | **TRIADS V2-82K (single seed)** | **0.3342** |
| 5 | TRIADS V1-218K | 0.3510 |
| 6 | TRIADS V1-82K | 0.3616 |
| 7 | AMMExpress v2020 | 0.4161 |

---

## 5. Feature Diagnostic Report (from training run)

```
  Total features:  354
  Total samples:   4604
  Magpie:          132
  Extra:           22
  Mat2Vec:         200

  Extra featurizer breakdown:
    Stoichiometry            :   6 feats, active=6/6
    ValenceOrbital           :   8 feats, active=8/8
    IonProperty              :   3 feats, active=3/3
    BandCenter               :   1 feats, active=1/1
    TMetalFraction           :   1 feats, active=1/1
    HOMO/LUMO gap            :   3 feats, active=3/3

  ✓ No features with >10% NaN rate
  ✓ 0 zero-variance features in Extra
  ✓ All 22 extra features are ACTIVE (nonzero variance)

  Magpie: 8 features with >95% sparsity (f-orbital related, expected)
  Mat2Vec: 0 issues
```

---

## 6. Key Findings

### 6.1 Featurization > Architecture Size
The most important finding: **0.0274 eV improvement (−7.6%) came from changing ZERO model code**. Just swapping features. A 75K model with the right features beats a 218K model with generic features. This validates the "featurization-first" strategy.

### 6.2 Fewer Features, Better Performance
354 features outperformed 468 features. The 114 removed features (mostly ElementFraction zeros) were actively hurting the model by:
- Diluting the pool layer input (136 extras with ~80 zeros vs 22 all-active extras)
- Wasting pool layer parameters on noise dimensions
- Reducing signal-to-noise ratio in the flat injection pathway

### 6.3 HOMO/LUMO Gap is a Strong Signal
The manual HOMO/LUMO gap computation (3 features from NIST orbital energies) appears to be a strong contributor. This is the most physically direct band-gap predictor in the feature set — it estimates the electronic excitation energy from first-principles atomic data.

### 6.4 TMetalFraction Captures d-Electron Effects
Single scalar feature capturing the fraction of transition metals. d-electrons from TMs are the primary mechanism for narrowing/closing band gaps in materials. Simple but physically motivated.

### 6.5 Matminer Version Differences
Feature counts differed from documentation on Kaggle's matminer version:
- ValenceOrbital: 8 (not 32 as in older versions)
- IonProperty: 3 (not 4)
- Stoichiometry: 6 (not 5)

This is important for reproducibility — always check `feat.feature_labels()` on the actual runtime environment.

### 6.6 Individual Fold Performance
Fold 1 (0.3182) and Fold 5 (0.3195) individually approach MODNet (#3, 0.3327) and CrabNet (#2, 0.3310). This suggests that with multi-seed ensembling, V2 can reach the top 2.

---

## 7. Next Steps

### Immediate (V2 Extensions)
1. **5-Seed Ensemble**: Expected 3-5% reduction → ~0.32 eV → Leaderboard #2-3
2. **Scale to 218K** (d_attn=64): V2 features + larger model. V1-218K was 0.3510; with V2 features expect ~0.31-0.32 eV
3. **Multi-config ensemble**: Combine 82K + 218K predictions

### Featurization (V3 ideas)
4. Investigate if `CohesiveEnergy` is accessible without API key
5. Custom orbital-aware features beyond HOMO/LUMO gap
6. Per-element d-electron count as explicit feature (not just TMetalFraction)
7. Crystal field splitting energy estimates

### Architecture (V4 ideas)
8. Tokenize ValenceOrbital as separate attention tokens (s/p/d/f × 2)
9. Multi-token Mat2Vec (per-element embeddings instead of pooled)
10. Learned adaptive halting (ACT-style) for reasoning depth

---

## 8. Files

| File | Location | Description |
|------|----------|-------------|
| Training script | `expt_gap_v2.py` | Full V2 training code with BandGapFeaturizer |
| Model checkpoints | `expt_gap_models_v2/` | 5 fold checkpoints (V2-82K_s42_f{1-5}.pt) |
| Results JSON | `expt_gap_summary_v2.json` | Machine-readable results |
| V1 experiment log | `research/experiment_log_v1.md` | Previous version results |
| Research notes | `research/research_notes.md` | Featurization analysis |
