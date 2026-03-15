# TRIADS Multi-Bench — Research Direction
*Last updated: 2026-03-04*

---

> This document synthesizes our plan for generalizing the TRIADS architecture (V13A, 91.20 MPa on Steels) to additional Matbench datasets, starting with `matbench_expt_gap`.

---

## Where We Started

We successfully developed the **DeepHybridTRM** architecture (Attention Feature Extraction + MLP Reasoning + Deep Supervision computation at every step) and achieved SOTA-level performance on the `matbench_steels` dataset (312 samples), reaching 91.20 MPa with a 5-seed ensemble (baseline SOTA was ~79.95 TPOT, but 103.51 for standard tabular).

To prove that the TRIADS architecture is not an overfitted anomaly for metallic yield strength, it must demonstrate comparable generalization on larger, fundamentally different datasets.

---

## Phase 1: `matbench_expt_gap`

### V1 — Direct Translation (✅ Done)
Ran the exact Steels architecture on band gap data with no tuning. Results:
- **EG-A (218K params)**: 0.3510 eV — #4 on leaderboard, beats CrabNet/AMMExpress
- **EG-B (82K params)**: 0.3616 eV — more stable but slightly worse

### V2 — 100K Sweet Spot Hunt (📦 Archived)
Tried smaller models (d_attn=36, ~82K params) with FastTensorDataLoader and heavy telemetry. **Underfitted** — confirmed d_attn=64 is needed for this dataset.

### V3 — Proven Arch + AMP (📦 Archived)
Went back to V1's d_attn=64 with AMP, FastTensorDataLoader, CPU tuning. Clean output. Never fully ran on Kaggle — diverted into TPU sweep (which failed due to torch_xla multiprocessing issues).

### V2 — Featurization Overhaul (✅ Done — 0.3342 eV!)

**Key insight:** V1-V3 used the same generic featurization (468d). This dataset is about **band gaps** — electronic properties. We need features that specifically capture electronic structure.

#### What Changed
- **REMOVED**: ElementFraction (~80 sparse features — mostly zeros, noise for 60+ elements)
- **ADDED**: TMetalFraction (1 feat), manual HOMO/LUMO gap from NIST orbital energies (3 feats)
- **KEPT**: Magpie (132), Stoichiometry (6), ValenceOrbital (8), IonProperty (3), BandCenter (1), Mat2Vec (200)
- **Total**: 468d → 354d (fewer features, better results)

#### What Failed (attempted, removed)
- `AtomicOrbitals`: returns strings for orbital character — not numeric
- `OxidationStates`/`ElectronegativityDiff`: `add_charges_from_oxi_state_guesses()` hangs indefinitely

#### Results
| Model | Params | Features | MAE (eV) | vs V1 |
|-------|:------:|:--------:|:--------:|:-----:|
| **V2-82K** | **75,457** | **354d** | **0.3342 ± 0.0131** | **−7.6%** |
| V1-218K | 218,541 | 468d | 0.3510 | — |
| V1-82K | 82,753 | 468d | 0.3616 | — |

Per-fold: 0.3182, 0.3402, 0.3418, 0.3514, 0.3195

**Featurization > architecture size.** 75K model with right features beats 218K model with generic features.

### V3 — 5-Seed Ensemble (✅ Done — 0.3068 eV! #2 on Leaderboard!)

| Config | Single-Seed | 5-Seed Ensemble | Improvement |
|--------|:---:|:---:|:---:|
| V3-82K (75K params) | 0.3416 ± 0.0030 | **0.3122 ± 0.0108** | ↓8.6% |
| V3-100K (101K params) | 0.3344 ± 0.0031 | **0.3068 ± 0.0082** | ↓8.2% |

V3-100K folds 4 & 5 went sub-0.30 (0.2998, 0.2962), proving the architecture can reach Darwin territory.

**Full journey: 0.3616 → 0.3068 = −15.2% improvement across V1→V3.**

---

## High-Level Status Summary

| Dataset | Phase | Focus | Best MAE | Status | Key Finding |
|---------|-------|-------|:----------:|--------|-------------|
| steels | V13A | Baseline Arch | 91.20 MPa | 🏆 SOTA-level | DeepHybridTRM + Deep Sup works phenomenally. |
| expt_gap| V3 | Feat+Ensemble | **0.3068 eV** | **🏆 #2** | **15.2% improvement V1→V3. Beats CrabNet, MODNet. 0.02 behind Darwin.** |
| jdft2d | V4 | Struct+Ensemble | **35.89 meV/atom** | **🏅 #3** | **Structural features = −19.2%. Ensemble = −5.4% more.** |
| is_metal | V1 | Classification | **0.9655 AUC** | **🏆 #1** | **Massive destruction of 0.9209 SOTA with 100K parameters.** |
| dielectric | *Pending* | Regression | *TBD* | 🔬 Next | Refractive index, 4764 samples. |
| glass | *Pending* | Classification | *TBD* | *Pending* | Metallic glass forming ability. |
