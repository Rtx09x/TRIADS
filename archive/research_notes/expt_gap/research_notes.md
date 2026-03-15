# Featurization Research Notes — matbench_expt_gap

## Problem
Band gap is fundamentally an **electronic** property. Our V1-V3 features were generic composition descriptors — they don't specifically encode the physics that determines band gaps.

## V1 Feature Pipeline (468 features) — SUPERSEDED by V2

### Magpie (132 features)
22 element properties × 6 statistics (mean, std, min, max, range, mode):
- MendeleevNumber, AtomicWeight, MeltingTemp, Column, Row, CovalentRadius, Electronegativity
- NsValence, NpValence, NdValence, NfValence, NValence, NsUnfilled, NpUnfilled, NdUnfilled, NfUnfilled, NUnfilled
- GSVolume_pa, GSBandGap, GSMagMom, SpaceGroupNumber

> **Note:** GSBandGap is legitimate signal — DFT-computed per-element band gap, not per-compound leakage.

### Extra Matminer (~136 features)
- **ElementFraction** (~60-80): REMOVED in V2 — too sparse, mostly zeros for 60+ element dataset
- **Stoichiometry** (5): KEPT
- **ValenceOrbital** (32): KEPT
- **IonProperty** (4): KEPT
- **BandCenter** (1): KEPT

### Mat2Vec (200 features)
- Composition-weighted pool of pre-trained element embeddings
- Captures latent element relationships learned from 3.3M abstracts

---

## V2 Feature Pipeline (354 features) — CURRENT

### What Changed
| Action | Featurizer | Features | Why |
|--------|-----------|:--------:|-----|
| **REMOVED** | ElementFraction | ~80 | Sparse/noisy. Mat2Vec covers element identity better. |
| **ADDED** | TMetalFraction | 1 | d-electrons narrow gaps. Single scalar = high signal. |
| **ADDED** | HOMO/LUMO gap (manual) | 3 | HOMO energy, LUMO energy, gap from NIST orbital energies. Most direct band-gap predictor. |
| **KEPT** | Magpie | 132 | Core properties. |
| **KEPT** | Stoichiometry | 6* | Composition complexity. |
| **KEPT** | ValenceOrbital | 8* | Orbital electron stats. |
| **KEPT** | IonProperty | 3* | Ionic character. |
| **KEPT** | BandCenter | 1 | Band center estimate. |
| **KEPT** | Mat2Vec | 200 | Latent chemistry. |

*Feature counts differ on Kaggle matminer vs docs (6/8/3 instead of 5/32/4).

### What Failed (attempted but removed)
| Featurizer | Problem |
|-----------|---------|
| `AtomicOrbitals` | Returns strings (`'s'`, `'p'`, `'d'`) for orbital character — not numeric. Replaced with manual HOMO/LUMO computation. |
| `OxidationStates` | `add_charges_from_oxi_state_guesses()` hangs indefinitely on complex compositions. IonProperty already covers this. |
| `ElectronegativityDiff` | Same oxi-state hang. Magpie electronegativity stats + BandCenter cover this signal. |

### V2 Results: **0.3342 ± 0.0131 eV** (75K params)
- **−7.6%** vs V1-82K (0.3616 eV, same architecture)
- **−4.8%** vs V1-218K (0.3510 eV, 3× bigger model)
- **Leaderboard #4** single-seed, approaching MODNet (#3, 0.3327)
- Per-fold: 0.3182, 0.3402, 0.3418, 0.3514, 0.3195

---

## Key Insights

### 1. Featurization > Architecture Size
0.0274 eV improvement from ZERO model code changes. Fewer features (354 vs 468), fewer parameters (75K vs 82K), better performance. The feature quality matters more than quantity.

### 2. ElementFraction Was Actively Hurting Performance
~80 sparse features diluting the pool layer input. Removing them gave the model a cleaner signal path: pool layer went from Linear(168→64) to Linear(54→64).

### 3. HOMO/LUMO Gap is Powerful
Manual computation from NIST orbital energies gives the model a direct electronic structure predictor. This is the most physics-informed feature in the set.

### 4. Matminer Featurizer Gotchas
- `AtomicOrbitals`: non-numeric output (strings) — unusable without encoding
- `OxidationStates`/`ElectronegativityDiff`: hang on oxi-state guessing
- Feature counts vary across matminer versions — always verify at runtime

---

## Next Experiments

### Immediate
- [x] V2 featurization test (0.3342 eV — success!)
- [x] 5-seed ensemble → **V3-82K: 0.3122 eV, V3-100K: 0.3068 eV (#2 on leaderboard!)**
- [x] ~~Scale to 218K~~ — Not needed. 100K sweet spot found.

### ✅ matbench_expt_gap — COMPLETE (0.3068 eV, #2)

### Future Featurization Ideas
- [ ] Custom orbital features beyond HOMO/LUMO gap
- [ ] Per-element d-electron count (not just TM fraction)
- [ ] Crystal field splitting energy estimates
- [ ] CohesiveEnergy (if accessible without API key)
- [ ] Tokenize ValenceOrbital as s/p/d/f attention tokens
- [ ] Multi-token Mat2Vec (per-element embeddings)
