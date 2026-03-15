# TRM-MatSci Experiment Log: V14

**Date:** March 2026
**Objective:** Having plateaued on architectural scaling with V13 (91.20 MPa ensemble, but 96.77 MPa single seed limit), V14 tests a massive feature expansion. We pivot from scaling the architecture to enriching the input signal, providing the model with domain-specific metallurgical properties.

**Strategies Implemented:**
1.  **Mega-Featurizer:** Expanded input from ~462d (Magpie + Mat2Vec) to ~670d by adding `DEML` (defect properties), `WenAlloys` (radii, shear modulus), `Miedema` (enthalpy), `YangSolidSolution` (omega parameter), and `TMetalFraction` (transition metals).
2.  **Flat vs. Tokenized Processing:**
    *   **V14A (Control):** All features concatenated into a flat vector. Only Magpie tokens go through attention (same as V13).
    *   **V14B (Tokenized):** All structured properties (Magpie, DEML, Alloy) are treated as distinct 64d tokens attending to each other via a shared-weight 2-pass Self-Attention loop (a TRM loop *inside* the extractor).

**Baseline (V13A Best Single Seed):** 96.77 MPa
*(Note: We compare single-seed to single-seed. V13A's 91.20 was a 5-seed ensemble).*

---

## Results: V14 (Mega Features, Single Seed)

**Final Leaderboard (Single Seeds)**

| Configuration | Architecture | Parameters | MAE (MPa) | Status |
| :--- | :--- | :---: | :---: | :--- |
| **V14A-Flat** | 2-Layer SA (Flat Extras) | 238,509 | **94.94 ± 14.21** | 🏆 **NEW SINGLE-SEED SOTA** (Beats V12A) |
| V14B-Tokenized | 2-Pass SA TRM Loop | 195,917 | 96.15 ± 12.14 | Beats V11B |

### Per-Fold Breakdown
| Fold | V14A-Flat | V14B-Tokenized | V13A (Baseline) |
| :---: | :---: | :---: | :---: |
| **1** | 122.25 | **113.30** | > 120.0 |
| **2** | 82.77 | 86.51 | ~81.0 |
| **3** | 85.37 | **81.56** | ~90.0 |
| **4** | 94.27 | 107.18 | ~97.0 |
| **5** | 90.04 | 92.20 | ~92.0 |

---

## Key Findings & Takeaways

### 1. The Mega-Features Work (Single Seed SOTA)
V14A-Flat achieved **94.94 MPa** with a single seed computationally equivalent to V13. This beats the best single seed from V13A (96.77) and beats the V12A SOTA (95.99). The infusion of alloy-specific features (shear modulus, solid solution parameters, Miedema enthalpy) successfully provided the MLP with the exact domain context it was missing.

If we ensemble V14A across 5 seeds (which typically yields a ~5.5 MPa reduction), it projects to **~89 MPa**, which would threaten MODNet (87.76).

### 2. Tokenization is Powerful but Noisy
V14B (Tokenized TRM Attention) was highly volatile. It won Folds 1 and 3 by massive margins (113.30 vs 122.25, 81.56 vs 85.37), but failed catastrophically on Fold 4 (107.18 vs 94.27). With 58 distinct tokens attending to each other, but only 312 training samples, the attention matrix is likely too sparse and prone to overfitting on specific compositional quirks.

### 3. The Structural Blindspot
Both models highlight a fundamental limitation of our current TRM design: Attention runs *once* to extract features, and then the MLP reasons blindly for 20 steps. The MLP cannot "ask for help" or re-evaluate the raw features based on its current hypothesis. 

This realization directly motivates **V15 (HTRM)**, which will implement a Hierarchical Reasoning Model architecture where the attention mechanism re-evaluates the features every 5 steps, conditioned heavily on the MLP's current prediction state.
