# TRM-MatSci Experiment Log: V13

**Date:** March 2026
**Objective:** Break the 95.99 MPa barrier and approach MODNet (87.76 MPa) by introducing higher-order property interactions and variance reduction techniques.
**Strategies Implemented:**
1.  **Architecture:** Added a second Self-Attention layer (`d_attn=64`) to the feature extractor (`DeepHybridTRM` and `DeepConfidenceHybridTRM`). This enables the model to learn 2nd-order compositions of material properties.
2.  **Variance Reduction:** 5-Seed Ensemble averaging (seeds 42, 123, 7, 0, 99). Multi-seed ensembling was previously rejected in V6 because models were too weak/made the same errors. With V12/V13-level models, the remaining error is largely variance driven.

**Baseline (V12A):** 95.99 MPa

---

## Configurations Tested

### V13A: 2-Layer SA + Standard Deep Supervision
*   **Architecture:** `DeepHybridTRM`
*   **Features:** Expanded (`matminer` + `mat2vec`)
*   **Scale:** `d_attn=64`, 2 SA layers, 1 CA layer
*   **Recursion Steps:** 20 steps
*   **Ensemble:** 5 random seeds averaged per sample

### V13B: 2-Layer SA + Confidence-Weighted DS (Running)
*   **Architecture:** `DeepConfidenceHybridTRM`
*   **Recursion Steps:** 22 steps
*   **Ensemble:** 5 random seeds averaged per sample

---

## Results: V13A (Standard DS) vs V13B (Confidence DS)

**Final Leaderboard (5-Seed Ensembles)**

| Configuration | Parameters | MAE (MPa) | Status |
| :--- | :---: | :---: | :--- |
| **V13A-2xSA-StdDS** | 224,685 | **91.20 ± 12.23** | 🏆 **NEW PROJECT SOTA** |
| V13B-2xSA-ConfDS | 229,390 | 93.04 ± 13.01 | Beats V12A, but worse than StdDS |
| V13A (Best Seed - 123) | | 96.77 | |
| V13B (Best Seed - 123) | | 97.08 | |

### Per-Fold Ensemble Breakdown
| Fold | V13A (StdDS) | V13B (ConfDS) |
| :---: | :---: | :---: |
| **1** | 114.32 | 106.48 (Wins) |
| **2** | 81.46 | 77.04 (Wins) |
| **3** | 80.55 | 77.80 (Wins) |
| **4** | 90.49 (Wins) | 105.08 |
| **5** | 89.18 (Wins) | 98.79 |

---

## Key Findings & Takeaways

### 1. The Power of Variance Reduction
The gap between the best single seed (96.77) and the ensemble average (91.20) is a massive **5.57 MPa**. This conclusively proves that at this level of performance, individual models are highly sensitive to weight initialization and batch ordering (variance), but their systematic bias is low. Averaging them destroys the noise.

### 2. Standard DS > Confidence DS (Again)
Just like in V12, standard linear-weighted Deep Supervision outperforms the Confidence-weighted approach overall (91.20 vs 93.04). However, the fold breakdown is fascinating: **V13B won on Folds 1, 2, and 3, but bombed Folds 4 and 5**. Confidence DS seems to create higher variance across folds, whereas Standard DS is a much safer, more consistent regularizer.

### 3. Approaching the Final Boss
With 91.20 MPa, we are now just **3.44 MPa away from MODNet**. V14 will pivot from scaling the architecture to massively expanding the feature set (adding alloy-specific properties and tokenizing all inputs for a 2-pass TRM attention loop) to feed this powerful architecture more signal.
