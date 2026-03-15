# TRM-MatSci V6 — Experiment Log
## Scaled Novel + Hybrid + Ensemble | matbench_steels | 2026-03-01

---

## 1. Setup

Same base as V4/V5 (332-dim combined features, 5-fold CV, 300 epochs, SWA@200, P100 GPU).

**Three approaches:**
- V6A: FeatureGroup-L (d=48, nhead=2) — scaled-up novel dual-reference TRM
- V6B: Hybrid-TRM — attention feature extractor → MLP-TRM reasoning
- V6C: MLP-SWA ×3 seeds — multi-seed ensemble, average predictions

---

## 2. Results

| Config | Params | Test MAE (MPa) | ±Std | Time |
|--------|:------:|:--------------:|:----:|:----:|
| V6C MLP-Ensemble ×3 | 66,889 ×3 | 129.04 | **8.99** | 11.1 min |
| V6B Hybrid-TRM | 67,305 | 134.97 | 23.15 | 4.4 min |
| V6A FeatGroup-L | 80,929 | 153.96 | **8.84** | 18.4 min |

---

## 3. Key Findings

### F1: Multi-Seed Ensemble Halves Variance, Same Mean
V6C (129.04 ± 8.99) ≈ V5A (128.98 ± 17.42) in mean, but std dropped from 17.4 → 9.0. The 3 seeds make similar errors — not diverse enough to reduce the mean. More diversity needed (different architectures, not just seeds).

### F2: Hybrid-TRM Has Best Val (84.1 MPa) But Worst Gap
Val reaches MODNet level (84.1 < 87.8) but test is 134.97. The one-shot attention overfits to val patterns more than recursive attention. Gap: ~50 MPa — worst of any model.

### F3: FeatureGroup-L Most Consistent (±8.84)
Lowest std across all versions. Recursive dual-reference attention provides very stable predictions. Architecture improved: 165→154 by scaling d=32→48.

### F4: Novel Architecture Progression
| Version | d_model | Params | MAE | ±Std | Improvement |
|---------|:-------:|:------:|:---:|:----:|:-----------:|
| V2 Trans-Novel | 64 | ~100K | 388 | — | Baseline (broken) |
| V5B FeatGroup | 32 | 38K | 165.1 | 17.6 | +223 MPa (property tokens) |
| V6A FeatGroup-L | 48 | 81K | 154.0 | 8.8 | +11 MPa (scale up) |

### F5: Val-Test Gap Is THE Bottleneck
All models reach CrabNet/MODNet-level validation but test MAE stays 128-154. The 37-sample val set cannot reliably select generalizing checkpoints.

---

## 4. Complete Results — All Versions

| Version | Best Model | MAE | Params | Gap to Darwin |
|---------|-----------|:---:|:------:|:-------------:|
| V1 | MLP-TRM-h64 | 184.4 | 100K | 61.1 |
| V2 | MLP-L | 184.6 | 115K | 61.3 |
| V3 | MLP-Magpie-L | 130.3 | 248K | 7.0 |
| V4 | MLP-Combined-S | 131.6 | 67K | 8.3 |
| **V5A** | **MLP-SWA** | **129.0** | **67K** | **5.7** |
| V6C | MLP-Ensemble ×3 | 129.0 | 67K×3 | 5.7 |
| — | Darwin (target) | 123.3 | — | 0 |
| — | CrabNet | 107.3 | — | — |
