# TRM-MatSci v1 — Experiment Log | matbench_expt_gap
## Initial Multi-Bench Generalization Test | 2026-03-04

---

## 1. Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Dataset** | `matbench_expt_gap` (4,604 samples, experimental band gap in eV) |
| **CV Strategy** | 5-fold nested cross-validation (KFold, shuffle=True, random_state=18012019) |
| **Input** | Expanded Featurizer: Magpie (132) + Extra matminer (~60-80) + Mat2Vec (200) |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max=200, eta_min=1e-4) |
| **Epochs** | 300 per fold (SWA starts at 200) |
| **Batch size** | 64 |
| **Dropout** | 0.15 |
| **Gradient clipping** | max_norm=1.0 |
| **Hardware** | GPU-Optimized (num_workers=2, pin_memory=True, persistent_workers=True) |
| **Data split** | 85% train / 15% val for early stopping/SWA within the outer fold |
| **Seed** | 42 |

### Architecture Variants (DeepHybridTRM with Deep Supervision)
- **EG-A-SameSize (V13A Equivalent)**: `d_attn=64, nhead=4, d_hidden=96, ff_dim=150, max_steps=20`. ~224K parameters. Tests if the winning architecture for Steels works out-of-the-box on a 15x larger dataset.
- **EG-B-Smaller**: `d_attn=32, nhead=4, d_hidden=64, ff_dim=96, max_steps=16`. ~50-60K parameters. Tests if the massive increase in samples makes a smaller model viable, or if the wider element distribution (60+ elements vs 15) demands higher capacity.

---

## 2. Results Summary (Test MAE — 5-Fold Average)

### TRIADS Models
| Config | Actual Params | Target MAE | Test MAE (eV) | ±Std | Time |
|--------|:-------------:|:----------:|:-------------:|:----:|:----:|
| **EG-A-SameSize** | 218,541 | <0.2865 | **0.3510** | 0.0285 | ~65m |
| **EG-B-Smaller** | 82,753 | <0.2865 | **0.3616** | 0.0103 | ~25m |

### Leaderboard Baselines (Composition-Only)
| Model | MAE (eV) | Type |
|-------|:--------:|:----:|
| **Darwin** | **0.2865** | Evolutionary Algorithm (SOTA) |
| Ax/SAASBO CrabNet | 0.3310 | Transformer / AutoML |
| MODNet v0.1.12 | 0.3327 | Neural Network |
| **TRIADS (EG-A, un-tuned)**| **0.3510**| **DeepHybridTRM** |
| AMMExpress v2020 | 0.4161 | AutoML |
| CrabNet | 0.4427 | Transformer |
| RF-SCM/Magpie | 0.5205 | Random Forest |
| Dummy (mean prediction)| 1.0280 | Baseline |

---

## 3. Key Observations & Patterns

### 3.1 Architectural Generalization is Real
Without a single hyperparameter optimization for this specific dataset aside from fixing batch size to 64, the exact same DeepHybridTRM architecture that achieved 91.20 MPa on `matbench_steels` (N=312) dropped straight into `matbench_expt_gap` (N=4,604) and immediately posted **0.3510 eV**, placing it #4 on the official leaderboard. It decisively crushed the standard CrabNet (0.4427) and AMMExpress (0.4161) baselines.

### 3.2 Bigger Model Wins on MAE, Smaller Wins on Stability
Surprisingly, the **larger** EG-A (218K params) beat EG-B (82K params) on absolute MAE: 0.3510 vs 0.3616. This contradicts the val-phase observations where EG-B showed better `Best=` values. However, EG-B has **dramatically lower variance** (±0.0103 vs ±0.0285), meaning SWA smoothed it into a more consistent but slightly higher predictor. This suggests that for a multi-seed ensemble, EG-B's stability could compound into a better ensemble mean.

**Per-fold comparison:**
| Fold | EG-A | EG-B |
|:----:|:----:|:----:|
| 1 | 0.3417 | 0.3444 |
| 2 | 0.3480 | 0.3632 |
| 3 | **0.4058** | 0.3575 |
| 4 | **0.3249** | 0.3684 |
| 5 | 0.3345 | 0.3743 |

EG-A has higher peaks (Fold 4: 0.3249!) but also a catastrophic outlier (Fold 3: 0.4058). EG-B never spikes above 0.3743.

### 3.3 The Next Bottleneck: GPU Starvation
Monitoring via Kaggle revealed that training was severely bottlenecked by the CPU DataLoader. CPU usage pinned at 109% while GPU usage hovered at 23%. Because the dataset is tabular and only ~10MB total, feeding it batch-by-batch over PCIe is vastly inefficient compared to the GPU's compute speed. Total time: 89.7 minutes for both models across 5 folds.

---

## 4. Planned V2 Experiments
**V2 Goal:** Eliminate DataLoader overhead and tune capacity.

1. **Full VRAM Loading**: Implement `FastTensorDataLoader` to pre-load the entire 4604-sample dataset directly into the P100's 16GB VRAM before the fold loop starts. This will drop CPU usage to 0% and allow the GPU to execute steps in microseconds.
2. **EG-B Execution**: Run the target `~50K` parameter model now that training is fast.
3. **Hyperparameter Scans**: With fast training, we can scan `dropout` (0.1 vs 0.2) and `d_attn` widths. The initial 0.3510 eV result proves the representation works; now it just needs tuning.
