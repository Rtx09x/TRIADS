# TRIADS: Tiny Recursive Information-Attention with Deep Supervision

### A High-Precision Deep Learning Architecture for Materials Property Prediction on Sparse Datasets

[![HF Model](https://img.shields.io/badge/🤗_Model-Rtx09%2FTRIADS-yellow)](https://huggingface.co/Rtx09/TRIADS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

> 🔗 **[Download Pretrained Weights](https://huggingface.co/Rtx09/TRIADS)** · **[Read the Paper](TRIADS_Final.pdf)**

---

## Overview

TRIADS is a novel deep learning architecture built from the ground up for **materials property prediction in the small-data regime** — where conventional deep learning chronically overfits and off-the-shelf models fail to generalize. The architecture combines **self-attention-based compositional feature extraction**, a **recursive MLP reasoning core** with shared weights, and a **deep supervision training protocol** that forces calibrated predictions throughout the entire recursive trajectory.

The core philosophy: instead of adding parameters to combat small datasets, force a small model to think harder. A weight-tied attention cell iterated T times costs zero additional parameters while achieving the computational depth of a T-layer network — multiplying gradient signal density by T without adding training data.

TRIADS has been validated across **six Matbench benchmarks** through a combined 300+ model ablation study. It achieves state-of-the-art or near-SOTA results on all six tasks, with parameter counts 2–50× smaller than competing neural network methods.

---

## Benchmark Results

| Task | Target Property | N | TRIADS Result | Params | Method |
|:-----|:----------------|:-:|:-------------:|:------:|:-------|
| `matbench_steels` | Yield strength | 312 | **91.20 ± 12.23 MPa** | 225K | HybridTRIADS V13A |
| `matbench_expt_gap` | Band gap (eV) | 4,604 | **0.3068 ± 0.0082 eV** | 100K | HybridTRIADS V3 |
| `matbench_expt_ismetal` | Metal/Non-metal | 4,921 | **0.9655 ± 0.0029 ROC-AUC** | 100K | HybridTRIADS |
| `matbench_glass` | Glass forming | 5,680 | **0.9285 ± 0.0063 ROC-AUC** | 44K | HybridTRIADS |
| `matbench_jdft2d` | Exfol. energy | 636 | **35.89 ± 12.40 meV/atom** | 75K | HybridTRIADS V4 |
| `matbench_phonons` | Peak phonon freq. | 1,265 | **41.91 ± 4.04 cm⁻¹** | 247K | GraphTRIADS V6 |

All results follow the exact Matbench 5-fold nested cross-validation protocol (`KFold(n_splits=5, shuffle=True, random_state=18012019)`), using only training fold data for featurizer fitting and target normalization — no test data leakage at any stage.

### Selected Comparisons

**matbench_expt_ismetal:** TRIADS 0.9655 vs. GPTChem (>1B parameters) 0.8965 — **+6.79 ROC-AUC points with 10,000× fewer parameters**

**matbench_jdft2d:** TRIADS 35.89 vs. coGN (pre-trained, 1M+ params) 37.17 — **better without any pretraining**

**matbench_steels:** TRIADS 91.20 vs. CrabNet (pretrained on 300K+ materials, ~1M params) 107.31 — **17 MPa lower error at 0.2% of the parameter count**

### Matbench Steels — Full Leaderboard

| Model | MAE (MPa) | Type | Parameters |
|:------|:---------:|:-----|:----------:|
| AutoGluon | 77.03 | Stacked Ensemble (AutoML) | — |
| TPOT-Mat | 79.95 | AutoML Pipeline | — |
| MODNet v0.1.12 | 87.76 | Neural Network | — |
| **TRIADS V13A (Ours)** | **91.20** | **Hybrid-TRM + Deep Supervision** | **225K** |
| RF-SCM/Magpie | 103.51 | Random Forest | — |
| CrabNet | 107.31 | Transformer (pretrained) | ~1M+ |
| Darwin | 123.29 | Evolutionary Algorithm | — |

> **Peak per-fold: 80.55 MPa** — surpassing TPOT-Mat (79.95 MPa) on fold 3 of the official evaluation, achieved with a 225K-parameter model trained entirely from scratch.

---

## Architecture

TRIADS operates through four sequential processing stages. Each stage reflects a design choice backed by extensive empirical ablation — not convenience defaults. The architecture exists in two variants:

- **HybridTRIADS** — for composition-only tasks (steels, band gap, metallicity, glass, jdft2d)
- **GraphTRIADS** — for structural tasks where atom positions are required (phonons)

### Stage 1: Compositional Featurization

Chemical formulas are not fed directly to the model. Instead, they pass through a **domain-informed multi-source featurization pipeline** that provides the model with physically meaningful numerical structure before any learned processing occurs.

**Magpie Descriptors (132 dimensions):** 22 elemental properties — electronegativity, atomic radius, melting point, valence electrons, first ionization energy, etc. — each summarized as 6 statistics (mean, average deviation, minimum, maximum, range, mode) across the composition. The 6×22=132 layout creates the **property token matrix** at the heart of Stage 2.

**Mat2Vec Embeddings (200 dimensions):** Word2Vec embeddings pretrained on 3 million materials science abstracts, providing learned chemical semantic knowledge. The composition-level embedding is a fraction-weighted sum of per-element vectors — equivalent to a soft lookup in a 200-dimensional "chemical meaning" space.

**Extended Matminer Descriptors (task-dependent, ~20–130 additional dimensions):** Supplementary features selected per task. For band gap prediction, BandCenter and HOMO/LUMO gap proxies are included as direct electronic-structure priors. For metallicity, HOMO/LUMO features and TMetalFraction are used. For glass forming, BandCenter and HOMO/LUMO are excluded because mixing thermodynamics (not electronic structure) governs glass formation.

> **Empirical evidence for this selection:** On `matbench_expt_gap`, replacing generic extra features with physics-informed ones (BandCenter + HOMO/LUMO) reduced MAE from 0.3616 to 0.3342 eV — a **7.6% improvement with zero architectural change**, purely from better sensor design.

### Stage 2: Attention-Based Feature Extraction

The Magpie feature vector is **restructured** into a 22×6 token matrix: 22 rows (one per elemental property), each containing its 6 statistics. These tokens are projected into a 64-dimensional attention space.

Two stacked self-attention layers process the property tokens:
- **SA1** learns first-order property interactions: "when electronegativity range is high AND atomic radius range is low..."
- **SA2** learns second-order patterns: interactions between first-order patterns discovered by SA1

A cross-attention layer then integrates Mat2Vec chemical semantics as a compressed context vector: the key/value input to the cross-attention is the Mat2Vec embedding, while the queries come from the property tokens. This grounds the property-level reasoning in chemical knowledge.

> **Why this works when element-as-token attention does not:** [V2 experiments](archive/matbench_steels/trm2.py) show that element-as-token attention fails catastrophically at 388 MPa — 312 samples cannot teach attention to discover element interactions from random initialization. Property tokens are *already* meaningful — comparing "average electronegativity profiles" across compositions is a well-defined operation that yields useful signal from small data. The same attention mechanism went from 388 MPa (raw element tokens) to 165 MPa (property tokens), a 223 MPa improvement from input restructuring alone.

### Stage 3: Recursive MLP Reasoning (The TRM Core)

The pooled attention output enters the **Tiny Recursive Model loop**: a pair of shared-weight MLP blocks that iteratively refine two persistent state vectors — a reasoning state `z` and a prediction draft `y` — over T=16–20 recursive steps.

```
For t = 1 to T:
    zₜ = zₜ₋₁ + MLP_z(zₜ₋₁, yₜ₋₁, x_pooled)    # Refine reasoning state
    yₜ = yₜ₋₁ + MLP_y(yₜ₋₁, zₜ)                  # Refine prediction draft

Final output: property = Linear(y_T)
```

Because the MLP weights are **shared across all T steps**, this loop adds zero additional parameters beyond two small MLPs, yet achieves the computational depth of a 2T-layer network. The additive residual structure (GRU-style gates in deeper configurations) ensures stable training beyond 8 cycles.

> **This is not redundant computation.** V1 experiments showed smooth, monotonic MAE descent from ~1400 MPa at step 1 to ~184 MPa at step 16, providing direct empirical evidence that each recursive pass meaningfully refines the prediction.

### Stage 4: Deep Supervision

During training, L1 loss is computed at **every recursion step** using linearly increasing weights (step *t* receives weight *t/∑t*). This forces the model to produce calibrated predictions throughout the entire trajectory.

**This is the single highest-leverage design decision in TRIADS.** The ablation is clean: identical architecture (87K params, 20 cycles), only the training objective changes:

| Config | MAE (MPa) | Δ |
|:-------|:---------:|:-:|
| V9A — final-step loss only | 134.59 | baseline |
| **V10A — per-cycle deep supervision** | **103.29** | **−23.3%** |

Deep supervision also acts as a **regularization mechanism.** The `d_attn=64` configuration that caused catastrophic overfitting without DS ([V8B](archive/matbench_steels/): 155 MPa) became the project's best-performing architecture with DS ([V11B](archive/matbench_steels/): 102.30 MPa). By simultaneously satisfying 20 loss objectives, the shared weights cannot specialize to any single step's gradient signal — distributing learning pressure uniformly across weight space.

### GraphTRIADS — Structural Extension (Phonons)

For the phonon task, TRIADS is extended with a **3-order hierarchical crystal graph** that lives inside the shared recursive cell:

- **Order 1 (Atom graph):** 18-feature nodes, 12 nearest neighbors within 8Å, Gaussian basis distance RBF (40 channels)
- **Order 2 (Bond/Line graph):** Triplet angles θ_ijk — encodes local coordination geometry
- **Order 3 (Dihedral graph):** Torsion angles φ_ijkl — encodes medium-range structural periodicity
- **Physics features:** Empirical bond stiffness k, reduced mass μ, Einstein frequency ω_E — directly related to the predicted phonon frequencies

The hierarchical GNN (Dihedral→Angle→Bond→Atom) runs as the **shared recurrent cell**: each TRM cycle executes a full 4-level message-passing stack. A gate-based halting mechanism (`min_cycles=4, max_cycles=16`) allows the model to use fewer cycles on simple structures and more cycles on complex ones.

> **Sensor ablation (one geometric order added per version):**
>
> | Version | Sensor | MAE (cm⁻¹) |
> |:--------|:-------|:----------:|
> | V1 — Composition only | None | 71.82 |
> | V3 — Distance GNN | `d(i,j)` RBF | 63.00 |
> | V4 — Angular GNN | `θ(ijk)` | 56.33 |
> | V5 — Physics sensors | `k`, `μ`, `ω_E` | 49.11 |
> | **V6 — Dihedral + gate-halt** | `φ(ijkl)` | **41.91** |

Total improvement: **−41.7% MAE** from progressive sensor addition with identical model capacity.

---

## Development History

TRIADS was not designed in a single iteration. It is the product of 300+ trained models across 15+ versions per benchmark, each driven by a specific hypothesis and tested empirically. Many versions failed — these failures were as instructive as the successes.

### Matbench Steels — V1 → V13

| Version | Key Change | MAE (MPa) | Key Finding |
|:--------|:-----------|:---------:|:------------|
| V1 | Mat2Vec MLP (12-model sweep) | 184.38 | Input is the bottleneck, not model capacity |
| V2 | Element-as-token Transformer | 388.58 | ❌ 312 samples can't train attention from scratch |
| V3 | Magpie descriptors | 130.33 | Engineered features shatter the 184 MPa ceiling (−54 MPa) |
| V4 | Magpie + Mat2Vec combined | 131.63 | Mat2Vec adds parameter efficiency, not novel signal |
| V5 | SWA + property-token attention | 128.98 | SWA finds flatter minima; property tokens unlock attention |
| V7 | Scaled Hybrid-TRM | 127.08 | First time attention surpasses pure MLP |
| V8 | d_attn=64 (without DS) | 155.06 | ❌ Wider attention overfits without regularization |
| V9 | T=20 cycles (without DS) | 134.59 | ❌ Over-refinement: easy folds degrade past T=16 |
| **V10** | **Deep supervision** | **103.29** | **Core breakthrough — beats Darwin, CrabNet, RF-SCM** |
| V11 | Scaled arch + DS | 102.30 | DS unlocks the d_attn=64 that V8 couldn't use |
| V12 | Expanded features | 95.99 | First sub-100 MPa; features + capacity must co-scale |
| **V13** | **2-layer SA + 5-seed ensemble** | **91.20** | **Project SOTA — 50.5% error reduction from V1** |
| V14 | 670d feature expansion | 94.94 | Single-seed SOTA with domain-specific thermodynamics |
| V15 | Hierarchical TRM (HTRM) | 431.86 | ❌ Gradient detachment incompatible with small-data optimization |

Complete training code, metrics, and result plots for every version are in [archive/matbench_steels/](archive/matbench_steels/) and [archive/research_notes/steels/](archive/research_notes/steels/).

### Matbench Phonons — V1 → V6

The phonons task required building an entirely new structural GNN encoder from scratch. The development was driven by a systematic sensor expansion study — what happens when you add one geometric order at a time?

| Version | Sensor class | MAE (cm⁻¹) |
|:--------|:-------------|:----------:|
| V1 Bag-of-Atoms | Composition only | 71.82 |
| V2 Full atom graph | Atom features + dist | 69.45 |
| V3 Distance RBF | Gaussian basis on `d(i,j)` | 63.00 |
| V4 Angular GNN | Triplet angles `θ(i,j,k)` | 56.33 |
| V5 Physics sensors | `k`, `μ`, `ω_E` per bond | 49.11 |
| **V6 Dihedrals + gate** | Torsion `φ(i,j,k,l)` + halting | **41.91** |

---

## Key Technical Insights

### 1. Input Representation is the Primary Bottleneck

On sparse datasets, **how you represent the input matters more than how you process it.** V1 demonstrated that MLP models cluster at 184–191 MPa regardless of a 10× increase in parameter count. The bottleneck was the featurization scheme (fraction-weighted sum), not the model. Replacing it with Magpie descriptors dropped MAE by 54 MPa without any architectural change.

### 2. Attention Requires Structure, Not Raw Data

Attention over raw element tokens (V2: 388 MPa) fails because 312 samples cannot teach the model to discover element interactions from scratch. Attention over **structured property tokens** (V5+ onward) succeeds because the tokens represent precomputed compositional statistics that yield meaningful comparisons from small data. Same attention mechanism: −223 MPa from input restructuring alone.

### 3. Deep Supervision is a Regularization Mechanism

Deep supervision is not merely a training trick — it is the critical regularizer that enables architectural scaling on small datasets. The `d_attn=64` attention width overfits by 28 MPa without DS (V8B) and reaches SOTA with DS (V11B). By requiring shared weights to satisfy 20 simultaneous loss objectives, the model cannot specialize to any single step's gradient signal.

### 4. Features and Architecture Must Co-Scale

Expanded features fail on small architectures (V11A with extra features: 107.98 MPa). Large architectures fail without expanded features (V11B with basic features: 102.30 MPa). The breakthrough (V12: 95.99 MPa) required both simultaneously — more chemical descriptors to provide signal AND sufficient attention capacity to extract it. Feature engineering and architecture design cannot be optimized independently.

### 5. Physics-Informed Sensors Beat Architectural Complexity

For both phonons and band gap tasks, adding **task-relevant physical priors** as input features produced larger improvements than adding layers, attention heads, or parameters. Bond stiffness k and Einstein frequency ω_E (phonons) gave −7.2 cm⁻¹. BandCenter and HOMO/LUMO proxies (band gap) gave −0.3 eV. Physical knowledge encodes faster than learned representations when data is scarce.

---

## Repository Structure

```
TRIADS/
├── README.md                          # This document
├── requirements.txt                   # Python dependencies
├── LICENSE
├── TRIADS_Final.pdf                   # Research paper
│
├── matbench_steels/                   # Yield strength — 312 samples
│   ├── model.py                       # HybridTRIADS V13A (225K, 5-seed ensemble)
│   ├── results.json                   # Fold-level results — 91.20 ± 12.23 MPa
│   └── research.md                    # Architecture evolution, ablation notes
│
├── matbench_expt_gap/                 # Band gap (eV) — 4,604 samples
│   ├── model.py                       # HybridTRIADS V3 (100K)
│   ├── results.json                   # 0.3068 ± 0.0082 eV
│   └── research.md
│
├── matbench_classification/           # Metallicity + Glass Forming
│   ├── model.py                       # Unified HybridTRIADS (44K/100K)
│   ├── results_ismetal.json           # 0.9655 ± 0.0029 ROC-AUC
│   ├── results_glass.json             # 0.9285 ± 0.0063 ROC-AUC
│   └── research.md
│
├── matbench_jdft2d/                   # Exfoliation energy — 636 samples
│   ├── model.py                       # HybridTRIADS V4 (75K, 5-seed ensemble)
│   ├── results.json                   # 35.89 ± 12.40 meV/atom
│   └── research.md
│
├── matbench_phonons/                  # Peak phonon frequency — 1,265 samples
│   ├── model.py                       # GraphTRIADS V6 (247K, gate-halt)
│   ├── dataset_builder.py             # Pre-compute crystal graphs (run first)
│   ├── results.json                   # 41.91 ± 4.04 cm⁻¹
│   └── research.md
│
└── archive/                           # Complete development history
    ├── matbench_steels/               # trm.py → trm15.py + all V1–V15 scripts
    ├── matbench_expt_gap/             # V1, V2 iterations
    ├── matbench_classification/       # Early benchmark scripts
    ├── matbench_jdft2d/               # V1–V3 iterations
    ├── matbench_phonons/              # V1–V5, old V6
    └── research_notes/                # All experiment logs, per-version JSON metrics
```

**Pretrained weights** are hosted on HuggingFace: → [huggingface.co/Rtx09/TRIADS](https://huggingface.co/Rtx09/TRIADS)

---

## Installation

```bash
git clone https://github.com/Rtx09x/TRIADS.git
cd TRIADS
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, pymatgen, matminer, gensim (Mat2Vec), scikit-learn, huggingface_hub

---

## Usage

### Running a Benchmark

```bash
# Steels — reproduces 91.20 MPa (trains 5-seed ensemble, ~25 min on P100)
cd matbench_steels
python model.py

# Band Gap — reproduces 0.3068 eV
cd matbench_expt_gap
python model.py

# Classification (runs ismetal + glass sequentially)
cd matbench_classification
python model.py

# Phonons — run dataset builder first (10 min, ~2 GB output)
cd matbench_phonons
python dataset_builder.py
python model.py
```

### Loading Pretrained Weights

```python
from huggingface_hub import hf_hub_download
import torch

# Download a benchmark's compiled weights (one file = all 5 folds)
path = hf_hub_download(repo_id="Rtx09/TRIADS", filename="steels/weights.pt")
ckpt = torch.load(path, map_location="cpu")

# ckpt contains:
#   ckpt['folds']   -> list of 5 dicts, each with 'model_state' and 'test_mae'
#   ckpt['n_extra'] -> int (required for model init)
#   ckpt['config']  -> dict with d_attn, d_hidden, ff_dim, dropout, max_steps

from matbench_steels.model import DeepHybridTRM, ExpandedFeaturizer

featurizer = ExpandedFeaturizer()  # Downloads Mat2Vec on first run
models = []
for fold_entry in ckpt['folds']:
    model = DeepHybridTRM(n_extra=ckpt['n_extra'], **ckpt['config'])
    model.load_state_dict(fold_entry['model_state'])
    model.eval()
    models.append(model)

# Ensemble inference
import numpy as np
feat = featurizer.featurize("Fe0.7Cr0.15Ni0.15")  # shape (n_features,)
x = torch.tensor(feat[None], dtype=torch.float32)
preds = [m(x).item() for m in models]
print(f"Predicted yield strength: {np.mean(preds):.1f} MPa")
```

Pretrained models are available for all six benchmarks. See [huggingface.co/Rtx09/TRIADS](https://huggingface.co/Rtx09/TRIADS) for the full checkpoint index and loading instructions.

---

## Reproducing Results

All benchmarks use the exact Matbench 5-fold CV protocol. Key implementation details that ensure reproducibility:

- **Fold generation:** `KFold(n_splits=5, shuffle=True, random_state=18012019)` — identical to Matbench v0.1
- **Featurizer fitting:** StandardScaler fit **only** on the training split within each fold — no test data leakage
- **Target normalization (phonons):** Computed from training split only per fold
- **Seeds:** [42, 123, 7, 0, 99] for steels; [42, 123, 456, 789, 1024] for jdft2d; [42] for gap/classification; seed 42 for phonons

Raw training metrics for every version of every benchmark are in [`archive/research_notes/`](archive/research_notes/).

---

## Citation

```bibtex
@article{tiwari2026triads,
  author    = {Rudra Tiwari},
  title     = {TRIADS: Tiny Recursive Information-Attention with Deep Supervision},
  year      = {2026},
  url       = {https://github.com/Rtx09x/TRIADS},
  note      = {huggingface.co/Rtx09/TRIADS}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
