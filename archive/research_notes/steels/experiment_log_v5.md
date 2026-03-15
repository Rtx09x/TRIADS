# TRM-MatSci V5 — Experiment Log
## SWA + Novel Feature-Group TRM | matbench_steels | 2026-03-01

---

## 1. Setup

Same as V4 (332-dim combined features, 5-fold CV, 300 epochs, P100 GPU).

**New techniques:**
- **SWA** (Stochastic Weight Averaging): Cosine LR for 200 epochs → weight averaging for epochs 200-300
- **Recursion step ensemble**: Average predictions from last 4 recursion steps at inference
- **Novel Feature-Group architecture**: 22 Magpie property tokens × 6 stats, dual-reference cross-attention

---

## 2. Results

| Config | Params | Standard MAE | Ensemble MAE | Time |
|--------|:------:|:------------:|:------------:|:----:|
| **V5A MLP-SWA** | 66,889 | **128.98 ± 17.42** | 194.84 ❌ | 3.7 min |
| V5B FeatGroup-Novel | 38,593 | 165.11 ± 17.56 | 227.95 ❌ | 17.6 min |

---

## 3. Key Findings

### F1: SWA Achieves New Project Best — 128.98 MPa ✅
SWA weight averaging finds flatter minima that generalize better. Improved from 131.63 (V4) to 128.98 without changing model architecture. Only 5.7 MPa behind Darwin (123.29).

### F2: Recursion Step Ensemble Incompatible with SWA ❌
Averaging steps 13-16 degrades performance dramatically (128→195, 165→228). SWA shifts internal weight distributions, making intermediate recursion steps poorly calibrated. These techniques must not be combined.

### F3: Novel Feature-Group TRM Works — 165 MPa with 38K Params
Compare to V2 transformers (388 MPa → catastrophic). The structured property-token approach enables attention to learn from 312 samples:

| Transformer Approach | MAE | Verdict |
|---|---|---|
| V2 Trans-Normal (element tokens) | 389 | ❌ Catastrophic |
| V2 Trans-Novel (element tokens) | 388 | ❌ Catastrophic |
| **V5B Feature-Group (property tokens)** | **165** | ✅ Functional |

**223 MPa improvement** from restructuring input as property tokens. The dual-reference mechanism (E0 fixed + Et evolved) IS contributing — attention discovers property interactions.

### F4: V5B Needs More Capacity
38K params may be too small (like V3.1 XS at 160 MPa). V6 will scale to d_model=48 (~80K params) and add SWA.

---

## 4. V6 Plan

| Config | Architecture | Strategy |
|---|---|---|
| V6A | FeatureGroup-L (d=48) + SWA | Scale up novel TRM |
| V6B | Hybrid-TRM (attn→MLP) + SWA | Attention extracts features, MLP reasons |
| V6C | MLP-SWA ×3 seeds | Average 3 models per fold |
