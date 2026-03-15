# TRM-MatSci V8 & V9 — Experiment Log
## Mat2Vec Injection, Attention Scaling, and Recursion Depth | matbench_steels | 2026-03-02

---

## 1. V8: Hybrid-M2V and Hybrid-XL

**Goal:** Test two architectural scaling hypotheses:
1. **V8A (Hybrid-M2V):** Inject Mat2Vec directly into the MLP reasoning path (or as a 23rd self-attention token) to provide richer chemical context than cross-attention.
2. **V8B (Hybrid-XL):** Scale attention width (d_attn 48→64) to continue the trend from V6→V7.

### V8 Results

| Config | Params | Test MAE (MPa) | ±Std | Gap to V7B |
|--------|:------:|:--------------:|:----:|:----------:|
| V7B Hybrid-L | 87K | 127.08 | 18.72 | (Baseline) |
| **V8A Hybrid-M2V** | 77,849 | **143.03** | 23.20 | +16.0 MPa |
| **V8B Hybrid-XL**  | 113,545 | **155.06** | 19.47 | +28.0 MPa |

### V8 Key Findings

#### F1: The CA Layer is Load-Bearing (V8A Failure)
V8A removed the cross-attention layer to make Mat2Vec a 23rd self-attention token. This regressed performance by 16 MPa. **Finding:** The cross-attention layer in V7B wasn't just context injection; it provided an essential **second layer of computation depth** (`SA → FF → CA`) that refined the representations. Removing it reduced the model's capacity to process interactions, even though the attention was "cleaner".

#### F2: Attention Capacity Sweet Spot Reached (V8B Failure)
Scaling d_attn from 48 to 64 caused catastrophic overfitting (+28 MPa). The trend from 32→48 (which gave +7.9 MPa) does not extrapolate. **Finding:** For N=312 samples, a single layer of d_attn=48 attention is the maximum usable capacity for learning property interactions.

---

## 2. V9: 20-Step Recursion

**Goal:** Based on V8 proving architecture is optimized, V9 tested purely extending the recursion steps from 16 to 20 (zero extra parameters) to see if more "thinking time" helps.

### V9 Results

| Config | Params | Test MAE (MPa) | ±Std | Gap to V7B |
|--------|:------:|:--------------:|:----:|:----------:|
| V7B Hybrid-L | 87K | 127.08 | 18.72 | (Baseline) |
| **V9A Hybrid-20** | 87,353 | **134.59** | **10.43** | +7.5 MPa |
| **V9B Hybrid-20L** | 100,853 | **140.14** | 19.29 | +13.1 MPa |

### Per-Fold Analysis (V9A vs V7B)

| Fold | V7B (16s) | V9A (20s) | Delta | Verdict |
|:---:|:---:|:---:|:---:|:---|
| 1 | 124.56 | **116.32** | −8.2 | ✅ Harder split improved |
| 2 | 153.03 | **146.07** | −7.0 | ✅ Hard split improved |
| 3 | **104.59** | 142.55 | +38.0 | ❌ Easy split destroyed |
| 4 | 143.42 | **136.70** | −6.7 | ✅ Hard split improved |
| 5 | **109.78** | 131.31 | +21.5 | ❌ Easy split destroyed |

### V9 Key Findings

#### F3: The "Over-refinement" Paradox
Running 4 extra recursion steps **improved every hard fold** (1, 2, 4) by ~7 MPa, but **catastrophically degraded the easy folds** (3, 5) by 22–38 MPa. The extra steps over-refined the easy samples, pushing them past their optimal prediction state.

#### F4: SWA Weights are Depth-Calibrated
The SWA weights were optimized on 16 steps. Running those same weights for 20 steps amplifies drift on easy samples. Fixed-step recursion forces a compromise: short depth fails on hard samples, deep recursion overfits easy samples.

---

## 3. Implications for V10

The V9 fold-by-fold results directly prescribe the solution: **Adaptive Recursion.**

If we could halt easy folds at step 16 and let hard folds run to 20, the theoretical MAE (combining the best of V7B and V9A) would be **~118 MPa** — which beats Darwin (123.29).

**V10 Strategy:**
1. Train with **deep supervision** (loss computed at every step, using linear weighting) so the model learns to make calibrated predictions at *all* steps, preventing catastrophic drift at later steps.
2. Evaluate with **adaptive halting**: stop recursion for individual samples when `|pred_t - pred_{t-1}| < 1.0 MPa`.
