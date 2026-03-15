# TRM-MatSci V2 — Architecture Plan
*Drafted: 2026-03-01 | Based on V1 findings*

---

> This document defines everything for V2. Every decision is explained. Every number is justified by V1 data or first principles. Nothing is arbitrary.

---

## Why V2 Exists

V1 proved the recursive reasoning mechanism works — Panel 4 of the plot showed smooth MAE descent from ~1400 to 184 across 16 steps. The model is learning. But it hit a hard ceiling at 184 MPa that no amount of extra parameters could break.

**Root cause (confirmed):** The input — a single 200-dim weighted average — destroys all element-element interaction information before the model even starts thinking. No recursive reasoning can recover information that was never provided.

**V2 goal:** Fix the food. Keep everything else that worked.

---

## Section 1 — The New Input Pipeline (The Food Fix)

### 1.1 Old Pipeline (V1 — Broken)

```
Steel: Fe0.80 C0.02 Mn0.10 Cr0.08
                ↓
Fe_vec×0.80 + C_vec×0.02 + Mn_vec×0.10 + Cr_vec×0.08
                ↓
        ONE 200-dim vector
        (all element identity destroyed)
                ↓
        TRM reasoning (working on mush)
```

**Problem:** Two steels `Fe0.80 C0.02 Cr0.18` and `Fe0.80 C0.02 Mn0.18` have very similar weighted sums if Cr and Mn happen to have similar Mat2Vec vectors. But they behave very differently in reality because Cr forms carbides with C, while Mn does solid solution strengthening. The model never gets to learn this.

### 1.2 New Pipeline (V2 — Element-as-Token)

```
Steel: Fe0.80 C0.02 Mn0.10 Cr0.08
                ↓
Each element becomes its own token with its own feature vector
                ↓
Token_Fe: [mat2vec_Fe(200) | radius(1) | electronegativity(1) | group(1) | period(1) | fraction(1)]  = 205 dims
Token_C:  [mat2vec_C(200)  | radius(1) | electronegativity(1) | group(1) | period(1) | fraction(1)]  = 205 dims
Token_Mn: [mat2vec_Mn(200) | radius(1) | electronegativity(1) | group(1) | period(1) | fraction(1)]  = 205 dims
Token_Cr: [mat2vec_Cr(200) | radius(1) | electronegativity(1) | group(1) | period(1) | fraction(1)]  = 205 dims
                ↓
        4 tokens of 205 dims each
        [4 × 205] — EACH ELEMENT IS SEPARATE
                ↓
        Self-attention across tokens (learn element interactions)
                ↓
        Pool → single vector
                ↓
        TRM recursive reasoning (16 steps, working on rich input)
```

### 1.3 Variable-Length Input (Solved)

Different steels have different numbers of elements. Transformers handle this natively:

```
Steel A (3 elements):  [Fe] [C]  [Cr]  [PAD] [PAD]  ← attention mask: ignore PAD
Steel B (5 elements):  [Fe] [C]  [Ni]  [Mn]  [Mo]   ← full attention on all 5
Steel C (2 elements):  [Fe] [C]  [PAD] [PAD] [PAD]  ← attention mask: ignore PAD
```

**Max sequence length = 15** (the most elements any steel in the dataset has). Shorter sequences are padded to 15 with zero vectors. The attention mask is a boolean tensor that marks PAD positions — PyTorch's `MultiheadAttention` accepts this natively via `key_padding_mask`.

---

## Section 2 — Elemental Properties: Where They Come From

### 2.1 Source: `pymatgen.core.Element`

All elemental properties come from **pymatgen** — already installed in our environment. No additional downloads. No API calls. Properties are looked up once at featurization time and cached.

```python
from pymatgen.core import Element

el = Element("Fe")
el.atomic_radius          # 1.26 Å  (Angstroms)
el.X                      # 1.83    (Pauling electronegativity)
el.group                  # 8       (periodic table group)
el.row                    # 4       (periodic table period)
el.atomic_mass            # 55.845  (g/mol)
el.melting_point          # 1811 K  (optional)
```

All properties are available for every element in the periodic table. No missing values for any steel-relevant element.

### 2.2 The 5 Properties We Include Per Element Token

| Property | Source | Dims | Why This One |
|----------|--------|:----:|-------------|
| **Mat2Vec embedding** | GCS Word2Vec model | 200 | Captures chemical knowledge from 3M+ papers. Proven in V1 to carry real signal. |
| **Atomic radius** | `el.atomic_radius` | 1 | Controls solid solution strengthening — smaller atoms fit in Fe lattice better |
| **Electronegativity** | `el.X` | 1 | Controls bond types — high ΔEN between elements → ionic character → different strength |
| **Group** | `el.group` | 1 | Periodic table column — elements in same group behave similarly (Cr/Mo/W all carbide formers) |
| **Period** | `el.row` | 1 | Periodic table row — governs electron shell, atomic size trends |

**+ fraction in alloy** (from composition): 1 dim

**Total per token: 200 + 1 + 1 + 1 + 1 + 1 = 205 dims**

### 2.3 Normalization

All properties are normalized before concatenation:

```python
# Mat2Vec: already unit-normalized (from Word2Vec training)
# Scalar properties: StandardScaler fit on TRAINING data only (no leakage)
scaler = StandardScaler()
scaler.fit(train_properties)  # fit on train only
token_features = scaler.transform(all_properties)  # apply to train+val+test
```

---

## Section 3 — Head Design

### 3.1 How Many Heads and Why

**Decision: 4 heads**

We have 4 distinct categories of information in each token:
1. Chemical identity (Mat2Vec — what is this element?)
2. Bonding behavior (electronegativity — how does it bond?)
3. Physical size/scale (atomic radius, period — how does it fit?)
4. Periodic family (group — what family does it belong to?) + fraction

4 categories → 4 heads, one per category. Clean, principled, and aligns with nhead=4, h=256 → 64 dims/head (the sweet spot from V1 analysis).

### 3.2 Head Specialization: Natural vs. Forced

**Important note:** Standard `MultiheadAttention` does NOT force each head to look at specific features. All features are projected together into attention space and heads specialize *through training*.

However, we structure the input so that specialization is *encouraged*:

```
Token structure (205 dims):
[0:200]   = Mat2Vec (chemical identity)
[200]     = atomic_radius
[201]     = electronegativity
[202]     = group
[203]     = period
[204]     = fraction

Input projection: Linear(205 → 256)
Then attention splits 256 into 4 heads of 64 each.

The model LEARNS which head attends to what during training.
But we can inspect attention weights afterwards to verify specialization.
```

**Why not hard-code heads to features?** Because forcing head 1 to only see Mat2Vec dims removes the model's ability to find useful cross-feature relationships (e.g., "high electronegativity AND short radius" might matter together). Let training discover the optimal combination.

### 3.3 Expected Head Behavior After Training

Based on what we know from materials science:
- At least one head should learn to attend strongly between C and Cr/Mo/V (carbide forming)
- At least one head should attend between main element (Fe) and all others (solid solution effects)
- At least one head should specialize on fraction magnitude (which elements are dominant?)

We will visualize attention maps after training to verify. This visualization is publishable as a figure showing the model learned real chemistry.

---

## Section 4 — Architecture Decisions (Locked)

### 4.1 Decided and Fixed

| Parameter | Value | Reason |
|-----------|:-----:|--------|
| **Input dims per token** | 205 | Mat2Vec(200) + 5 elemental properties |
| **Max sequence length** | 15 | Longest steel in dataset |
| **hidden_dim (D)** | 256 (minimum 128) | 64 dims/head with nhead=4 |
| **nhead** | 4 | 4 information categories in input |
| **dims_per_head** | 64 | D/nhead = 256/4. Sweet spot confirmed from V1 |
| **recursion_steps** | 16 | Smooth convergence observed. Diminishing returns after 16 |

### 4.2 Still To Decide (After Seeing V2 Results)

| Parameter | Options | Decision Basis |
|-----------|---------|----------------|
| **ff_dim** | 256, 512, 1024 | Controls total param count. Decide based on what budget we want |
| **dropout** | 0.1, 0.2, 0.3 | Higher = more regularization. Start at 0.1, test 0.2 |
| **num_layers** | 1, 2 (current) | Keep 2, only increase if V2 Transformer still underperforms |
| **MLP vs Transformer** | Both | Train both families. Compare on equal footing |
| **HRM option** | Hierarchical | Test after seeing Transformer V2 results |

---

## Section 5 — The Full V2 Feature Engineering Pipeline (Code Spec)

### 5.1 Featurizer Class: `ElementTokenFeaturizer`

```python
class ElementTokenFeaturizer:
    """
    Converts a pymatgen Composition into a padded sequence of element tokens.
    
    Each token = [mat2vec(200) | radius(1) | electronegativity(1) | group(1) | period(1) | fraction(1)]
    
    Output shape: [max_len, 205] + attention_mask [max_len]
    """
    
    def __init__(self, mat2vec_embeddings: dict, max_len: int = 15):
        self.embeddings = mat2vec_embeddings  # {symbol: np.array(200)}
        self.max_len = max_len
        self.feature_dim = 205
        self.scaler = None  # fit on training data
    
    def get_element_features(self, symbol: str, fraction: float) -> np.ndarray:
        """Build 205-dim feature vector for one element."""
        el = Element(symbol)
        mat2vec = self.embeddings.get(symbol, np.zeros(200))
        scalar_features = np.array([
            el.atomic_radius or 0.0,
            el.X or 0.0,          # Pauling electronegativity
            float(el.group),
            float(el.row),
            fraction
        ])
        return np.concatenate([mat2vec, scalar_features])  # 205-dim
    
    def featurize(self, composition: Composition) -> tuple:
        """
        Returns:
            tokens: np.array [max_len, 205]  (padded)
            mask:   np.array [max_len]        (True = PAD, ignore in attention)
        """
        tokens = np.zeros((self.max_len, self.feature_dim))
        mask = np.ones(self.max_len, dtype=bool)  # True = ignore (PAD)
        
        elements = [(str(el), amt) for el, amt in composition.items()]
        for i, (symbol, fraction) in enumerate(elements[:self.max_len]):
            tokens[i] = self.get_element_features(symbol, fraction)
            mask[i] = False  # False = attend to this token
        
        return tokens, mask
```

### 5.2 Property Normalization Protocol

```python
# CRITICAL: fit scaler on training data ONLY to prevent data leakage
def fit_scaler(train_compositions, featurizer):
    all_scalars = []
    for comp in train_compositions:
        for el, frac in comp.items():
            el_obj = Element(str(el))
            all_scalars.append([
                el_obj.atomic_radius or 0.0,
                el_obj.X or 0.0,
                float(el_obj.group),
                float(el_obj.row),
                frac
            ])
    scaler = StandardScaler().fit(all_scalars)
    return scaler

# Apply to all splits AFTER fitting on train only
```

### 5.3 Dataset Class: `SteelsTokenDataset`

```python
class SteelsTokenDataset(Dataset):
    def __init__(self, compositions, targets, featurizer, scaler):
        self.data = []
        for comp, target in zip(compositions, targets):
            tokens, mask = featurizer.featurize(comp)
            # normalize scalar part (last 5 dims) using fitted scaler
            tokens[:, 200:] = scaler.transform(tokens[:, 200:].reshape(-1, 5)).reshape(-1, 5)
            # normalize Mat2Vec part (already normalized by Word2Vec, skip or re-normalize)
            self.data.append((
                torch.tensor(tokens, dtype=torch.float32),     # [15, 205]
                torch.tensor(mask, dtype=torch.bool),           # [15]
                torch.tensor(target, dtype=torch.float32)       # scalar
            ))
    
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
```

---

## Section 6 — V2 Model Configs (Proposed — ff_dim TBD)

Configs are named `{arch}-V2-{hidden}h-{nhead}n`. ff_dim will be adjusted to hit specific param targets.

| Config | Arch | D | nhead | dims/head | ff_dim | Est. Params | Notes |
|--------|------|:-:|:-----:|:---------:|:------:|:-----------:|-------|
| MLP-V2-128h | MLP-TRM | 128 | — | — | 256 | ~350K | Baseline, h128 minimum |
| MLP-V2-256h | MLP-TRM | 256 | — | — | 256 | ~700K | Bigger MLP |
| Trans-V2-256h-4n | Trans-TRM | 256 | 4 | 64✓ | 256 | ~900K | Sweet spot config |
| Trans-V2-256h-4n-large | Trans-TRM | 256 | 4 | 64✓ | 512 | ~1.3M | Larger version |
| HRM-V2-256h | HRM-TRM | 256 | 4 | 64✓ | 256 | ~1.2M | Hierarchical, if time allows |

All configs:
- ✅ 1000 epochs
- ✅ Fixed cosine schedule (no early stopping for Transformer)
- ✅ Stratified val split
- ✅ Element-wise token input (205-dim per token)
- ✅ GPU pipeline: num_workers=4, pin_memory=True, batch_size=32

---

## Section 7 — Training Protocol Changes from V1

| Change | V1 | V2 | Reason |
|--------|----|----|--------|
| **Epochs** | 200 | 1000 | Transformer not converged at 200 |
| **Early stopping** | Yes (for all) | Only for MLP (not Transformer) | Transformer's `Best=Val` at ep198 shows it needs more time |
| **Val split** | Random 15% | Stratified 15% | Fold 3 bias in V1 distorted early stopping |
| **Batch size** | 16 | 32 | Larger batches = more stable gradients, better GPU utilization |
| **num_workers** | 0 | 4 | Pre-compute tokens, parallel loading → GPU stays busy |
| **pin_memory** | False | True | Faster CPU→GPU transfer |
| **Dropout** | 0.1 | 0.1 → 0.2 (test) | Higher regularization may help with richer input |

---

## Section 8 — What V2 Will Tell Us

| Question | How V2 Answers It |
|----------|-------------------|
| Does element-wise input beat weighted sum? | Compare V2 best vs V1 best (184.4 MPa). Any improvement = the food was the problem |
| Is 64 dims/head enough? | Watch Transformer training curves — should they converge now with more epochs? |
| MLP vs Transformer with proper input? | Head-to-head at same config, same epochs |
| Does HRM beat TRM on this task? | Include one HRM config, compare directly |
| Where is the new ceiling? | If V2 clusters at some new value, that's the next bottleneck to identify |

---

## Section 9 — What Stays the Same from V1

- ✅ Dataset: `matbench_steels` (312 samples, same 5-fold splits, random_state=18012019)
- ✅ Mat2Vec source: GCS download, 200-dim Word2Vec embeddings
- ✅ Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- ✅ Scheduler: CosineAnnealingLR
- ✅ Gradient clipping: max_norm=1.0
- ✅ Recursion steps: 16
- ✅ Output: single MAE scalar
- ✅ Hardware: Kaggle P100

---

## Section 10 — Open Questions (Decide Before Coding)

1. **ff_dim**: What total param count are we targeting? 500K? 1M? This sets ff_dim.
2. **HRM in V2 or V3?**: Time on Kaggle is limited. Include HRM in V2 or save for V3?
3. **Mat2Vec normalization**: Should we re-normalize Mat2Vec vectors per-sample or leave as-is from Word2Vec?
4. **Attention pooling vs mean pooling**: After element tokens go through attention, how do we pool into one vector? Options:
   - **Mean pool**: average all non-PAD tokens (simple, CrabNet uses this)
   - **Fraction-weighted pool**: weight each element's output by its fraction (more chemically motivated)
   - **[CLS] token**: add a learnable summary token at position 0 (standard NLP approach)

---

## Summary — What V2 Is

> **V2 is the same TRM reasoning engine, fed real food instead of mush.**

The 16-step recursive loop stays exactly the same. The only thing that changes is:
1. Input is now `[N_elements × 205]` instead of `[200]`
2. An attention encoder processes this into a rich pooled vector before the TRM loop
3. Training runs 5x longer so the Transformer can actually converge
4. Architecture dimensions are calculated from first principles (food → heads → dims), not set blindly

If V2 achieves 130–150 MPa, it confirms the input was the bottleneck and we proceed to V3 (element-wise cross-attention inside the recursive loop itself). If V2 achieves <107 MPa, we've beaten CrabNet with a fundamentally simpler model and have a publishable result.

---

## FINAL DECISIONS — All Locked (2026-03-01, post V1 benchmark)

> Everything below is finalized. No more open questions. Ready to code.

### Final Model List — 6 Models

| # | Name | Arch | D | ff | nhead | dims/head | ~Params |
|:-:|------|:----:|:-:|:--:|:-----:|:---------:|--------:|
| 1 | MLP-S | MLP | 64 | 128 | — | — | ~30K |
| 2 | MLP-L | MLP | 128 | 256 | — | — | ~115K |
| 3 | Trans-Normal-S | Transformer | 256 | 256 | 4 | 64 ✓ | ~710K |
| 4 | Trans-Normal-L | Transformer | 256 | 512 | 4 | 64 ✓ | ~970K |
| 5 | Trans-Novel-S | Transformer | 256 | 256 | 4 | 64 ✓ | ~800K |
| 6 | Trans-Novel-L | Transformer | 256 | 512 | 4 | 64 ✓ | ~1.1M |

**MLP models** have no Normal/Novel split — MLP can't do sequence attention cleanly (16 steps over variable N elements is asymmetric and broken). Both MLP models use attention-weighted pool → MLP-TRM loop.

**Trans-Normal vs Trans-Novel** is the key scientific comparison.

---

### Final Training Hyperparameters

| Parameter | Value | Reason |
|-----------|:-----:|--------|
| **Epochs** | **300** | 200 was too few for Transformer; 1000 risks memorization. 300 is the sweet spot to observe |
| **Dropout** | **0.2** | Increased from 0.1 — richer input (205-dim tokens) gives model more to memorize |
| **Batch size** | **32** | Double V1 for better GPU utilization |
| **num_workers** | **4** | Parallel data loading — GPU stays busy |
| **pin_memory** | **True** | Fast CPU→GPU transfer |
| **Early stopping** | **Both use it** | At 300 epochs, both MLP and Transformer need guard against memorization |
| **Val split** | **Stratified 15%** | Fix the Fold 3 bias from V1 |
| **Optimizer** | **AdamW** lr=1e-3, wd=1e-4 | Same as V1 |
| **Scheduler** | **CosineAnnealingLR** T_max=300 | Same as V1, scaled to 300 epochs |

### On Overfitting (addressed)

With 312 samples and models ranging 30K–1.1M params, overfitting is a real risk — especially for the larger Transformer models. Mitigation strategy:
- **Dropout 0.2** at every recursive step
- **Early stopping** on stratified val set
- **Weight decay 1e-4** in AdamW
- **Gradient clipping** max_norm=1.0
- **Small batch size 32** introduces stochastic noise that regularizes

If V2 results show val-test gap ≥ 40 MPa on any model, that config is overfitting and should be excluded from the final paper comparison. The 30K MLP models are unlikely to overfit; the 1.1M Trans-Novel-L is the highest risk.

---

### Trans-Normal Architecture (forward pass)

Standard dual-state TRM but with richer input:

```
Input: element tokens [T1...TN, 205-dim each]
       ↓
Attention-weighted pool → single 205-dim vector
       ↓
input_proj: Linear(205 → 256)  →  x_proj
       ↓
TRM loop (16 steps, shared weights):
  Step t:
    seq = stack([x_proj, y, z])  →  [B, 3, 256]
    z   = z + SelfAttn(seq)[z_slot]      # z attends to x_proj and y
    y   = y + CrossAttn(query=y, kv=z)   # y attends to z
       ↓
output_head(y) → predicted yield strength
```

Identical to V1 Transformer-TRM but with the pooled input vector being richer (205-dim from true element tokens instead of 200-dim weighted sum).

---

### Trans-Novel Architecture (forward pass — the new design)

**Core idea:** Two parallel element representations — a fixed reference (where we started) and a living state (where we are now). The reasoning states z and y use both.

```
Input: element tokens [T1...TN, 205-dim each]
       ↓
input_proj: Linear(205 → 256) applied to EACH token independently
       ↓
E0 = projected tokens  →  [B, N, 256]  ← FIXED REFERENCE (never changes)
Et = copy of E0        →  [B, N, 256]  ← LIVING STATE (evolves each step)

TRM loop (16 steps, shared weights):
  Step t:
    # Living elements self-attend → evolve understanding of each other
    Et = Et + SelfAttn(Et, mask=padding_mask)   →  [B, N, 256]

    # z cross-attends to BOTH living state AND original reference
    z_from_live = CrossAttn(query=z, kv=Et)     # "where are we now?"
    z_from_ref  = CrossAttn(query=z, kv=E0)     # "where did we start?"
    z = z + z_from_live + z_from_ref

    # y cross-attends to z (same as always)
    y = y + CrossAttn(query=y, kv=z)
       ↓
output_head(y) → predicted yield strength
```

**Why the dual reference helps:**
- `Et` (living): captures how element interactions evolve over recursive steps
- `E0` (fixed reference): always provides stable gradient path back to original input — cleaner backpropagation
- The model can implicitly compute `Et - E0` (how much understanding changed) by comparing both attended values
- This is a genuinely novel mechanism — not published in any existing TRM or materials science model

---

### What We Still Intend to Explore (V3)

After V2 results:
- HRM (Hierarchical Reasoning Model) — nested fast/slow recursive loops
- Longer training experiments if Transformer still hasn't converged at 300 epochs
- Bigger models (>1M params) if V2 results show scaling still helps
