# TRIADS Architecture Evolution: From First Principles to State-of-the-Art

*A comprehensive account of the architectural decisions, failures, breakthroughs, and iterative refinements that shaped TRIADS across 15 major versions and approximately 200 trained models.*

> **See also:** [Hyperparameter Studies](./Hyperparameter_Studies.md) · [Performance Logs](./Performance_Logs.md) · [README](../README.md)
>
> **Training code:** [V1](../Training%20Code/trm.py) · [V2](../Training%20Code/trm2.py) · [V3](../Training%20Code/trm3.py) · [V4](../Training%20Code/trm4.py) · [V5](../Training%20Code/trm5.py) · [V6](../Training%20Code/trm6.py) · [V7](../Training%20Code/trm7.py) · [V8](../Training%20Code/trm8.py) · [V9](../Training%20Code/trm9.py) · [V10](../Training%20Code/trm10.py) · [V11](../Training%20Code/trm11.py) · [V12](../Training%20Code/trm12.py) · [V13](../Training%20Code/trm13.py) · [V14](../Training%20Code/trm14.py) · [V15](../Training%20Code/trm15.py)

---

## 1. The Founding Hypothesis ([V1](../Training%20Code/trm.py))

TRIADS began as an application of the **Tiny Recursive Model (TRM)** concept, originally proposed by Samsung SAIL, to the domain of materials property prediction. The central hypothesis was deceptively simple:

> *Can a model that reuses the same small neural network 16 times in a recursive loop—accumulating "reasoning" with each pass—compete with models orders of magnitude larger for predicting the yield strength of steel alloys?*

The target benchmark was **matbench_steels**: 312 steel alloy compositions mapped to yield strength in MPa, evaluated under strict nested 5-fold cross-validation. This dataset was selected deliberately: it is small enough to challenge deep learning methods (where Random Forests have historically dominated), publicly benchmarked with comparable baselines, and composition-only—matching the TRM's input strategy.

### Initial Architecture: The Dual-State Recursive Loop

The TRM operates on two persistent state vectors, `z` (the reasoning state) and `y` (the prediction draft), which are initialized from a projection of the input and then iteratively refined by passing through the same shared-weight neural block for a fixed number of steps:

```
Initialization:
    x_proj = Linear(input → d_model)
    z₀ = x_proj
    y₀ = x_proj

Recursive Loop (t = 1 to T):
    zₜ = zₜ₋₁ + f_z(zₜ₋₁, yₜ₋₁, x_proj)    # Update reasoning state
    yₜ = yₜ₋₁ + f_y(yₜ₋₁, zₜ)                # Update prediction draft

Output:
    ŷ = output_head(y_T)
```

Two families of `f_z` and `f_y` were tested in V1:
- **MLP-TRM**: 2-layer MLP with LayerNorm, GELU, Dropout, and residual connections.
- **Transformer-TRM**: Multi-head self-attention over the triplet `[x_proj, y, z]` for `f_z`, and cross-attention (y attending to z) for `f_y`.

### V1 Results: The Input Bottleneck Discovery ([📊 results](../Images/trm_results_v2.png) · [📄 JSON](../Training%20Json/summary.json))

The V1 sweep trained **12 models** across parameter budgets of 10K, 50K, and 100K, with hidden dimensions of 64 and 128. The input was the **Mat2Vec** 200-dimensional embedding—a Word2Vec model trained on 3 million materials science abstracts—reduced to a single vector per alloy via fraction-weighted summation.

**Critical finding:** All MLP models clustered between 184–191 MPa MAE despite a 10× increase in parameter budget. The model had sufficient capacity at even 10K parameters; the information bottleneck was the **input representation**, not the reasoning architecture. By collapsing all elements into a single averaged vector, the weighted sum destroyed element-element interaction information before reasoning could even begin.

The Transformer variants suffered a different fate entirely. At a hidden dimension of 64 (producing 16 dimensions per attention head), attention was computationally degenerate—producing errors of 992 MPa, far worse than predicting the dataset mean (229 MPa). At a hidden dimension of 128 (32 dims/head), the Transformer achieved 583 MPa—functional but still catastrophic. An important technical discovery was that all three Transformer-h128 configurations (10K, 50K, 100K targets) mapped to the same physical model (~167K parameters), because the attention matrix parameter floor exceeded all three targets. The build system silently returned the minimum viable model three times.

**Key architectural insight:** Width has asymmetric effects across architectures. For MLP, narrower bottlenecks (h64) provided implicit regularization that outperformed wider variants. For Transformers, width was existential—below 32 dimensions per head, attention cannot compute meaningful similarity scores.

---

## 2. The Element-as-Token Experiment ([V2](../Training%20Code/trm2.py) · [📊 results](../Images/trm_results_v2%20(1).png) · [📄 JSON](../Training%20Json/summary_v2.json))

V2 tested whether converting individual elements into separate tokens for the Transformer—with 205-dimensional feature vectors per element (Mat2Vec 200d + atomic radius + electronegativity + group + period + fraction)—would unlock attention's ability to learn element interactions.

**Result: Complete failure.** The Transformer with element tokens scored 388–391 MPa, worse than the dummy baseline. With only 312 training samples, attention could not discover element interactions from scratch. The MLP, which still used the weighted-sum input, remained unchanged at ~184 MPa.

This failure was consequential: it conclusively demonstrated that 312 samples are insufficient for attention to learn chemical interactions in an unsupervised manner. The models that dominate the matbench_steels leaderboard either use ensemble methods that handle small N natively (TPOT, AutoGluon), rely on heavy feature engineering (RF-SCM/Magpie with 132 hand-crafted descriptors), or pretrain on 300K+ materials (CrabNet). The path forward required encoding interaction information explicitly.

---

## 3. The Feature Engineering Breakthrough (V3–V4)

### V3: Magpie Descriptors ([code](../Training%20Code/trm3.py) · [📊 results](../Images/trm_results_v3.png) · [📄 JSON](../Training%20Json/summary_v3.json))

Recognizing that the model could not discover element interactions from 312 samples, V3 replaced the raw Mat2Vec input with **Magpie compositional descriptors**—132 engineered statistics computed over 22 elemental properties (mean, average deviation, minimum, maximum, range, and mode of properties like electronegativity, atomic radius, melting point, etc.). These descriptors encode interaction information through precomputed statistics rather than requiring the model to discover it.

**The ceiling shattered.** MLP-Magpie-L (248K parameters) achieved **130.33 ± 12.93 MPa**—a 54 MPa improvement over V2 and the first evidence that the TRM architecture could compete on this benchmark. Notably, the relationship between model size and performance reversed: with Magpie input, larger models outperformed smaller ones (130.3 vs 138.4 MPa), confirming that the richer input provided enough signal to justify additional capacity.

### V4: Combined Features ([code](../Training%20Code/trm4.py) · [📊 results](../Images/trm_results_v4.png) · [📄 JSON](../Training%20Json/summary_v4.json))

V4 concatenated Magpie descriptors (132d) with the Mat2Vec pooled embedding (200d) into a 332-dimensional combined feature vector. The combined input did not improve absolute performance (131.63 MPa), but achieved it with 4× fewer parameters (67K vs 248K). This demonstrated that Mat2Vec provided complementary information that improved parameter efficiency, even if it did not contribute novel signal beyond Magpie.

A pivotal observation from V4: MLP-Combined-L reached a validation MAE of **87.9 MPa** on Fold 3—below CrabNet (107.3) and rivaling MODNet (87.8). The model *could* learn patterns at leaderboard-competitive levels. The problem was not capacity or data quality; it was the **validation-test gap** caused by early stopping on a 37-sample validation set.

---

## 4. Stochastic Weight Averaging and the Hybrid Architecture (V5–V7)

### V5A: SWA — Flatter Minima for Better Generalization ([code](../Training%20Code/trm5.py) · [📊 results](../Images/trm_results_v5.png) · [📄 JSON](../Training%20Json/summary_v5.json))

V5 introduced **Stochastic Weight Averaging (SWA)**, which trains normally for 200 epochs with cosine annealing, then averages the model weights over epochs 200–300. SWA finds flatter loss landscape minima that generalize better to unseen data.

**Result:** MLP-SWA achieved **128.98 ± 17.42 MPa** with 67K parameters—a meaningful improvement from 131.63 (V4) without any architectural change. An important negative result: combining SWA with recursion step ensembling (averaging predictions from the last 4 recursion steps) degraded performance dramatically to 194.8 MPa, because SWA shifts internal weight distributions and makes intermediate steps poorly calibrated.

### V5B: Feature-Group TRM — Attention Over Property Tokens ([code](../Training%20Code/trm5.py))

V5B tested a fundamentally different approach to attention: instead of treating each element as a token (which failed in V2), it treated each **property statistic** as a token. The 132 Magpie features were restructured into 22 property tokens × 6 statistics, with a dual-reference cross-attention mechanism.

This architecture went from catastrophic (V2 Transformer: 388 MPa) to functional (**165 MPa** with only 38K parameters)—a 223 MPa improvement from restructuring the input. Attention could learn from 312 samples when the tokens represented structured compositional statistics rather than raw elements.

### V6–V7: The Rise of Hybrid-TRM ([V6 code](../Training%20Code/trm6.py) · [V7 code](../Training%20Code/trm7.py) · [📊 V7 results](../Images/trm_results_v7.png) · [📄 V6 JSON](../Training%20Json/summary_v6.json) · [📄 V7 JSON](../Training%20Json/summary_v7.json))

V6 introduced the **Hybrid-TRM** concept: use a self-attention module to extract features from Magpie property tokens, then feed the pooled representation into the MLP-TRM recursive reasoning loop. Attention handles feature extraction; the MLP handles reasoning.

V7 scaled the Hybrid-TRM's attention dimension from 32 to 48 (`d_attn=48`), yielding a **7.9 MPa improvement** and a new project best of **127.08 ± 18.72 MPa** with only 87K parameters. This was the first time an attention-containing architecture surpassed pure MLP, confirming that the right decomposition was not "attention for everything" or "MLP for everything," but **attention for feature extraction, MLP for recursive reasoning**.

Individual folds reached **104.6** and **109.8 MPa**—below CrabNet (107.3)—demonstrating that the architecture was already capable of state-of-the-art performance on favorable data splits.

---

## 5. The Scaling Wall and the Over-Refinement Paradox (V8–V9)

### V8: Attention Capacity Ceiling ([code](../Training%20Code/trm8.py) · [📊 results](../Images/trm_results_v8.png) · [📄 JSON](../Training%20Json/summary_v8.json))

V8 tested scaling `d_attn` from 48 to 64. **Result: Catastrophic overfitting** (+28 MPa regression). The trend from 32→48 did not extrapolate. For 312 samples, a single layer of d_attn=48 attention represented the maximum usable capacity for learning property interactions without a regularization mechanism strong enough to prevent memorization.

A second critical finding: removing the cross-attention layer (to inject Mat2Vec as a 23rd self-attention token) regressed performance by 16 MPa. The cross-attention layer was not merely providing context—it served as a **second computational layer** (`SA → FF → CA`) that refined representations. Its removal collapsed the model's processing depth.

### V9: The Over-Refinement Paradox ([code](../Training%20Code/trm9.py) · [📊 results](../Images/trm_results_v9.png) · [📄 JSON](../Training%20Json/summary_v9.json))

V9 extended recursion from 16 to 20 steps at zero additional parameter cost. The per-fold results revealed a paradox:
- **Hard folds** (historically high-error splits) **improved** by ~7 MPa each
- **Easy folds** (historically low-error splits) **degraded** by 22–38 MPa

The extra recursion steps over-refined easy samples, pushing them past their optimal prediction state. The SWA weights, calibrated for 16 steps, drifted when run for 20. Fixed-step recursion forces an irreconcilable compromise: short depth under-serves hard samples, deep recursion destroys easy samples.

This observation directly prescribed the solution: the model needed to produce calibrated predictions at *every* step, not just the final one.

---

## 6. Deep Supervision: The Core Breakthrough ([V10](../Training%20Code/trm10.py) · [📊 results](../Images/trm_results_v10.png) · [📄 JSON](../Training%20Json/summary_v10.json))

V10 introduced **Deep Supervision** to the Hybrid-TRM architecture. During training, L1 loss is computed at every recursion step using linearly increasing weights (step 1 gets weight 1, step 2 gets weight 2, ..., step 20 gets weight 20). This forces the model to learn a stable, calibrated prediction trajectory where every step must produce a meaningful output—eliminating the late-step drift that destroyed V9.

### Results: A Paradigm Shift

| Model | MAE (MPa) | Context |
|:------|:---------:|:--------|
| V7B Hybrid-L (Previous Best) | 127.08 | 16 steps, no DS |
| V9A (20 steps, No DS) | 134.59 | Over-refined |
| **V10A (20 steps, Deep Supervision)** | **103.28** | **Beat Darwin, CrabNet, and RF-SCM** |

The per-fold breakdown was transformative. Fold 2 dropped from 153.03 → 95.32 (−57 MPa). Fold 3, which V9 had blown up to 142.55, was brought down to 91.57 by Deep Supervision. Every fold that V9 destroyed was not merely fixed but dramatically improved.

**Why Deep Supervision works for TRM:** Without it, only the final step's loss gradient propagates backward through the shared-weight loop. Earlier steps receive increasingly diluted gradients, allowing them to drift into uncalibrated states. Deep Supervision provides direct gradient signal at every step, forcing the entire trajectory to remain supervised and stable. The shared weights must now satisfy 20 simultaneous objectives rather than one, which acts as an extremely powerful regularizer.

### Reproducibility ([V10.1](../Training%20Code/trm10_1.py) · [📊 results](../Images/trm_results_v10_1.png) · [📄 JSON](../Training%20Json/summary_v10_1.json))

A 3-seed reproducibility test confirmed the architecture's stability: **Grand Mean of 105.85 MPa (±3.00 MPa between seeds)**. All three seeds defeated Darwin (123.29), two defeated CrabNet (107.31), and the original seed defeated RF-SCM/Magpie (103.51). The cross-seed variance of 3.00 MPa on a 312-sample dataset is exceptionally tight.

---

## 7. Scaling With Deep Supervision (V11–V12)

### V11B: Deep Supervision Unlocks Previously Overfitting Architectures ([code](../Training%20Code/trm11.py) · [📊 results](../Images/trm_results_v11.png) · [📄 JSON](../Training%20Json/summary_v11.json))

V8 had proven that scaling `d_attn` from 48 to 64 caused catastrophic overfitting. V11B tested the exact same `d_attn=64` configuration, but now regularized by Deep Supervision.

**Result: 102.30 ± 8.61 MPa**—a new project SOTA. The fold standard deviation of ±8.61 was the lowest ever recorded, indicating the most uniformly generalizing model in the project's history. Three folds dropped below 100 MPa (95.60, 99.82, 95.48).

**The principle:** Deep Supervision is not merely a training trick; it is a **regularization mechanism** that unlocks architectural scaling on small datasets. Configurations that would catastrophically overfit without it become the best-performing models with it.

### V12A: Feature Expansion Requires Attention Capacity ([code](../Training%20Code/trm12.py) · [📊 results](../Images/trm_results_v12.png) · [📄 JSON](../Training%20Json/summary_v12.json))

V12A combined expanded `matminer` features (Magpie + Mat2Vec + additional chemical descriptors from ElementFraction, Stoichiometry, ValenceOrbital, IonProperty, and BandCenter featurizers) with the `d_attn=64` architecture and standard Deep Supervision.

**Result: 95.99 ± 10.56 MPa**—the first model to break the 100 MPa barrier. Four out of five folds landed under 100 MPa. The key insight was a **coupling effect**: expanded features had failed on smaller architectures (V11A: 107.98 MPa with `d_attn=48`), and larger architectures had failed without expanded features (V11B: 102.30 MPa with Magpie-only). The synergy required both—more chemical descriptors to provide signal *and* sufficient attention capacity to extract it.

V12B tested a confidence-weighted Deep Supervision variant (a learned `confidence_head` that softmax-weights predictions across steps instead of using fixed linear weights). While it produced the single best fold result ever recorded—**74.55 MPa** on Fold 3, surpassing TPOT-Mat (79.95) on that split—the overall mean was slightly worse (97.59 vs 95.99) with higher variance. The confidence head nearly always placed maximum weight on the final step, effectively collapsing into standard DS but with extra parameters and instability.

---

## 8. The Final Architecture: TRIADS ([V13](../Training%20Code/trm13.py) · [📊 results](../Images/trm_results_v13.png) · [📄 JSON](../Training%20Json/summary_v13.json))

V13 introduced two final architectural refinements:

### Two-Layer Self-Attention

A second self-attention layer was stacked in the feature extractor, enabling the model to learn **second-order compositions of material properties**—interactions between interactions. The architecture became:

```
Input (Magpie + Mat2Vec + Matminer features)
    ↓
Tokenize into 22 Magpie property groups
    ↓
Self-Attention Layer 1 (d_attn=64): Learn 1st-order property interactions
    ↓
Self-Attention Layer 2 (d_attn=64): Learn 2nd-order interaction patterns
    ↓
Cross-Attention with Mat2Vec context (chemical semantics)
    ↓
Pool → MLP-TRM Recursive Loop (20 steps, Deep Supervised)
    ↓
Output: Predicted yield strength (MPa)
```

### 5-Seed Ensemble

With single-seed models operating at ~96 MPa, the remaining error was predominantly **variance** (sensitivity to weight initialization and batch ordering) rather than **bias** (systematic architectural limitation). Averaging predictions from 5 independently trained models (seeds 42, 123, 7, 0, 99) destroyed the variance component.

**Final result: 91.20 ± 12.23 MPa** (5-seed ensemble average). The gap between the best single seed (96.77) and the ensemble (91.20) was 5.57 MPa—conclusive proof that at this performance level, individual models have low systematic bias but high variance, and ensemble averaging is the optimal variance reduction strategy.

---

## 9. Feature Expansion to the Limit ([V14](../Training%20Code/trm14.py) · [📊 results](../Images/trm_results_v14.png) · [📄 JSON](../Training%20Json/summary_v14.json))

V14 tested a **Mega-Featurizer** that expanded the input from ~462 dimensions (Magpie + Mat2Vec) to ~670 dimensions by adding domain-specific alloy properties: DEML defect properties, WenAlloys radii and shear modulus, Miedema enthalpy of mixing, YangSolidSolution omega parameter, and TMetalFraction transition metal content.

**V14A (Flat concatenation): 94.94 ± 14.21 MPa**—a new single-seed SOTA, surpassing V12A (95.99). The metallurgical descriptors (shear modulus, solid solution parameters, mixing enthalpy) provided exactly the domain context the model had been missing.

V14B tested tokenizing all structured properties into 58 distinct attention tokens with a 2-pass self-attention TRM loop inside the feature extractor. While it won individual folds by massive margins (81.56 vs 85.37 on Fold 3), the 58-token attention matrix was too sparse for 312 samples and created high fold-to-fold variance.

---

## 10. Hitting the Ceiling ([V15](../Training%20Code/trm15.py) · [📊 results](../Images/trm_results_v15.png) · [📄 JSON](../Training%20Json/summary_v15.json)) and Conclusions

V15 implemented the **Hierarchical Tiny Reasoning Model (HTRM)** inspired by arXiv:2506.21734, with a slow abstract planning module (Attention Transformer) steering a fast detailed module (MLP-TRM) using detached gradients between hierarchy cycles.

**Result: Catastrophic failure at 431.86 MPa.** Detaching gradients between hierarchy levels destabilized training entirely on this small dataset. The architecture produced errors worse than V1's initial baselines.

This failure, combined with V14B's volatility, established the project's capability ceiling:
- **Ensemble SOTA:** V13A at **91.20 MPa** (5-seed average of 2-layer SA + Deep Supervised Hybrid-TRM)
- **Single-Seed SOTA:** V14A at **94.94 MPa** (Mega-Features + 2-layer SA Hybrid-TRM)

---

## Summary: The Six Pillars of TRIADS

The complete architectural evolution, distilled to its essential components:

| Pillar | Introduced | Impact | Mechanism |
|:-------|:-----------|:-------|:----------|
| **Engineered Feature Statistics** | V3 | −54 MPa | Magpie descriptors encode interaction information the model cannot discover from 312 samples |
| **Mat2Vec Chemical Semantics** | V4 | +Parameter efficiency | Cross-attention context from 3M+ paper embeddings |
| **Stochastic Weight Averaging** | V5 | −3 MPa | Finds flatter loss minima for better generalization |
| **Hybrid Architecture** | V6–V7 | −2 MPa | Attention extracts features; MLP reasons recursively |
| **Deep Supervision** | V10 | −24 MPa | Forces calibrated predictions at every step; enables architectural scaling |
| **Multi-Seed Ensembling** | V13 | −6 MPa | Destroys variance when bias is low |

**Total improvement: 184.4 → 91.20 MPa (50.5% reduction in error)**
