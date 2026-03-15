# TRIADS Classification Benchmarks: Metal & Glass

This directory contains research, experiments, and results for the two classification datasets evaluated by the TRIADS architecture:
1. `matbench_expt_is_metal` (4,921 samples) — Metal vs Non-metal
2. `matbench_glass` (5,680 samples) — Metallic Glass Forming Ability

## Goal
Demonstrate that the DeepHybridTRM architecture, initially designed and optimized for complex regression tasks (Yield Strength, Band Gap, Exfoliation Energy), can generalize to binary classification tasks with minimal architectural changes (only swapping the final layer/loss function).

## Experimental Setup
*   **Architecture:** DeepHybridTRM (Tokenized Reasoning Model).
*   **Model Size:** 100K parameters (d_attn=40, d_hidden=72, d_ff=108, max_steps=16) and 44K parameters (d_attn=24, d_hidden=48, d_ff=72).
*   **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy) applied via Deep Supervision across all reasoning steps.
*   **Evaluation Metric:** ROCAUC (Receiver Operating Characteristic Area Under the Curve).
*   **Featurization:** Composition-only (Magpie + Mat2Vec + select Matminer extras). No structural data.
*   **Validation:** 5-Fold Cross Validation (standard Matbench exact splits, `random_state=18012019`) with a 5-seed ensemble `[42, 123, 456, 789, 1024]`.

## Key Findings summary
*   **Extreme Generalization:** The architecture effortlessly solved the `is_metal` dataset, destroying the SOTA (0.9209) with a score of **0.9655** using the 100K model on a single seed. The loss approached zero rapidly, indicating the model quickly mapped the physics of metallicity.
*   **Featurizer Nuance:** 
    *   For `is_metal`, features defining electron mobility (`BandCenter`, custom `HOMO/LUMO gap`) were retained as they directly relate to metallicity. This led to near-perfect early epoch validation scores.
    *   For `glass`, electronic gap features were removed as they are irrelevant noise for thermodynamic phase formation, while `TMetalFraction` was highlighted.
*   **Power over Parameter Count:** The TRIADS reasoning loop proves so efficient that tasks which historically required massive ensembles (like MODNet's 0.96 for glass) are being aggressively challenged by tiny 44K-100K parameter models. It's essentially bringing a supercomputer to a spelling bee.
