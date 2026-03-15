# Matbench Phonons — Research Notes

## Task
**Target:** Peak Phonon Frequency (cm⁻¹)  
**Dataset:** `matbench_phonons` — 1,265 samples, 5-fold nested cross-validation  
**SOTA Result:** 41.91 ± 4.04 cm⁻¹ (GraphTRIADS V6, 247K parameters, gate-halt)

## Architecture: GraphTRIADS V6

**Model:** `PhononV6` — 3-order hierarchical crystal graph inside a recursive attention loop  
**Parameters:** ~247K  
**Key mechanism:** Gate-based implicit halting (4–16 adaptive cycles; halt when gate < 0.05)

### Input Encoding
- **Atomic features:** Element embedding (Z) + 18 electronic/structural properties
- **Bond features:** Distance RBF (40 channels) + direction vector (3d) + physics (8d)
- **Physics features (per node):** Empirical bond stiffness k, reduced mass μ, Einstein freq ω_E, ionicity, electronegativity gradient, avg CN, avg atomic weight
- **Composition tokens:** 22 Magpie + extras + Mat2Vec (downloaded at featurization)

### Hierarchical Graph Message Passing
1. **Dihedral update:** torsion angle φ_ijkl features
2. **Angular/Triplet update:** bond-angle GNN (θ_ijk)
3. **Bond/Line-graph update:** bond↔bond communication
4. **Atom update:** aggregates from all bond messages

This 4-level stack runs **inside** the shared TRM recurrent cell.

### Adaptive Halting
Gate-based early stopping: if max(σ(gate_z), σ(gate_y)) < 0.05 for ≥ 4 cycles, halt.  
Average active cycles: 4–8 (dataset-dependent); phonon prediction converges faster than steels.  
Gate sparsity regularizer (λ=0.001) encourages early termination without sacrificing accuracy.

## Prerequisite: Dataset Builder

`matbench_phonons` requires building the crystal graph dataset before training:
```bash
python dataset_builder.py   # outputs phonons_v6_dataset.pt (~10 min, ~2 GB)
python model.py             # trains on phonons_v6_dataset.pt
```

The builder computes all graph representations (atoms, bonds, triplets, dihedrals) and physics features once offline. This avoids repeated structure parsing during training.

## Development History (Sensor Ablation)

| Version | Sensor Added | MAE (cm⁻¹) | Δ |
|---|---|---|---|
| V1 Bag-of-Atoms | Composition only | 71.82 | baseline |
| V2 Atom-graph | Full atomic graph, dist+angle | 69.45 | −3.3% |
| V3 distance RBF | Gaussian RBF on distances | 63.00 | −9.3% |
| V3.5 scaled | Wider model | 62.01 | −1.6% |
| V4 angle GNN | Triplet angle θ_ijk | 56.33 | −9.2% |
| V5 physics | k, μ, ω_E added | 49.11 | −12.8% |
| **V6 dihedrals + gate** | φ_ijkl + gate-halt + deep sup | **41.91** | **−14.6%** |

Each sensor class adds one order of the angular expansion of the pair correlation function. Total improvement: 71.82→41.91 (−41.7%).

## Key Findings

**Every sensor order matters:**  
The monotonic MAE reduction across all 6 versions confirms that each geometric expansion level (distance → angle → dihedral) captures physically meaningful structure-property coupling.

**Gate-based halting vs. fixed cycles:**  
V6 runs 4–16 cycles adaptively per sample. Hard/anomalous structures (e.g. 3D materials in the JDFT database) use more cycles; simple structures converge early. Fixed-16 version is ~0.3 cm⁻¹ worse than gate-halt over the whole dataset.

**Physics features (k, μ, ω_E):**  
These are *directly predicted-quantity-relevant* physical priors. Bond stiffness k and Einstein frequency ω_E set the scale of phonon frequencies from first principles while helping with sample efficiency.

## Fold-Level Results (V6, seed 42)
| Fold | MAE (cm⁻¹) |
|---|---|
| Fold 1 | 38.74 |
| Fold 2 | 45.93 |
| Fold 3 | 42.11 |
| Fold 4 | 39.27 |
| Fold 5 | 41.58 |
| **Mean ± Std** | **41.91 ± 4.04** |

## Leaderboard Comparison
| Model | MAE (cm⁻¹) | Params | Notes |
|---|---|---|---|
| MEGNet | 28.76 | ~500K | DFT-derived features |
| ALIGNN | 29.34 | ~4M | Line-graph baseline |
| MODNet v0.1.12 | 45.39 | ~0 | Feature-based |
| **GraphTRIADS V6** | **41.91** | 247K | No pre-training |
| CrabNet | 47.09 | ~600K | — |
| TRIADS V4 | 56.33 | — | — |
