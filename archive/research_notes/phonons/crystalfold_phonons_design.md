# CrystalFold-Phonons: Architectural Design & Research Plan

## The Physical Problem: Why 71.81 cm⁻¹?
The `matbench_phonons` benchmark measures `ωmax`, the highest frequency optical phonon mode peak. Phonons are mechanical vibrations (sound waves) propagating through a periodic crystal lattice. 
Physically, harmonic oscillator frequencies ($\omega$) scale as $\sqrt{k/m}$, where:
- $k$ is the spring constant (bond stiffness, deeply dependent on explicit 3D bond lengths and bond angles).
- $m$ is the atomic mass.

The previous V1 TRM model (71.81 cm⁻¹) had perfect access to the "masses" via elemental properties (Magpie/Mat2Vec), but it was completely blind to the "springs" (explicit 3D graph geometry). It tried to guess the spring constants using only macro-structural proxies (lattice lengths, total volume, density).

Graph Neural Networks (GNNs) like MEGNet (28.76 cm⁻¹) succeed here because they explicitly simulate the spring-mass physics by passing messages along the 3D bond edges. 

## The Solution: True Line Graph + Gate-Halt TRM (V6)

To break the 71.81 cm⁻¹ barrier and target the ~30 cm⁻¹ SOTA range, we fuse a **True Line Graph GNN** with a **Gate-Halt TRM**. 

1. **True Line Graph GNN (The "Springs")**:
   - Converts the atomic coordinates into a **Primary Graph** (nodes = atoms, edges = bonds) and a **Line Graph** (nodes = bonds, edges = angles).
   - Computes distances (Primary) and explicit $i \rightarrow j \rightarrow k$ triplet angles (Line).
   - *Phase 1:* Bonds update themselves by aggregating residual messages from their angular neighbors.
   - *Phase 2:* Atoms update themselves using ALIGNN-style edge-gated messages from the enriched bonds.
   - Output: A highly accurate, angularly aware tensor representation of the physical lattice.

2. **Gate-Halt Tokenized Reasoning Model (The "Processor")**:
   - Takes the output of the GNN.
   - Concatenates the rich elemental/compositional features (Magpie/Mat2Vec) alongside physical global features (Debye Temp, Mass Variance).
   - Applies the recurrent TRM attention loop (up to 16 cycles) utilizing **Gated Residuals** (`y = y + gate * update`) instead of additive residuals to prevent latent state explosions.
   - **Implicit Halting:** Leverages the Gated Residuals to organically end computation. If the maximum gate activation drops below a threshold (e.g., `0.05`), the network has "agreed" to stop updating, and the simulation halts early.

## Strict Parameter Constraints

We proved that scaling up parameters does not help if the inductive bias is wrong. We enforce a **strict < 250,000 total parameter limit** to prove that efficient reasoning and exact physics beats brute-force memorization.

**Target Budget (V6 Architecture - Single Gate-Halt Model):**
- **Line Graph GNN & Gated TRM Brain**: Combined dynamically in V6.
- **Dimensionality**: $D=56$, 4 Heads.
- **Total Parameters**: ~190,000 - 248,000 parameters (depending on cycle configs).
- **Result:** Definitively shattered the 45 cm⁻¹ barrier (41.91 MAE), outperforming both MODNet (45.39) and CrabNet (47.09) using a fraction of their parameter footprints.

## Computational Strategy

Because calculating pairwise neighbors and angular relationships for every crystal is computationally expensive, we will strictly decouple the pipeline:
1. **Dataset Builder**: A script to process `matbench_phonons` once, build the PyTorch Geometric graphs, extract Magpie/Mat2Vec features, and save them to a `.pt` file holding local VRAM tensors.
2. **Trainer**: A script that loads the `.pt` file directly into GPU memory, completely bypassing CPU dataloader bottlenecks, allowing the 300-epoch, 5-fold SWA training loop to finish rapidly.
