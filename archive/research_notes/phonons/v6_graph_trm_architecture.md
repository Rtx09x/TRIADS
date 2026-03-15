# TRIADS V6: The Graph-TRM Architecture with Gate-Based Halting

This document details the complete end-to-end architecture, physical ideation, dataset construction, and algorithmic breakthroughs comprising the latest **V6 Gate-Halt Graph-TRM** model for phonon prediction (specifically `matbench_phonons`).

## 1. The Core Physical Philosophy

Why did the original TRIADS TRM effortlessly achieve state-of-the-art results on metallic properties, band gaps, and yield strengths, but struggle on phonons (71.82 cm⁻¹)?

The answer lies in the fundamental physics of the property:
*   **Electronic/Thermodynamic Properties** (e.g., band gap, metallicity) are largely determined by **compositional interactions and orbital mixing**. A "Bag-of-Atoms" Transformer (like the original TRM) handles this flawlessly because it learns to attend to valence mismatches and elemental synergy without needing exact 3D geometry.
*   **Vibrational Properties** (Phonons) are fundamentally mechanical waves propagating through a periodic lattice. The frequency of a classical harmonic oscillator ($\omega = \sqrt{k/\mu}$) depends entirely on the exact **masses** ($\mu$) and exact **spring constants** ($k$). 

The original TRM was trying to predict spring constants while entirely blind to the actual physical topology (bond distances and angles) of the crystal. To fix this, we had to fuse the compositional reasoning of the TRM with a physical network capable of explicitly simulating multi-body spatial kinematics holding the crystal together.

---

## 2. Deep Physics Featurization (Dataset Layer)

Instead of forcing a multi-million parameter model to learn fundamental physics from generic coordinates (like ALIGNN or MEGNet), we pre-compute the physical kinematics directly into the dataset tensors. 

We model every crystal as a **3-Order Hierarchical Graph**:
1.  **Nodes (Atoms)**: Element identity, MAGPIE properties, mat2vec embeddings, fractional coordinates, coordination numbers, and Voronoi volumes.
2.  **Edges (Bonds)**: 12-nearest neighbors (8Å cutoff). Features include:
    *   Explicit 3D direction vectors ($dx, dy, dz$) and Gaussian distance expansion.
    *   **Estimated Bond Stiffness ($k$)**: $k \sim \chi_{avg} / r^2$, an empirical force constant derived from electronegativity and length.
    *   **Reduced Mass ($\mu$)**: $(\mu = \frac{m_1 m_2}{m_1 + m_2})$.
    *   **Estimated Einstein Frequency ($\omega_E$)**: $\sqrt{k/\mu}$, representing an initialized zero-order guess of the phonon frequency.
    *   **Bond Ionicity**: Difference in electronegativities.
3.  **Angles (Triplets)**: Edges connecting two bonds that share a central atom. This forms a "Line Graph" capturing explicit bond angles, crucial for transverse shear vibrations.

We also compute **Global Physics Context**: Estimated Debye Temperature, average stiffness, mass variance (which causes phonon scattering), and overall packing fraction.

---

## 3. The Architecture: Graph-TRM (The Brain)

The V6 architecture fundamentally transforms Graph Neural Networks by wrapping them inside the recurrent reasoning loop of the Tokenized Reasoning Model (TRM). Instead of passing messages forward through fixed independent graph layers, the Graph-TRM simulates the lattice settling into an equilibrium state over $T$ abstract "time steps" (reasoning cycles).

The architecture receives:
*   **Atom Embeddings** ($N_{atoms}$)
*   **Bond Embeddings** ($N_{bonds}$)
*   **Composition/Global Tensors** (MAGPIE, physics globals)

### The Recurrent Loop (Cycles $t = 1 \dots T_{max}$)

During each cycle, the model executes a strict unrolled sequence (with shared weights across time):

#### Phase 1 & 2: True Line Graph Message Passing (Simulating Springs)
1.  **Angle-to-Bond Updates:** Bond representations are updated by sweeping over the angles between them. This teaches the model how a bond flexes when a neighboring bond pulls on it.
2.  **Bond-to-Atom Updates:** Atom representations are updated by sweeping over their connected bonds (which now contain angular context). We use **ALIGNN-style Edge-Gated Convolutions**, where the bond embedding acts as a dynamic gate on the incoming atom message.

#### Phase 3 & 4: Cross-Attention (Compositional Context)
The physically updated graph atoms now perform Multi-Head Cross-Attention against the global composition tokens (MAGPIE + globals). 
*   *Physical Meaning*: The local vibrating atom queries the entire crystal's chemical landscape (e.g., "I know my neighbor is Carbon, but what is the overall metallic sea of electrons like?").

#### Phase 5: Gated Residual Latent Evolution
The information from the graph pass and the attention pass must be merged into the persistent latent state representing the atom's vibration.
*   *The Problem*: Earlier versions (V5) used additive residuals ($y_{t+1} = y_t + \Delta$). Because we run up to 16 cycles, the latent state magnitudes exploded, causing `NaN` gradients and poor early stabilization.
*   *The Solution — Gated Update*: We introduced GRU-style learned gates.
    $$g_y = \sigma(\text{Linear}([y_t, \Delta_t]))$$
    $$y_{t+1} = y_t + g_y \cdot \text{Update}(\dots)$$
    This stabilizes deep reasoning by allowing the model to smoothly "close the gate" when an atom's state has converged.

---

## 4. The Breakthrough: Implicit Gate-Based Halting

We do not know in advance how many "reasoning cycles" a specific crystal needs. Simple crystals (NaCl) might solve instantly, while complex disordered alloys might need deep simulation.

*   *Old Approach (Adaptive Recursive Halting / ARH)*: Required a separate physical network to predict a halting probability, accumulating a "ponder cost" penalty. Very hard to tune, sensitive to $\lambda$ regularization.
*   *V6 Approach (Gate-Halt)*: We realized the **Gated Residuals already contain the halting signal**. 
    If all $N$ atoms in the graph compute $g_y \approx 0$, they are mathematically refusing to update their state. They have reached equilibrium.

At the end of each cycle $t \ge t_{min} (4)$, we check the maximum gate activation across all atoms in the batch:
```python
if gate_activations.max() < 0.05:
    break  # Halts the simulation organically!
```
To encourage efficiency, our loss function adds a simple **Gate Sparsity Regularizer**:
$$\mathcal{L}_{Total} = \mathcal{L}_{MAE} + \gamma \cdot \text{mean}(gates)$$
This actively pushes the network toward early completion without needing complex explicit halting layers. 

---

## 5. Deep Supervision 

Because the TRM outputs a valid prediction at *every* cycle step $t$, we do not just supervise the final step. We calculate the output at every step $p_1, p_2, \dots p_t$ and sum the loss over the trajectory:
$$\mathcal{L}_{MAE} = \sum_{t=1}^{T} w_t |p_t - Target|$$
Where $w_t$ linearly increases (later cycles carry more weight). This provides massively dense gradients to the single shared set of parameters, allowing a tiny network to learn profound physics.

---

## 6. Final Results & Parameter Efficiency

By keeping the dimensionality tight ($D=56$, 4 attention heads) and radically reusing weights across the time dimension, we proved that strict inductive biases crush raw parameter counts:

*   **Total Parameters**: $\approx 199,000$ (Graph Message Passing + TRM Attention + Regression Head)

*   **Final Verified MAE**: **41.91 (Fixed 16 cycles)** / **43.29 (Adaptive Gate-Halt 4-16 cycles)**

*   **Conclusion**: We successfully broke the 45 cm⁻¹ barrier on a single Kaggle T4 GPU in under an hour, decisively beating heavily parameterized baselines like **CrabNet (47.09)** and **MODNet (45.39)**.

The V6 Graph-TRM represents a highly efficient synthesis of explicit 3D message-passing (for precise mechanical kinematics) and transformer-based global reasoning (for compositional context), governed by organic Gate-Based Halting.
