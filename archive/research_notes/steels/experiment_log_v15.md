# TRM-MatSci Experiment Log: V15

**Date:** March 2026
**Objective:** Test the Hierarchical Tiny Reasoning Model (HTRM) architecture inspired by arXiv:2506.21734, using a separated H-module (Attention) and L-module (MLP-TRM) with detached gradients to steer reasoning.

## Results: V15 HTRM (Catastrophic Failure)

**Final Score: 431.86 ± 49.59 MPa**

| Fold | V15-HTRM | V14A-Flat (Baseline) |
| :---: | :---: | :---: |
| 1 | 367.70 | 122.25 |
| 2 | 412.33 | 82.77 |
| 3 | 422.00 | 85.37 |
| 4 | 519.40 | 94.27 |
| 5 | 437.91 | 90.04 |

## Key Findings & Takeaways

### 1. The Architecture Collapsed
The HTRM architecture failed completely. The 431 MPa error is worse than our V1 dummy baselines (229 MPa). Detaching gradients between H-cycles or forcing attention to re-evaluate without proper BPTT severely destabilized training on this small dataset.

### 2. We Have Reached the Absolute Peak
The project officially peaks at **91.20 MPa** (V13A 5-seed Ensemble) and **94.94 MPa** (V14A Single Seed). Every attempt to add structural complexity beyond V13/V14 has yielded severe diminishing returns or catastrophic failure. 

### 3. Pivot to Meta-Analysis
With ~200 models trained across 15 major versions, the experimental phase is concluded. The focus now shifts entirely to meta-analysis: analyzing the trajectory from 184 MPa down to 91 MPa, understanding the exact mechanisms of our successes (Deep Supervision, Mega Features, Ensembling), and preparing the research for publication.
