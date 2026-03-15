# TRIADS Classification Research Direction

This document tracks the generalization of the DeepHybridTRM architecture to classification tasks within the Matbench suite.

## Status Summary

| Dataset | Phase | Focus | Best ROCAUC | Status | Key Finding |
|---------|-------|-------|:----------:|--------|-------------|
| is_metal| V1 | Classification | **0.9655** | **🏆 #1** | Massively destroys AMMExpress SOTA (0.9209). 100K params + compositional analytic features (BandCenter/Gap). |
| glass | V1 | Classification | *TBD (~0.94+)* | ⏳ Running | Awaiting final ensemble. Currently tracking at #2 spot over AMMExpress (0.8607), chasing MODNet (0.96). |

### Current Objectives
1.  **Finalize `matbench_glass`**: Collect the final 5-fold ensemble predictions and determine the precise ranking.
2.  **Cross-Benchmark Synthesis**: Evaluate how a single generalized architecture (DeepHybridTRM) achieved Top 3 finishes across 5 vastly different material property datasets (Steels, Expt Gap, JDFT2D, Is Metal, Glass).
3.  **Future Architecture**: The 44K-100K reasoning loop is brutally efficient. Future work will determine if scaling to 500K+ parameters provides any benefit on larger classification datasets, or if the model simply requires more nuanced representation vectors. 
