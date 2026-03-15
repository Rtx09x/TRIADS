# matbench_expt_gap — V4 Plan (Fresh Route)

## Status
- V2 (100K sweet spot) → ARCHIVED
- V3 (218K proven arch) → ARCHIVED  
- V4 → **NEW**: Better featurization + 3 model sizes

## 12 Models: 3 Sizes × 4 Configs

| Config | Steps | Dropout | ~87K | ~100K | ~218K |
|--------|-------|---------|------|-------|-------|
| S16-D15 | 16 | 0.15 | ✅ | ✅ | ✅ |
| S16-D20 | 16 | 0.20 | ✅ | ✅ | ✅ |
| S20-D15 | 20 | 0.15 | ✅ | ✅ | ✅ |
| S20-D20 | 20 | 0.20 | ✅ | ✅ | ✅ |

### Size Definitions
- **~87K**: d_attn=36, d_hidden=54, ff_dim=84 (smaller than V2's 100K)
- **~100K**: d_attn=36, d_hidden=72, ff_dim=112 (V2's original size)
- **~218K**: d_attn=64, d_hidden=96, ff_dim=150 (V1/V3 proven arch)

## Key Change: Better Featurization
- This is a different dataset (band gaps, not steel yield strength)
- Need dataset-specific features that capture electronic structure better
- Reference our steels runs for proven featurization approach
- More features → better input representation → better results

## Featurization Leakage Check
✅ NO leakage — featurizing whole dataset is safe because:
- All features (Magpie, Mat2Vec, ElementFraction, etc.) are deterministic lookups
- StandardScaler is fitted ONLY on train split each fold
- No information flows from test labels into features
