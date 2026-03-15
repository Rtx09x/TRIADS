
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

imgs_dir = r'e:\Coding\TRM Material Science\matbench_steels\Research\Images'

# ============================================================
# FIGURE 1: Version progression MAE
# ============================================================
versions = ['V1','V2','V3','V4','V5A','V6C','V7B','V8A','V9A','V10A','V10.1\nmean','V11B','V12A','V13A\nens.','V14A']
maes     = [184.38, 184.57, 130.33, 131.63, 128.98, 129.04, 127.08, 143.03, 134.59, 103.28, 105.85, 102.30, 95.99, 91.20, 94.94]
colors = []
for m in maes:
    if m > 130: colors.append('#e07b39')
    elif m > 103: colors.append('#59a14f')
    else: colors.append('#2196F3')

baselines = {'TPOT-Mat': 79.95, 'MODNet': 87.76, 'RF-SCM/Magpie': 103.51, 'CrabNet': 107.31, 'Darwin': 123.29}
bline_colors = {'TPOT-Mat':'#c0392b','MODNet':'#8e44ad','RF-SCM/Magpie':'#27ae60','CrabNet':'#16a085','Darwin':'#d35400'}

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(versions))
bars = ax.bar(x, maes, color=colors, zorder=3)

for bl, val in baselines.items():
    ax.axhline(val, linestyle='--', linewidth=1.2, color=bline_colors[bl], alpha=0.85, zorder=2)
    ax.text(len(versions)-0.5, val+1.5, bl, fontsize=8, color=bline_colors[bl], ha='right', va='bottom')

for bar, mae in zip(bars, maes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{mae:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=9)
ax.set_ylabel('MAE (MPa)', fontsize=12)
ax.set_title('TRIADS: Version-wise Best MAE Progression (V1 — V14)', fontsize=14, fontweight='bold')
ax.set_ylim(60, 200)
ax.grid(axis='y', alpha=0.3, zorder=0)

patch_pre = mpatches.Patch(color='#e07b39', label='Pre-breakthrough (>130 MPa)')
patch_mid = mpatches.Patch(color='#59a14f', label='Post-Magpie phase (103-130 MPa)')
patch_sota = mpatches.Patch(color='#2196F3', label='Post-DeepSupervision SOTA (<103 MPa)')
ax.legend(handles=[patch_pre, patch_mid, patch_sota], fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, 'version_progression_mae.png'), dpi=150, bbox_inches='tight')
plt.close()
print('version_progression_mae.png done')

# ============================================================
# FIGURE 2: Fold heatmap for key models
# ============================================================
models = ['V7B\n(16s,noDS)', 'V9A\n(20s,noDS)', 'V10A\n(20s,DS)', 'V11B\n(scaled+DS)', 'V12A\n(exp+DS)', 'V13A\n(ens.)', 'V14A\n(mega)']
fold_data = np.array([
    [124.56, 153.03, 104.59, 143.42, 109.78],
    [116.32, 146.07, 142.55, 136.70, 131.31],
    [118.67,  95.32,  91.57, 112.65,  98.23],
    [118.82, 101.79,  95.60,  99.82,  95.48],
    [114.71,  82.75,  97.48,  94.07,  90.95],
    [114.32,  81.46,  80.55,  90.49,  89.18],
    [122.25,  82.77,  85.37,  94.27,  90.04],
])
means = fold_data.mean(axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios':[3,1]})

im = ax1.imshow(fold_data, cmap='RdYlGn_r', vmin=75, vmax=175, aspect='auto')
ax1.set_xticks(range(5))
ax1.set_xticklabels([f'Fold {i+1}' for i in range(5)], fontsize=10)
ax1.set_yticks(range(len(models)))
ax1.set_yticklabels(models, fontsize=9)
ax1.set_title('Per-Fold MAE Heatmap (MPa) — Key TRIADS Models', fontsize=12, fontweight='bold')
for i in range(len(models)):
    for j in range(5):
        val = fold_data[i,j]
        col = 'white' if val > 140 else 'black'
        ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8, color=col, fontweight='bold')
plt.colorbar(im, ax=ax1, label='MAE (MPa)')

colors_bar = ['#e07b39' if m > 130 else '#59a14f' if m > 103 else '#2196F3' for m in means[::-1]]
ax2.barh(range(len(models)), means[::-1], color=colors_bar)
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models[::-1], fontsize=9)
ax2.set_xlabel('Mean MAE (MPa)', fontsize=10)
ax2.set_title('Mean MAE', fontsize=11, fontweight='bold')
ax2.axvline(103.51, linestyle='--', color='#27ae60', linewidth=1.2, label='RF-SCM')
ax2.axvline(87.76, linestyle='--', color='#8e44ad', linewidth=1.2, label='MODNet')
ax2.legend(fontsize=8)
for i, m in enumerate(means[::-1]):
    ax2.text(m+0.5, i, f'{m:.1f}', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, 'fold_heatmap_key_models.png'), dpi=150, bbox_inches='tight')
plt.close()
print('fold_heatmap_key_models.png done')

# ============================================================
# FIGURE 3: V10 convergence + halting stats
# ============================================================
conv = [1306.01, 1191.57, 1077.12, 962.67, 848.22, 733.77, 619.33, 504.90, 391.67, 293.57, 220.94, 177.79, 150.90, 129.97, 117.02, 114.53, 111.84, 108.00, 104.88, 103.29]
steps = list(range(1, 21))

halt_folds = {
    'Fold 1': (20.0, 0.0),
    'Fold 2': (15.94, 82.5),
    'Fold 3': (14.50, 95.2),
    'Fold 4': (14.94, 90.3),
    'Fold 5': (14.48, 91.9),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(steps, conv, 'o-', color='#2196F3', linewidth=2.5, markersize=6, label='V10A Fixed-20 DS')
ax1.axhline(103.51, linestyle='--', color='#27ae60', linewidth=1.5, label='RF-SCM/Magpie (103.51)')
ax1.axhline(107.31, linestyle='--', color='#16a085', linewidth=1.5, label='CrabNet (107.31)')
ax1.axhline(123.29, linestyle='--', color='#d35400', linewidth=1.5, label='Darwin (123.29)')
ax1.set_xlabel('Recursion Step', fontsize=12)
ax1.set_ylabel('Aggregate MAE (MPa)', fontsize=12)
ax1.set_title('V10A Deep-Supervised Convergence\nacross 20 Recursion Steps', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

fold_names = list(halt_folds.keys())
avg_steps = [v[0] for v in halt_folds.values()]
pct_early = [v[1] for v in halt_folds.values()]

x2 = np.arange(len(fold_names))
w = 0.35
b1 = ax2.bar(x2 - w/2, avg_steps, w, label='Avg halt step', color='#2196F3', alpha=0.8)
ax3 = ax2.twinx()
b2 = ax3.bar(x2 + w/2, pct_early, w, label='% early halt', color='#FF9800', alpha=0.8)
ax2.axhline(16, linestyle='--', color='gray', linewidth=1, label='V7B depth (16)')
ax2.set_xticks(x2)
ax2.set_xticklabels(fold_names, fontsize=9)
ax2.set_ylabel('Avg Halt Step', fontsize=11)
ax3.set_ylabel('% Samples Halted Early', fontsize=11)
ax2.set_title('V10B Adaptive Halting\nBehavior by Fold', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, 'v10_halting_stats.png'), dpi=150, bbox_inches='tight')
plt.close()
print('v10_halting_stats.png done')

# ============================================================
# FIGURE 4: V13 per-seed variance
# ============================================================
seeds = ['Seed 42', 'Seed 123', 'Seed 7', 'Seed 0', 'Seed 99']
per_seed_v13a = [98.72, 96.77, 98.76, 102.55, 102.11]
ensemble_mae = 91.20

fig, ax = plt.subplots(figsize=(10, 5))
x4 = np.arange(len(seeds))
bars4 = ax.bar(x4, per_seed_v13a, color='#4e79a7', alpha=0.85, width=0.55, label='Single-seed MAE')
ax.axhline(ensemble_mae, linestyle='-', color='#c0392b', linewidth=2.5, label=f'5-Seed Ensemble = {ensemble_mae} MPa')
ax.axhline(95.99, linestyle='--', color='#27ae60', linewidth=1.5, label='V12A Best Single (95.99)')
ax.axhline(87.76, linestyle='--', color='#8e44ad', linewidth=1.5, label='MODNet (87.76)')
ax.set_xticks(x4)
ax.set_xticklabels(seeds, fontsize=11)
ax.set_ylabel('MAE (MPa)', fontsize=12)
ax.set_title('V13A: Per-Seed MAE vs 5-Seed Ensemble\nVariance Reduction = 5.57 MPa', fontsize=12, fontweight='bold')
for bar, val in zip(bars4, per_seed_v13a):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(80, 110)
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, 'v13_seed_variance.png'), dpi=150, bbox_inches='tight')
plt.close()
print('v13_seed_variance.png done')

# ============================================================
# FIGURE 5: V1 MLP recursion convergence
# ============================================================
mlp_conv = [1403.69, 1362.70, 1293.91, 1205.98, 1110.42, 1012.79, 914.53, 816.01, 717.35, 618.60, 524.07, 433.27, 346.30, 269.35, 208.13, 184.38]
trans_conv = [583.93, 583.63, 583.57, 583.56, 583.55, 583.55, 583.55, 583.55, 583.55, 583.55, 583.55, 583.55, 583.55, 583.55, 583.55, 583.55]
steps16 = list(range(1, 17))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(steps16, mlp_conv, 'o-', color='#2196F3', linewidth=2.5, markersize=7, label='MLP-TRM-100K-h64 (Best MLP)')
ax.plot(steps16, trans_conv, 's--', color='#e74c3c', linewidth=2, markersize=6, label='Trans-TRM-h128 (all 3 configs identical)')
ax.axhline(229.74, linestyle=':', color='gray', linewidth=1.5, label='Dummy baseline (229.7)')
ax.set_xlabel('Recursion Step', fontsize=12)
ax.set_ylabel('Mean MAE (MPa)', fontsize=12)
ax.set_title('V1: Recursive Convergence — Each Step Refines the Prediction\n(Direct Empirical Evidence for the TRM Mechanism)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, 'v1_recursive_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print('v1_recursive_convergence.png done')

print('All 5 figures generated successfully!')
