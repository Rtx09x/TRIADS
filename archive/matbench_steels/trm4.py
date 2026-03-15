"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V4 — Combined Features (Magpie + Mat2Vec) + MLP-TRM     ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV          ║
║                                                                      ║
║  Input: Magpie (132) + Mat2Vec pooled (200) = 332-dim features       ║
║  Target: Beat RF-SCM/Magpie (103.51 MPa) on the leaderboard         ║
║                                                                      ║
║  Models:                                                             ║
║    MLP-Combined-M  : h=128, ff=256  (~350K params)                   ║
║    MLP-Combined-L  : h=192, ff=384  (~700K params)                   ║
║    MLP-Combined-XL : h=256, ff=512  (~1.2M params)                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, copy, json, time, logging, warnings, shutil, urllib.request
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO, format='%(name)s │ %(message)s')
log = logging.getLogger("TRM4")

SEED = 42

BASELINES = {
    'TPOT-Mat (best)':   79.9468,
    'AutoML-Mat':        82.3043,
    'MODNet v0.1.12':    87.7627,
    'RF-SCM/Magpie':    103.5125,
    'CrabNet':          107.3160,
    'MLP-Magpie-L V3':  130.3264,
    'MLP-TRM V2 best':  184.5662,
    'Dummy':            229.7445,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMBINED FEATURIZER (Magpie + Mat2Vec)
# ══════════════════════════════════════════════════════════════════════════════

class CombinedFeaturizer:
    """Compute Magpie (132) + Mat2Vec pooled (200) = 332-dim features.

    Magpie: 22 elemental properties × 6 statistics = 132 compositional descriptors.
    Mat2Vec: Fraction-weighted sum of 200-dim Word2Vec embeddings trained on 3M+
             materials science papers.

    Together they provide complementary info:
    - Magpie: statistical properties (what properties do these elements have?)
    - Mat2Vec: learned chemical semantics (what does the literature say?)
    """

    GCS_BASE = "https://storage.googleapis.com/mat2vec/"
    M2V_FILES = ["pretrained_embeddings",
                 "pretrained_embeddings.wv.vectors.npy",
                 "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache_dir="mat2vec_cache"):
        self.ep = ElementProperty.from_preset("magpie")
        self.magpie_names = self.ep.feature_labels()
        self.n_magpie = len(self.magpie_names)
        self.scaler = None

        # Download and load Mat2Vec
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.m2v = self._download_and_load_m2v()
        self.m2v_embeddings = {w: self.m2v.wv[w] for w in self.m2v.wv.index_to_key}

        self.n_features = self.n_magpie + 200
        log.info(f"Combined featurizer: {self.n_magpie} Magpie + 200 Mat2Vec = {self.n_features} features")

    def _download_and_load_m2v(self):
        for fname in self.M2V_FILES:
            fpath = os.path.join(self.cache_dir, fname)
            if not os.path.exists(fpath):
                log.info(f"  Downloading {fname}...")
                urllib.request.urlretrieve(self.GCS_BASE + fname, fpath)
        return Word2Vec.load(os.path.join(self.cache_dir, "pretrained_embeddings"))

    def _mat2vec_pooled(self, comp):
        """Fraction-weighted sum of Mat2Vec embeddings."""
        vec, total = np.zeros(200, dtype=np.float32), 0.0
        for sym, frac in comp.get_el_amt_dict().items():
            if sym in self.m2v_embeddings:
                vec += frac * self.m2v_embeddings[sym]
                total += frac
        return vec / max(total, 1e-8)

    def featurize_all(self, comps):
        """Returns [N, 332] array (Magpie 132 + Mat2Vec 200)."""
        features = []
        for comp in tqdm(comps, desc="  Combined featurization", leave=False):
            # Magpie
            try:
                magpie = np.array(self.ep.featurize(comp), dtype=np.float32)
            except Exception:
                magpie = np.zeros(self.n_magpie, dtype=np.float32)
            magpie = np.nan_to_num(magpie, nan=0.0)

            # Mat2Vec pooled
            m2v = self._mat2vec_pooled(comp)

            features.append(np.concatenate([magpie, m2v]))
        return np.array(features)

    def fit_scaler(self, features):
        """Fit StandardScaler on training features only."""
        self.scaler = StandardScaler().fit(features)

    def transform(self, features):
        """Apply fitted scaler and handle NaN."""
        if self.scaler is None:
            return features
        return np.nan_to_num(
            self.scaler.transform(features), nan=0.0
        ).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MagpieDataset(Dataset):
    """Pre-computed Magpie feature dataset for MLP models."""
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets  = torch.tensor(np.array(targets, dtype=np.float32))
    def __len__(self):        return len(self.targets)
    def __getitem__(self, i): return self.features[i], self.targets[i]


# ══════════════════════════════════════════════════════════════════════════════
# 3. MLP-TRM MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MLPTRM(nn.Module):
    """
    Dual-state TRM with MLP core.
    Input: Magpie feature vector (132-dim by default).
    """
    def __init__(self, input_dim=132, hidden_dim=64, ff_dim=128,
                 dropout=0.2, steps=16):
        super().__init__()
        self.steps = steps
        self.D     = hidden_dim
        self.arch_type = 'MLP-Magpie'

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU())

        self.z_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim), nn.LayerNorm(hidden_dim))

        self.y_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim), nn.LayerNorm(hidden_dim))

        self.head = nn.Linear(hidden_dim, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_trajectory=False):
        B  = x.size(0)
        xp = self.input_proj(x)
        z  = torch.zeros(B, self.D, device=x.device)
        y  = torch.zeros(B, self.D, device=x.device)
        traj = []
        for _ in range(self.steps):
            z = z + self.z_update(torch.cat([xp, y, z], -1))
            y = y + self.y_update(torch.cat([y, z],     -1))
            if return_trajectory:
                traj.append(self.head(y).squeeze(1))
        out = self.head(y).squeeze(1)
        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 4. STRATIFIED VAL SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def stratified_val_split(targets, val_size=0.15, seed=42):
    """Split targets into train/val indices, stratified by target value."""
    bins   = np.percentile(targets, [25, 50, 75])
    labels = np.digitize(targets, bins)
    train_idx, val_idx = [], []
    rng = np.random.RandomState(seed)
    for b in range(4):
        bin_mask = np.where(labels == b)[0]
        if len(bin_mask) == 0:
            continue
        n_val  = max(1, int(len(bin_mask) * val_size))
        chosen = rng.choice(bin_mask, n_val, replace=False)
        val_idx.extend(chosen.tolist())
        train_idx.extend(np.setdiff1d(bin_mask, chosen).tolist())
    return np.array(train_idx), np.array(val_idx)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(model, train_loader, val_loader, device,
               epochs=300, fold_idx=1, config_name=""):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    best_val  = float('inf')
    best_wts  = copy.deepcopy(model.state_dict())
    history   = {'train': [], 'val': []}
    patience, no_imp = 60, 0

    pbar = tqdm(range(epochs), desc=f"  [{config_name}] F{fold_idx}/5",
                leave=False, ncols=120)

    for epoch in pbar:
        model.train()
        tr_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            preds = model(bx)
            loss  = F.l1_loss(preds, by)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(by)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                preds  = model(bx)
                vl_loss += F.l1_loss(preds, by).item() * len(by)
        vl_loss /= len(val_loader.dataset)

        scheduler.step()
        history['train'].append(tr_loss)
        history['val'].append(vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            best_wts = copy.deepcopy(model.state_dict())
            no_imp   = 0
        else:
            no_imp += 1

        pbar.set_postfix({
            'Tr': f'{tr_loss:.1f}', 'Val': f'{vl_loss:.1f}',
            'Best': f'{best_val:.1f}',
            'LR': f'{scheduler.get_last_lr()[0]:.2e}'
        })

        if no_imp >= patience:
            log.info(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_wts)
    return best_val, model, history


# ══════════════════════════════════════════════════════════════════════════════
# 6. RECURSION TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

def eval_trajectory(model, loader, device):
    model.eval()
    step_preds, targets = None, []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            _, traj = model(bx, return_trajectory=True)
            if step_preds is None:
                step_preds = [[] for _ in range(len(traj))]
            for i, sp in enumerate(traj):
                step_preds[i].extend(sp.cpu().numpy().tolist())
            targets.extend(by.cpu().numpy().tolist())
    t = np.array(targets)
    return [float(np.mean(np.abs(np.array(p) - t))) for p in step_preds]


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    t0 = time.time()
    print("\n" + "═"*72)
    print("  TRM-MatSci V4 │ Magpie + Mat2Vec Combined │ matbench_steels")
    print("  Dataset: 312 samples │ 5-Fold Nested CV │ 332-dim combined features")
    print("═"*72 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        log.info("Device: CPU")

    # ── Load dataset ──────────────────────────────────────────────────────
    log.info("Loading matbench_steels...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_steels")
    comps_raw   = df['composition'].tolist()
    targets_all = np.array(df['yield strength'].tolist(), dtype=np.float32)
    comps_all   = [Composition(c) for c in comps_raw]
    log.info(f"Dataset: {len(df)} samples")

    # ── Precompute ALL combined features once ──────────────────────────────
    log.info("Computing combined Magpie + Mat2Vec features...")
    feat = CombinedFeaturizer()
    all_features = feat.featurize_all(comps_all)
    INPUT_DIM = all_features.shape[1]
    log.info(f"Feature matrix: {all_features.shape} ({INPUT_DIM} features)")

    # ── Official MatBench splits ──────────────────────────────────────────
    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    official_folds = list(kfold.split(comps_all))

    os.makedirs('trm_models_v4', exist_ok=True)

    # ── Model configs ─────────────────────────────────────────────────────
    CONFIGS = [
        ('MLP-Combined-S',  dict(input_dim=INPUT_DIM, hidden_dim=64,
                                  ff_dim=100, dropout=0.2, steps=16)),
        ('MLP-Combined-L',  dict(input_dim=INPUT_DIM, hidden_dim=80,
                                  ff_dim=160, dropout=0.2, steps=16)),
    ]

    all_results, all_histories, all_trajectories = {}, {}, {}

    for cfg_idx, (cfg_name, cfg_kwargs) in enumerate(CONFIGS):
        t_cfg = time.time()
        print(f"\n{'═'*72}")
        print(f"  [{cfg_idx+1:02d}/{len(CONFIGS):02d}]  {cfg_name}")
        print(f"  input={INPUT_DIM}  hidden={cfg_kwargs['hidden_dim']}  "
              f"ff={cfg_kwargs['ff_dim']}  steps=16  dropout=0.2  epochs=300")
        print(f"{'─'*72}")
        _tmp = MLPTRM(**cfg_kwargs)
        log.info(f"Params: {_tmp.count_parameters():,}")
        del _tmp

        fold_maes, fold_histories = [], []
        fold_trajectories, fold_details = [], []

        for fold, (tv_idx, test_idx) in enumerate(official_folds):
            tv_features = all_features[tv_idx]
            tv_targets  = targets_all[tv_idx]
            te_features = all_features[test_idx]
            te_targets  = targets_all[test_idx]

            torch.manual_seed(SEED + fold)

            # Stratified train/val split (LOCAL indices into tv_*)
            tr_idx, vl_idx = stratified_val_split(
                tv_targets, val_size=0.15, seed=SEED + fold)

            train_features = tv_features[tr_idx]
            train_targets  = tv_targets[tr_idx]
            val_features   = tv_features[vl_idx]
            val_targets    = tv_targets[vl_idx]

            # Fit scaler on TRAINING features only → no leakage
            feat.fit_scaler(train_features)
            train_scaled = feat.transform(train_features)
            val_scaled   = feat.transform(val_features)
            test_scaled  = feat.transform(te_features)

            train_ds = MagpieDataset(train_scaled, train_targets)
            val_ds   = MagpieDataset(val_scaled,   val_targets)
            test_ds  = MagpieDataset(test_scaled,  te_targets)

            dl_kw = dict(batch_size=32,
                         pin_memory=(device.type == 'cuda'),
                         num_workers=0)
            train_dl = DataLoader(train_ds, shuffle=True,  **dl_kw)
            val_dl   = DataLoader(val_ds,   shuffle=False, **dl_kw)
            test_dl  = DataLoader(test_ds,  shuffle=False, **dl_kw)

            model = MLPTRM(**cfg_kwargs).to(device)
            best_val, model, history = train_fold(
                model, train_dl, val_dl, device,
                epochs=300, fold_idx=fold+1, config_name=cfg_name)

            # ── Test evaluation ───────────────────────────────────────
            model.eval()
            te_loss = 0.0
            with torch.no_grad():
                for bx, by in test_dl:
                    bx, by = bx.to(device), by.to(device)
                    preds   = model(bx)
                    te_loss += F.l1_loss(preds, by).item() * len(by)
            te_mae = te_loss / len(test_dl.dataset)

            log.info(f"  Fold {fold+1}/5 → Test: {te_mae:.4f}  "
                     f"Val: {best_val:.4f}")
            fold_maes.append(te_mae)
            fold_histories.append(history)
            fold_details.append({'val_mae': best_val, 'test_mae': te_mae})
            fold_trajectories.append(
                eval_trajectory(model, test_dl, device))

            torch.save({
                'config_name': cfg_name, 'fold': fold+1,
                'kwargs': cfg_kwargs,
                'test_mae': te_mae, 'val_mae': best_val,
                'model_state': model.state_dict(), 'history': history,
            }, f'trm_models_v4/{cfg_name}_fold{fold+1}.pt')

        avg_mae = float(np.mean(fold_maes))
        std_mae = float(np.std(fold_maes))
        t_cfg   = time.time() - t_cfg
        tag = ("  ← BEATS TPOT 🏆"   if avg_mae < 79.95  else
               "  ← Beats AutoML ✓"   if avg_mae < 82.30  else
               "  ← Beats MODNet ✓"   if avg_mae < 87.76  else
               "  ← Beats CrabNet ✓"  if avg_mae < 107.32 else
               "  ← Beats V2 MLP ✓"   if avg_mae < 184.57 else "")
        print(f"\n  ✓ {cfg_name}: {avg_mae:.4f} ± {std_mae:.4f} MPa{tag}"
              f"  (took {t_cfg/60:.1f} min)")

        all_results[cfg_name] = {
            'avg_mae': avg_mae, 'std_mae': std_mae,
            'fold_maes': fold_maes, 'fold_details': fold_details,
            'params': MLPTRM(**cfg_kwargs).count_parameters(),
        }
        all_histories[cfg_name]    = fold_histories
        all_trajectories[cfg_name] = np.mean(
            fold_trajectories, axis=0).tolist()

    # ── Final leaderboard ─────────────────────────────────────────────────
    t_total = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V4 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'Params':>9} {'MAE(MPa)':>10} {'±Std':>8}")
    print(f"  {'─'*52}")
    for name, r in sorted(all_results.items(),
                           key=lambda x: x[1]['avg_mae']):
        print(f"  {name:<22} {r['params']:>9,} "
              f"{r['avg_mae']:>10.4f} {r['std_mae']:>8.4f}")
    print(f"  {'─'*52}")
    for bname, bval in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bname:<22} {'baseline':>9} {bval:>10.4f}")
    print(f"\n  Total time: {t_total/60:.1f} minutes\n")

    generate_plots(all_results, all_trajectories, all_histories)
    save_summary(all_results, all_trajectories, t_total)
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 8. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    'MLP-Combined-S':  '#1565C0',
    'MLP-Combined-L':  '#0D47A1',
}

# Previous best results for comparison
PREV_RESULTS = {
    'Magpie-L (V3)':     {'mae': 130.3264, 'std': 12.9266},
    'Magpie-S (V3)':     {'mae': 138.4033, 'std': 18.2472},
    'MLP-V2-L':          {'mae': 184.5662, 'std': 11.2077},
}


def generate_plots(results, trajectories, histories):
    names  = list(results.keys())
    maes   = [results[n]['avg_mae'] for n in names]
    stds   = [results[n]['std_mae'] for n in names]
    params = [results[n]['params']  for n in names]
    colors = [PALETTE.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30)

    # ── Panel 1: Main bar chart with V2 comparison ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    all_names  = names + list(PREV_RESULTS.keys())
    all_maes   = maes + [PREV_RESULTS[n]['mae'] for n in PREV_RESULTS]
    all_stds   = stds + [PREV_RESULTS[n]['std'] for n in PREV_RESULTS]
    all_colors = colors + ['#90CAF9', '#BBDEFB', '#EF9A9A']

    bars = ax1.bar(all_names, all_maes, yerr=all_stds, capsize=5,
                   color=all_colors, alpha=0.88,
                   edgecolor='white', linewidth=1.2)
    for bv, col, ls, lbl in [
        (79.95,  '#2E7D32', '--', 'TPOT (79.95)'),
        (87.76,  '#F57F17', '--', 'MODNet (87.76)'),
        (103.51, '#795548', ':',  'RF-SCM/Magpie (103.51)'),
        (107.32, '#9E9E9E', ':',  'CrabNet (107.32)'),
    ]:
        ax1.axhline(bv, color=col, linestyle=ls, linewidth=1.8,
                    label=lbl, alpha=0.85)
    for bar, m, s in zip(bars, all_maes, all_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 2,
                 f'{m:.1f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylabel('MAE (MPa)', fontsize=11)
    ax1.set_title('TRM-MatSci V4 (Magpie+Mat2Vec) vs Previous │ matbench_steels',
                  fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(all_maes) * 1.25)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20,
             ha='right', fontsize=9)

    # ── Panel 2: Recursion convergence ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for n, traj in trajectories.items():
        ax2.plot(range(1, len(traj)+1), traj, label=n,
                 color=PALETTE.get(n, '#888'), linewidth=2.0, alpha=0.9)
    for bv, col, ls, lbl in [
        (107.32, '#795548', ':', 'CrabNet'),
        (87.76,  '#F57F17', '--', 'MODNet'),
        (79.95,  '#2E7D32', '--', 'TPOT'),
    ]:
        ax2.axhline(bv, color=col, linestyle=ls, linewidth=1.5,
                    label=lbl, alpha=0.8)
    ax2.set_xlabel('Recursion Step (1 → 16)', fontsize=10)
    ax2.set_ylabel('MAE (MPa)', fontsize=10)
    ax2.set_title('Recursion Convergence', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3)

    # ── Panel 3: Training curves per fold ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for n in names:
        for fold_i, h in enumerate(histories[n]):
            lbl = f'{n} F{fold_i+1}' if fold_i == 0 else None
            ax3.plot(h['train'], alpha=0.25, linewidth=0.8,
                     color='#1565C0')
            ax3.plot(h['val'], alpha=0.7, linewidth=1.0,
                     color=PALETTE.get(n, '#888'), label=lbl)
    ax3.axhline(107.32, color='#795548', linestyle=':',
                linewidth=1.2, label='CrabNet')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('MAE (MPa)', fontsize=10)
    ax3.set_title('Training Curves (Val MAE per Fold)',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(alpha=0.2)

    plt.suptitle(
        'TRM-MatSci V4 │ Magpie + Mat2Vec Combined │ matbench_steels',
        fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v4.png', dpi=150, bbox_inches='tight')
    log.info("✓ Saved: trm_results_v4.png")
    plt.close(fig)

    # ── Detailed training curves (separate figure) ────────────────────────
    n_models = len(names)
    fig2, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
    fig2.suptitle('V3 Training Curves — Train (blue) vs Val (colored)',
                  fontsize=13, fontweight='bold')
    if n_models == 1:
        axes = [axes]
    for ax, n in zip(axes, names):
        for fold_i, h in enumerate(histories[n]):
            ax.plot(h['train'], alpha=0.3, linewidth=1, color='#1565C0')
            ax.plot(h['val'],   alpha=0.85, linewidth=1.2,
                    color=PALETTE.get(n, '#888'), label=f'F{fold_i+1}')
        ax.axhline(107.32, color='#795548', linestyle=':',
                   linewidth=1.2, label='CrabNet')
        ax.set_title(n, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (MPa)')
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.2)
    plt.tight_layout()
    fig2.savefig('trm_curves_v4.png', dpi=130, bbox_inches='tight')
    plt.close(fig2)
    log.info("✓ Saved: trm_curves_v4.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(results, trajectories, total_s):
    summary = {
        'version': 'V4',
        'task': 'matbench_steels',
        'input_type': 'magpie_132_plus_mat2vec_200_combined_332',
        'recursion_steps': 16, 'epochs': 300, 'dropout': 0.2,
        'total_train_min': round(total_s / 60, 1),
        'baselines': BASELINES,
        'models': {}
    }
    for name, r in results.items():
        summary['models'][name] = {
            'params':       r['params'],
            'avg_mae':      round(r['avg_mae'], 4),
            'std_mae':      round(r['std_mae'], 4),
            'fold_maes':    [round(m, 4) for m in r['fold_maes']],
            'fold_details': r['fold_details'],
            'convergence':  [round(v, 4) for v in trajectories[name]],
            'beats_tpot':   r['avg_mae'] < 79.9468,
            'beats_modnet': r['avg_mae'] < 87.7627,
            'beats_crabnet': r['avg_mae'] < 107.316,
            'beats_v2':     r['avg_mae'] < 184.57,
        }
    with open('trm_models_v4/summary_v3.json', 'w') as f:
        json.dump(summary, f, indent=2)

    rows = []
    for name, r in results.items():
        for fi, (fm, fd) in enumerate(
                zip(r['fold_maes'], r['fold_details'])):
            rows.append({
                'model': name, 'fold': fi+1,
                'test_mae': round(fm, 4),
                'val_mae':  round(fd['val_mae'], 4)
            })
    pd.DataFrame(rows).to_csv(
        'trm_models_v4/fold_results_v3.csv', index=False)
    log.info("✓ Saved: trm_models_v4/summary_v3.json + fold_results_v3.csv")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_benchmark()

    # Create zip for easy download
    shutil.make_archive("trm_v4_all", "zip", "trm_models_v4")
    log.info("✓ Created trm_v4_all.zip for download")
