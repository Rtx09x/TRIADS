"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V10.1 — Reproducibility Test (Multi-Seed)               ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV          ║
║                                                                      ║
║  PURPOSE: Confirm V10A's 103.28 MPa is not a lucky seed.            ║
║  Run the EXACT same architecture 3× with different seeds:           ║
║    Seed A = 42  (original V10 seed)                                  ║
║    Seed B = 123                                                      ║
║    Seed C = 7                                                        ║
║                                                                      ║
║  Architecture: V10A Deep-Supervised Hybrid-TRM (unchanged)          ║
║  d_attn=48, nhead=2, d_hidden=64, ff=100, 20 steps                 ║
║  Deep supervision + SWA@200 + 300 epochs                            ║
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
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO, format='%(name)s │ %(message)s')
log = logging.getLogger("TRM10.1")

SEEDS = [42, 123, 7]  # Three seeds for reproducibility
SEED_NAMES = ['Seed-42', 'Seed-123', 'Seed-7']

BASELINES = {
    'TPOT-Mat':       79.9468,
    'AutoML-Mat':     82.3043,
    'MODNet':         87.7627,
    'RF-SCM/Magpie': 103.5125,
    'CrabNet':       107.3160,
    'Darwin':        123.2932,
    'V10A (orig)':   103.2867,
}


# ══════════════════════════════════════════════════════════════════════
# 1. FEATURIZER + DATASET (identical to V10)
# ══════════════════════════════════════════════════════════════════════

class CombinedFeaturizer:
    GCS = "https://storage.googleapis.com/mat2vec/"
    FILES = ["pretrained_embeddings",
             "pretrained_embeddings.wv.vectors.npy",
             "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache="mat2vec_cache"):
        self.ep = ElementProperty.from_preset("magpie")
        self.n_mg = len(self.ep.feature_labels())
        self.scaler = None
        os.makedirs(cache, exist_ok=True)
        for f in self.FILES:
            p = os.path.join(cache, f)
            if not os.path.exists(p):
                log.info(f"  Downloading {f}...")
                urllib.request.urlretrieve(self.GCS + f, p)
        self.m2v = Word2Vec.load(os.path.join(cache, "pretrained_embeddings"))
        self.emb = {w: self.m2v.wv[w] for w in self.m2v.wv.index_to_key}
        log.info(f"Features: {self.n_mg} Magpie + 200 Mat2Vec = {self.n_mg+200}d")

    def _pool(self, c):
        v, t = np.zeros(200, np.float32), 0.0
        for s, f in c.get_el_amt_dict().items():
            if s in self.emb: v += f * self.emb[s]; t += f
        return v / max(t, 1e-8)

    def featurize_all(self, comps):
        out = []
        for c in tqdm(comps, desc="  Featurizing", leave=False):
            try: mg = np.array(self.ep.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)
            out.append(np.concatenate([np.nan_to_num(mg, nan=0.0), self._pool(c)]))
        return np.array(out)

    def fit_scaler(self, X): self.scaler = StandardScaler().fit(X)
    def transform(self, X):
        if not self.scaler: return X
        return np.nan_to_num(self.scaler.transform(X), nan=0.0).astype(np.float32)


class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y, np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════
# 2. MODEL: IDENTICAL TO V10A
# ══════════════════════════════════════════════════════════════════════

class HybridTRM(nn.Module):
    """Exact same architecture as V10A. No changes."""
    def __init__(self, n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                 dropout=0.2, max_steps=20, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.arch = 'Hybrid-DS-20s'

        self.tok_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        self.sa = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa_n = nn.LayerNorm(d_attn)
        self.sa_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa_fn = nn.LayerNorm(d_attn)

        self.ca = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d_attn)

        self.pool = nn.Sequential(
            nn.Linear(d_attn, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())

        self.z_up = nn.Sequential(
            nn.Linear(d_hidden*3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.y_up = nn.Sequential(
            nn.Linear(d_hidden*2, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.head = nn.Linear(d_hidden, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _attention(self, x):
        B = x.size(0)
        mg = x[:, :self.n_props * self.stat_dim]
        m2v = x[:, self.n_props * self.stat_dim:]
        tok = self.tok_proj(mg.view(B, self.n_props, self.stat_dim))
        ctx = self.m2v_proj(m2v).unsqueeze(1)
        tok = self.sa_n(tok + self.sa(tok, tok, tok)[0])
        tok = self.sa_fn(tok + self.sa_ff(tok))
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx)[0])
        return self.pool(tok.mean(dim=1))

    def forward(self, x, deep_supervision=False, return_trajectory=False):
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        step_preds = []
        for _ in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            step_preds.append(self.head(y).squeeze(1))
        if deep_supervision:
            return step_preds
        elif return_trajectory:
            return step_preds[-1], step_preds
        else:
            return step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# 3. DEEP SUPERVISION LOSS (identical to V10)
# ══════════════════════════════════════════════════════════════════════

def deep_supervision_loss(step_preds, targets):
    n = len(step_preds)
    weights = [(i + 1) for i in range(n)]
    total_w = sum(weights)
    loss = 0.0
    for pred, w in zip(step_preds, weights):
        loss += (w / total_w) * F.l1_loss(pred, targets)
    return loss


# ══════════════════════════════════════════════════════════════════════
# 4. UTILS + TRAINING (identical to V10)
# ══════════════════════════════════════════════════════════════════════

def strat_split(targets, val_size=0.15, seed=42):
    bins = np.percentile(targets, [25, 50, 75])
    lbl = np.digitize(targets, bins)
    tr, vl = [], []
    rng = np.random.RandomState(seed)
    for b in range(4):
        m = np.where(lbl == b)[0]
        if len(m) == 0: continue
        n = max(1, int(len(m) * val_size))
        c = rng.choice(m, n, replace=False)
        vl.extend(c.tolist()); tr.extend(np.setdiff1d(m, c).tolist())
    return np.array(tr), np.array(vl)


def train_fold(model, tr_dl, vl_dl, device,
               epochs=300, swa_start=200, fold=1, name=""):
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=swa_start, eta_min=1e-4)
    swa_m = AveragedModel(model)
    swa_s = SWALR(opt, swa_lr=5e-4)
    swa_on = False
    best_v, best_w = float('inf'), copy.deepcopy(model.state_dict())
    hist = {'train': [], 'val': []}

    pbar = tqdm(range(epochs), desc=f"  [{name}] F{fold}/5",
                leave=False, ncols=120)
    for ep in pbar:
        model.train(); tl = 0.0
        for bx, by in tr_dl:
            bx, by = bx.to(device), by.to(device)
            step_preds = model(bx, deep_supervision=True)
            loss = deep_supervision_loss(step_preds, by)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += F.l1_loss(step_preds[-1], by).item() * len(by)
        tl /= len(tr_dl.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                vl += F.l1_loss(pred, by).item() * len(by)
        vl /= len(vl_dl.dataset)
        hist['train'].append(tl); hist['val'].append(vl)

        if ep < swa_start:
            sch.step()
            if vl < best_v: best_v, best_w = vl, copy.deepcopy(model.state_dict())
        else:
            if not swa_on: swa_on = True
            swa_m.update_parameters(model); swa_s.step()

        pbar.set_postfix(Tr=f'{tl:.1f}', Val=f'{vl:.1f}',
                        Best=f'{best_v:.1f}', Ph='SWA' if swa_on else 'COS')

    if swa_on:
        update_bn(tr_dl, swa_m, device=device)
        model.load_state_dict(swa_m.module.state_dict())
    else:
        model.load_state_dict(best_w)
    return best_v, model, hist


def predict(model, dl, device):
    model.eval(); preds = []
    with torch.no_grad():
        for bx, _ in dl:
            preds.append(model(bx.to(device)).cpu())
    return torch.cat(preds)


def get_targets(dl):
    tgts = []
    for _, by in dl: tgts.append(by)
    return torch.cat(tgts)


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN BENCHMARK — MULTI-SEED
# ══════════════════════════════════════════════════════════════════════

def run_benchmark():
    t0 = time.time()
    print("\n" + "═"*72)
    print("  TRM-MatSci V10.1 │ REPRODUCIBILITY TEST │ matbench_steels")
    print("  Same V10A architecture × 3 seeds: 42, 123, 7")
    print("  If results cluster tightly → result is real, not lucky")
    print("═"*72 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log.info("Loading matbench_steels...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_steels")
    comps_raw = df['composition'].tolist()
    targets_all = np.array(df['yield strength'].tolist(), np.float32)
    comps_all = [Composition(c) for c in comps_raw]

    log.info("Computing features...")
    feat = CombinedFeaturizer()
    all_X = feat.featurize_all(comps_all)
    log.info(f"Features: {all_X.shape}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    os.makedirs('trm_models_v10_1', exist_ok=True)

    MODEL_KW = dict(n_props=22, stat_dim=6, mat2vec_dim=200,
                    d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                    dropout=0.2, max_steps=20)
    _m = HybridTRM(**MODEL_KW); n_params = _m.count_parameters(); del _m
    dl_kw = dict(batch_size=32, num_workers=0)

    # Storage: seed → fold MAEs
    all_results = {}  # seed_name → {'folds': [...], 'avg': ..., 'std': ...}
    all_fold_details = {}

    print(f"  Model: Hybrid-DS-20 ({n_params:,} params)")
    print(f"  Config: d_attn=48, nhead=2, d_hidden=64, ff=100, 20 steps")
    print(f"  Training: deep supervision (linear) + SWA@200")
    print(f"{'─'*72}")

    for si, (seed, sname) in enumerate(zip(SEEDS, SEED_NAMES)):
        print(f"\n{'▓'*72}")
        print(f"  SEED RUN {si+1}/3: {sname} (seed={seed})")
        print(f"{'▓'*72}")

        seed_maes = []

        for fi, (tv_i, te_i) in enumerate(folds):
            print(f"\n  ── [{sname}] Fold {fi+1}/5 {'─'*44}")

            tri, vli = strat_split(targets_all[tv_i], 0.15, seed+fi)
            feat.fit_scaler(all_X[tv_i][tri])
            tr_s = feat.transform(all_X[tv_i][tri])
            vl_s = feat.transform(all_X[tv_i][vli])
            te_s = feat.transform(all_X[te_i])

            pin = device.type == 'cuda'
            tr_dl = DataLoader(DS(tr_s, targets_all[tv_i][tri]), shuffle=True,
                               pin_memory=pin, **dl_kw)
            vl_dl = DataLoader(DS(vl_s, targets_all[tv_i][vli]), shuffle=False,
                               pin_memory=pin, **dl_kw)
            te_dl = DataLoader(DS(te_s, targets_all[te_i]), shuffle=False,
                               pin_memory=pin, **dl_kw)
            te_tgt = get_targets(te_dl)

            torch.manual_seed(seed + fi); np.random.seed(seed + fi)
            model = HybridTRM(**MODEL_KW).to(device)
            bv, model, hist = train_fold(model, tr_dl, vl_dl, device,
                                          fold=fi+1, name=sname)

            pred = predict(model, te_dl, device)
            mae = F.l1_loss(pred, te_tgt).item()
            seed_maes.append(mae)
            log.info(f"  [{sname}] Fold {fi+1}: {mae:.2f}  (val {bv:.2f})")

            torch.save({'model_state': model.state_dict(), 'test_mae': mae,
                        'seed': seed},
                       f'trm_models_v10_1/{sname}_fold{fi+1}.pt')

        avg = float(np.mean(seed_maes))
        std = float(np.std(seed_maes))
        all_results[sname] = {
            'avg': avg, 'std': std, 'folds': seed_maes,
            'seed': seed, 'params': n_params
        }
        all_fold_details[sname] = seed_maes
        print(f"\n  ═══ {sname} RESULT: {avg:.4f} ±{std:.4f} MPa ═══")

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS — REPRODUCIBILITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════

    tt = time.time() - t0
    avgs = [r['avg'] for r in all_results.values()]
    grand_mean = float(np.mean(avgs))
    grand_std = float(np.std(avgs))

    print(f"\n{'═'*72}")
    print(f"  REPRODUCIBILITY RESULTS — matbench_steels V10.1")
    print(f"{'═'*72}")
    print(f"  {'Seed':<12} {'Avg MAE':>10} {'±Fold Std':>10} {'Folds':>40}")
    print(f"  {'─'*72}")
    for sname, r in all_results.items():
        fstr = '  '.join([f'{f:.1f}' for f in r['folds']])
        tag = " ← original" if r['seed'] == 42 else ""
        print(f"  {sname:<12} {r['avg']:>10.4f} {r['std']:>10.4f}   [{fstr}]{tag}")
    print(f"  {'─'*72}")
    print(f"  {'GRAND MEAN':<12} {grand_mean:>10.4f}")
    print(f"  {'SEED STD':<12} {grand_std:>10.4f}  ← stability across seeds")
    print(f"  {'─'*72}")

    # Stability verdict
    if grand_std < 3.0:
        verdict = "HIGHLY STABLE ✅ (seed std < 3 MPa)"
    elif grand_std < 6.0:
        verdict = "STABLE ✅ (seed std < 6 MPa)"
    elif grand_std < 10.0:
        verdict = "MODERATE ⚠️ (seed std < 10 MPa)"
    else:
        verdict = "UNSTABLE ❌ (seed std ≥ 10 MPa)"

    print(f"  Verdict: {verdict}")

    # Context
    beats_darwin = sum(1 for a in avgs if a < 123.29)
    beats_crabnet = sum(1 for a in avgs if a < 107.32)
    beats_rf = sum(1 for a in avgs if a < 103.51)
    print(f"\n  Beats Darwin  (123.29): {beats_darwin}/3 seeds")
    print(f"  Beats CrabNet (107.32): {beats_crabnet}/3 seeds")
    print(f"  Beats RF-SCM  (103.51): {beats_rf}/3 seeds")
    print(f"\n  Total time: {tt/60:.1f} minutes")

    # Baselines
    print(f"\n  {'─'*52}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<22} {'baseline':>10} {bv:>10.4f}")
    print()

    # Per-fold comparison across seeds
    print(f"\n{'═'*72}")
    print(f"  PER-FOLD COMPARISON ACROSS SEEDS")
    print(f"{'═'*72}")
    print(f"  {'Fold':<6}", end="")
    for sname in SEED_NAMES:
        print(f" {sname:>10}", end="")
    print(f" {'FoldAvg':>10} {'FoldStd':>8}")
    print(f"  {'─'*56}")
    for fi in range(5):
        fold_vals = [all_fold_details[sn][fi] for sn in SEED_NAMES]
        print(f"  {fi+1:<6}", end="")
        for v in fold_vals:
            print(f" {v:>10.2f}", end="")
        print(f" {np.mean(fold_vals):>10.2f} {np.std(fold_vals):>8.2f}")
    print()

    generate_plots(all_results, grand_mean, grand_std, verdict)
    save_summary(all_results, all_fold_details, grand_mean, grand_std,
                 verdict, tt)
    return all_results


# ══════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════

SEED_COLORS = {'Seed-42': '#1565C0', 'Seed-123': '#E65100', 'Seed-7': '#2E7D32'}

def generate_plots(all_results, grand_mean, grand_std, verdict):
    names = list(all_results.keys())
    avgs = [all_results[n]['avg'] for n in names]
    stds = [all_results[n]['std'] for n in names]
    cols = [SEED_COLORS.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)

    # ── Bar chart: MAE per seed ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(names, avgs, yerr=stds, capsize=6, color=cols,
                   alpha=0.88, edgecolor='white', linewidth=1.5)
    for bv, c, ls, lb in [
        (87.76, '#F57F17', '--', 'MODNet (87.76)'),
        (103.51, '#9E9E9E', ':', 'RF-SCM (103.51)'),
        (107.32, '#B0BEC5', ':', 'CrabNet (107.32)'),
        (123.29, '#FF5722', ':', 'Darwin (123.29)'),
    ]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    # Grand mean band
    ax1.axhspan(grand_mean - grand_std, grand_mean + grand_std,
                alpha=0.15, color='#1565C0', label=f'Grand mean ±σ ({grand_mean:.1f}±{grand_std:.1f})')
    ax1.axhline(grand_mean, color='#1565C0', ls='-', lw=2, alpha=0.5)

    for bar, m, s in zip(bars, avgs, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.5,
                 f'{m:.1f}', ha='center', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_ylabel('MAE (MPa)')
    ax1.set_ylim(0, max(avgs)*1.5)
    ax1.set_title(f'V10.1 Reproducibility │ {verdict}',
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # ── Per-fold comparison across seeds ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(1, 6)
    w = 0.25
    for i, (sname, col) in enumerate(zip(names, cols)):
        fold_vals = all_results[sname]['folds']
        ax2.bar(x + (i - 1) * w, fold_vals, w, color=col, alpha=0.8,
                label=sname, edgecolor='white')
    ax2.axhline(103.51, color='#9E9E9E', ls=':', lw=1.5, label='RF-SCM')
    ax2.axhline(107.32, color='#B0BEC5', ls=':', lw=1.5, label='CrabNet')
    ax2.axhline(123.29, color='#FF5722', ls=':', lw=1.5, label='Darwin')
    ax2.set_xlabel('Fold'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_xticks(x); ax2.set_xticklabels([f'F{i}' for i in range(1,6)])
    ax2.set_title('Per-Fold Stability Across Seeds', fontweight='bold')
    ax2.legend(fontsize=7); ax2.grid(axis='y', alpha=0.2)

    fig.suptitle('TRM-MatSci V10.1 │ Multi-Seed Reproducibility │ matbench_steels',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v10_1.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: trm_results_v10_1.png")


def save_summary(all_results, fold_details, grand_mean, grand_std,
                 verdict, total_s):
    s = {
        'version': 'V10.1', 'task': 'matbench_steels',
        'purpose': 'Reproducibility test — V10A architecture × 3 seeds',
        'seeds': SEEDS,
        'grand_mean_mae': round(grand_mean, 4),
        'seed_std': round(grand_std, 4),
        'verdict': verdict,
        'total_min': round(total_s/60, 1),
        'models': {n: {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()}
                   for n, r in all_results.items()},
        'fold_details': {k: [round(v, 4) for v in vals]
                         for k, vals in fold_details.items()},
    }
    with open('trm_models_v10_1/summary_v10_1.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    log.info("✓ Saved: summary_v10_1.json")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v10_1_all", "zip", "trm_models_v10_1")
    log.info("✓ Created trm_v10_1_all.zip")
