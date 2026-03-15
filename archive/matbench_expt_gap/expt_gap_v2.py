"""
+==============================================================+
|  TRIADS V2 on matbench_expt_gap - 100K Sweet Spot Hunt       |
|  4 Models: Steps(16,20) x Dropout(0.15,0.20)                |
|  FastTensorDataLoader + batch_size=256 for MAX GPU usage     |
|  Esports-grade telemetry: live race, gap tracker, alarms    |
+==============================================================+

GPU FIX: The CPU bottleneck wasn't data loading - it was Python
loop overhead. With batch_size=64, each epoch had ~57 iterations.
The GPU finished each batch in microseconds but Python spent all
its time in loop/tqdm overhead. batch_size=256 = only ~14 iters.
"""

import os, copy, json, time, logging, warnings, random, urllib.request
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
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
log = logging.getLogger("TRIADS-EG")

SEEDS = [42]
BATCH_SIZE = 256  # 4x bigger than V1 => 4x fewer Python loop iterations

BASELINES = {
    'Darwin':              0.2865,
    'Ax/SAASBO CrabNet':   0.3310,
    'MODNet v0.1.12':      0.3327,
    'AMMExpress v2020':    0.4161,
    'CrabNet':             0.4427,
    'RF-SCM/Magpie':       0.5205,
    'Dummy':               1.0280,
}
V1_BEST = {'EG-A (V1)': 0.3510, 'EG-B (V1)': 0.3616}

# ======================================================================
# FUN TELEMETRY
# ======================================================================

FACTS = [
    "Diamond has a band gap of 5.5 eV - that's why it's an insulator and sparkles!",
    "Silicon's band gap (1.12 eV) is literally the foundation of the digital age.",
    "GaN (3.4 eV gap) makes your phone charger 10x smaller. Thank gallium!",
    "Perovskites: 3.8% to 25.7% solar efficiency in just 10 years. Fastest material ever.",
    "The element gallium melts in your hand (29.8C) but its arsenide powers WiFi.",
    "VO2 switches from insulator to metal at 68C. Scientists still argue about WHY.",
    "Graphene has ZERO band gap. Great conductor, terrible transistor.",
    "The band gap of InN was wrongly reported as 1.9 eV for DECADES. It's 0.7 eV.",
    "ZnO: 3.37 eV band gap AND piezoelectric. Generates power from vibrations.",
    "CdTe powered NASA spacecraft since the 1960s. Band gap: 1.5 eV (perfect for solar).",
    "SiC works at 600C+. That's why it's in electric vehicle power electronics.",
    "TiO2 (3.2 eV) makes white paint white AND splits water with sunlight!",
    "Cu2O was the FIRST semiconductor ever studied (1874). Accidental solar cell.",
    "AlN (6.2 eV gap) - used in deep-UV LEDs for water purification.",
    "MoS2 was first isolated by sticky tape, just like graphene!",
    "HgCdTe can be tuned from 0 to 1.5 eV by changing the Hg:Cd ratio. Band gap dial!",
    "The first LED (1962): infrared from GaAs. Visible LEDs took another decade.",
    "Tin (Sn) is BOTH a metal (white tin) and a semiconductor (grey tin).",
]

DRIFT_MSGS = [
    "Model is emotionally unstable. Give it space.",
    "Val going up? Someone's having a bad epoch.",
    "Three in a row? That's not noise, that's a cry for help.",
    "Model: 'I'm fine.' Narrator: It was not fine.",
    "Overfitting detected. Model is memorizing, not learning.",
]

SWA_MSGS = [
    "SWA saved your career. You're welcome.",
    "SWA: turning chaos into flat minima since 2018.",
    "Without SWA, this model would be unemployable.",
    "SWA just did what 100 more epochs couldn't.",
    "SWA: the duct tape of deep learning. Ugly but effective.",
]

GAP_MSGS_GOOD = [
    "Bias aligned. Train and val are BFFs.",
    "Gap is tight. This model knows what it's doing.",
    "Generalization gap under control. Steady hands.",
]

GAP_MSGS_BAD = [
    "Overfit brewing. Train is leaving val behind.",
    "Gap widening. Model is getting too comfortable with training data.",
    "Train-val divergence detected. Regularize or suffer.",
]


def fact(): print(f"\n  >> {random.choice(FACTS)}\n")


# ======================================================================
# FAST TENSOR DATALOADER
# ======================================================================

class FastTensorDataLoader:
    """Zero-CPU DataLoader. Entire dataset in GPU VRAM.
    Combined with batch_size=256, this minimizes Python loop overhead.
    """
    def __init__(self, *tensors, batch_size=256, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = (self.dataset_len + batch_size - 1) // batch_size

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.dataset_len, device=self.tensors[0].device)
            self.tensors = tuple(t[idx] for t in self.tensors)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


# ======================================================================
# FEATURIZER
# ======================================================================

class ExpandedFeaturizer:
    GCS = "https://storage.googleapis.com/mat2vec/"
    FILES = ["pretrained_embeddings",
             "pretrained_embeddings.wv.vectors.npy",
             "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache="mat2vec_cache"):
        from matminer.featurizers.composition import (
            ElementFraction, Stoichiometry, ValenceOrbital,
            IonProperty, BandCenter
        )
        from matminer.featurizers.base import MultipleFeaturizer
        self.ep_magpie = ElementProperty.from_preset("magpie")
        self.n_mg = len(self.ep_magpie.feature_labels())
        self.extra_feats = MultipleFeaturizer([
            ElementFraction(), Stoichiometry(), ValenceOrbital(),
            IonProperty(), BandCenter(),
        ])
        self.n_extra = None
        self.scaler = None
        os.makedirs(cache, exist_ok=True)
        for f in self.FILES:
            p = os.path.join(cache, f)
            if not os.path.exists(p):
                log.info(f"  Downloading {f}...")
                urllib.request.urlretrieve(self.GCS + f, p)
        self.m2v = Word2Vec.load(os.path.join(cache, "pretrained_embeddings"))
        self.emb = {w: self.m2v.wv[w] for w in self.m2v.wv.index_to_key}

    def _pool(self, c):
        v, t = np.zeros(200, np.float32), 0.0
        for s, f in c.get_el_amt_dict().items():
            if s in self.emb: v += f * self.emb[s]; t += f
        return v / max(t, 1e-8)

    def featurize_all(self, comps):
        out = []
        for c in tqdm(comps, desc="  Featurizing", leave=False):
            try: mg = np.array(self.ep_magpie.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)
            try: ex = np.array(self.extra_feats.featurize(c), np.float32)
            except: ex = np.zeros(self.n_extra or 200, np.float32)
            if self.n_extra is None:
                self.n_extra = len(ex)
                log.info(f"Features: {self.n_mg} Magpie + {self.n_extra} Extra + 200 Mat2Vec")
            out.append(np.concatenate([
                np.nan_to_num(mg, nan=0.0),
                np.nan_to_num(ex, nan=0.0),
                self._pool(c)
            ]))
        return np.array(out)

    def fit_scaler(self, X): self.scaler = StandardScaler().fit(X)
    def transform(self, X):
        if not self.scaler: return X
        return np.nan_to_num(self.scaler.transform(X), nan=0.0).astype(np.float32)


# ======================================================================
# MODEL
# ======================================================================

class DeepHybridTRM(nn.Module):
    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, max_steps=20, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.n_props, self.stat_dim, self.n_extra = n_props, stat_dim, n_extra

        self.tok_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        self.sa1 = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa1_n = nn.LayerNorm(d_attn)
        self.sa1_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa1_fn = nn.LayerNorm(d_attn)

        self.sa2 = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa2_n = nn.LayerNorm(d_attn)
        self.sa2_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa2_fn = nn.LayerNorm(d_attn)

        self.ca = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d_attn)

        pool_in = d_attn + (n_extra if n_extra > 0 else 0)
        self.pool = nn.Sequential(
            nn.Linear(pool_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())

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
        mg_dim = self.n_props * self.stat_dim
        if self.n_extra > 0:
            extra = x[:, mg_dim:mg_dim + self.n_extra]
            m2v = x[:, mg_dim + self.n_extra:]
        else:
            extra, m2v = None, x[:, mg_dim:]

        tok = self.tok_proj(x[:, :mg_dim].view(B, self.n_props, self.stat_dim))
        ctx = self.m2v_proj(m2v).unsqueeze(1)

        tok = self.sa1_n(tok + self.sa1(tok, tok, tok)[0])
        tok = self.sa1_fn(tok + self.sa1_ff(tok))
        tok = self.sa2_n(tok + self.sa2(tok, tok, tok)[0])
        tok = self.sa2_fn(tok + self.sa2_ff(tok))
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx)[0])

        pooled = tok.mean(dim=1)
        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=-1)
        return self.pool(pooled)

    def forward(self, x, deep_supervision=False, eval_steps=None):
        """
        eval_steps: if set, return predictions at these specific step indices
                    e.g. eval_steps=[4,8,12,16,20] for recursion sensitivity
        """
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        step_preds = []
        for s in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            step_preds.append(self.head(y).squeeze(1))

        if deep_supervision:
            return step_preds
        elif eval_steps is not None:
            return {s: step_preds[s-1] for s in eval_steps if s <= self.max_steps}
        else:
            return step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# LOSS + UTILS
# ======================================================================

def deep_supervision_loss(step_preds, targets):
    n = len(step_preds)
    weights = [(i+1) for i in range(n)]
    tw = sum(weights)
    return sum((w/tw) * F.l1_loss(p, targets) for p, w in zip(step_preds, weights))


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


def predict(model, dl):
    model.eval(); preds = []
    with torch.no_grad():
        for bx, _ in dl:
            preds.append(model(bx).cpu())
    return torch.cat(preds)


def recursion_sensitivity(model, te_dl, te_tgt, max_steps):
    """Evaluate model at different recursion depths to see if extra steps help."""
    checkpoints = [s for s in [4, 8, 12, 16, 20] if s <= max_steps]
    results = {}
    model.eval()
    all_preds = {s: [] for s in checkpoints}
    with torch.no_grad():
        for bx, _ in te_dl:
            step_dict = model(bx, eval_steps=checkpoints)
            for s, p in step_dict.items():
                all_preds[s].append(p.cpu())
    for s in checkpoints:
        pred = torch.cat(all_preds[s])
        results[s] = F.l1_loss(pred, te_tgt).item()
    return results


# ======================================================================
# TRAINING WITH LIVE TELEMETRY
# ======================================================================

def train_fold(model, tr_dl, vl_dl, device,
               epochs=300, swa_start=200, fold=1, name=""):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=swa_start, eta_min=1e-4)
    swa_m = AveragedModel(model)
    swa_s = SWALR(opt, swa_lr=5e-4)
    swa_on = False
    best_v, best_w = float('inf'), copy.deepcopy(model.state_dict())
    pre_swa_best = float('inf')
    hist = {'train': [], 'val': [], 'gap': []}
    val_up_streak = 0
    prev_val = float('inf')

    pbar = tqdm(range(epochs), desc=f"  [{name}] F{fold}/5",
                leave=False, ncols=120)
    for ep in pbar:
        model.train(); tl = 0.0
        for bx, by in tr_dl:
            sp = model(bx, deep_supervision=True)
            loss = deep_supervision_loss(sp, by)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += F.l1_loss(sp[-1], by).item() * len(by)
        tl /= tr_dl.dataset_len

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                vl += F.l1_loss(model(bx), by).item() * len(by)
        vl /= vl_dl.dataset_len
        gap = vl - tl
        hist['train'].append(tl); hist['val'].append(vl); hist['gap'].append(gap)

        # ---- DRIFT ALARM ----
        if vl > prev_val:
            val_up_streak += 1
            if val_up_streak == 3 and not swa_on:
                tqdm.write(f"  >> {random.choice(DRIFT_MSGS)}")
                val_up_streak = 0
        else:
            val_up_streak = 0
        prev_val = vl

        # ---- GAP TRACKER (every 50 epochs) ----
        if (ep + 1) % 50 == 0:
            gap_ratio = gap / max(tl, 1e-6)
            if gap_ratio < 1.0:
                tqdm.write(f"  [Ep {ep+1}] Gap={gap:.4f} ({gap_ratio:.0%} of train)  "
                           f">> {random.choice(GAP_MSGS_GOOD)}")
            else:
                tqdm.write(f"  [Ep {ep+1}] Gap={gap:.4f} ({gap_ratio:.0%} of train)  "
                           f">> {random.choice(GAP_MSGS_BAD)}")

        if ep < swa_start:
            sch.step()
            if vl < best_v:
                best_v = vl
                best_w = copy.deepcopy(model.state_dict())
        else:
            if not swa_on:
                swa_on = True
                pre_swa_best = best_v
            swa_m.update_parameters(model); swa_s.step()

        pbar.set_postfix(Tr=f'{tl:.4f}', Val=f'{vl:.4f}',
                        Best=f'{best_v:.4f}', Gap=f'{gap:.3f}',
                        Ph='SWA' if swa_on else 'COS')

    # ---- SWA IMPACT ----
    if swa_on:
        update_bn(tr_dl, swa_m, device=device)
        model.load_state_dict(swa_m.module.state_dict())
        # Quick eval to see SWA effect
        model.eval(); swa_val = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                swa_val += F.l1_loss(model(bx), by).item() * len(by)
        swa_val /= vl_dl.dataset_len
        swa_improvement = pre_swa_best - swa_val
        if swa_improvement > 0.005:
            print(f"  >> SWA improved val by {swa_improvement:.4f}! {random.choice(SWA_MSGS)}")
        elif swa_improvement < -0.005:
            print(f"  >> SWA actually hurt by {-swa_improvement:.4f}. Loading pre-SWA best instead.")
            model.load_state_dict(best_w)
        else:
            print(f"  >> SWA effect: {swa_improvement:+.4f} (neutral). Using SWA weights.")
    else:
        model.load_state_dict(best_w)
    return best_v, model, hist


# ======================================================================
# LIVE SCOREBOARD
# ======================================================================

def print_scoreboard(fold_table, model_names, current_fold):
    """Print a live-updating fold ranking board."""
    print(f"\n  {'='*70}")
    print(f"  LIVE SCOREBOARD (after Fold {current_fold}/5)")
    print(f"  {'='*70}")

    # Header
    header = f"  {'Fold':<6}"
    for mn in model_names:
        header += f"  {mn:>14}"
    print(header)
    print(f"  {'-'*70}")

    # Rows
    for fi in range(current_fold):
        row = f"  F{fi+1:<5}"
        fold_vals = []
        for mn in model_names:
            if mn in fold_table and fi < len(fold_table[mn]):
                v = fold_table[mn][fi]
                fold_vals.append((mn, v))
                row += f"  {v:>14.4f}"
            else:
                row += f"  {'...':>14}"
        # Mark best in this fold
        print(row)

    # Averages so far
    print(f"  {'-'*70}")
    avg_row = f"  {'AVG':<6}"
    model_avgs = {}
    for mn in model_names:
        if mn in fold_table and len(fold_table[mn]) > 0:
            avg = np.mean(fold_table[mn])
            model_avgs[mn] = avg
            avg_row += f"  {avg:>14.4f}"
        else:
            avg_row += f"  {'...':>14}"
    print(avg_row)

    # Leader announcement
    if model_avgs:
        leader = min(model_avgs, key=model_avgs.get)
        leader_mae = model_avgs[leader]
        print(f"\n  Current Leader: {leader}  -->  {leader_mae:.4f} eV")
        
        # Context vs baselines
        if leader_mae < 0.2865:
            print(f"  STATUS: BEATING DARWIN!!! YOU ARE #1!!!")
        elif leader_mae < 0.3310:
            print(f"  STATUS: Top 2! {0.2865 - leader_mae:.4f} eV to Darwin")
        elif leader_mae < 0.3327:
            print(f"  STATUS: Top 3! {0.2865 - leader_mae:.4f} eV to Darwin")
        elif leader_mae < 0.3510:
            print(f"  STATUS: Beating V1! {0.3327 - leader_mae:.4f} eV to MODNet")
        else:
            print(f"  STATUS: {0.3510 - leader_mae:.4f} eV to beat V1")
    print()


def print_fold_difficulty(fold_table, current_fold):
    """Track which folds are hardest across all models."""
    if current_fold < 2:
        return
    
    fold_avgs = {}
    for fi in range(current_fold):
        vals = []
        for mn, folds in fold_table.items():
            if fi < len(folds):
                vals.append(folds[fi])
        if vals:
            fold_avgs[fi+1] = np.mean(vals)
    
    if fold_avgs:
        sorted_folds = sorted(fold_avgs.items(), key=lambda x: x[1], reverse=True)
        difficulty = " > ".join([f"F{f}({v:.3f})" for f, v in sorted_folds])
        print(f"  Fold Difficulty: {difficulty}")
        hardest = sorted_folds[0]
        easiest = sorted_folds[-1]
        print(f"  Hardest: F{hardest[0]} ({hardest[1]:.4f})  |  "
              f"Easiest: F{easiest[0]} ({easiest[1]:.4f})  |  "
              f"Spread: {hardest[1]-easiest[1]:.4f} eV")


def print_recursion_analysis(rec_results, name):
    """Print recursion step sensitivity analysis."""
    print(f"\n  Recursion Sensitivity [{name}]:")
    print(f"  {'Steps':>8} {'MAE(eV)':>10} {'vs Final':>10}  Visual")
    print(f"  {'-'*50}")
    
    steps = sorted(rec_results.keys())
    final_mae = rec_results[steps[-1]]
    
    for s in steps:
        mae = rec_results[s]
        diff = mae - final_mae
        bar_len = max(1, min(40, int((mae - 0.25) * 100)))
        bar = '#' * bar_len
        marker = " <-- BEST" if mae == min(rec_results.values()) else ""
        print(f"  {s:>8} {mae:>10.4f} {diff:>+10.4f}  |{bar}{marker}")
    
    best_step = min(rec_results, key=rec_results.get)
    if best_step < steps[-1]:
        print(f"  >> Peak at step {best_step}! Extra steps HURT by "
              f"{rec_results[steps[-1]] - rec_results[best_step]:.4f} eV. Over-refinement!")
    else:
        print(f"  >> More steps = better. No over-refinement detected.")


def save_live_plot(all_hists, fold_table, model_names, n_done):
    """Save a live-updating plot after each model completes."""
    if not all_hists:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    colors = {'100K-S16-D15': '#1565C0', '100K-S16-D20': '#1E88E5',
              '100K-S20-D15': '#E65100', '100K-S20-D20': '#FB8C00'}
    
    # Panel 1: Training curves (latest model, all folds)
    ax1 = axes[0]
    for cname, hists in all_hists.items():
        col = colors.get(cname, '#888')
        for fi, h in enumerate(hists):
            lbl = f'{cname}' if fi == 0 else None
            ax1.plot(h['train'], alpha=0.3, lw=0.8, color=col)
            ax1.plot(h['val'], alpha=0.7, lw=1.2, color=col, ls='--', label=lbl)
    ax1.axhline(0.2865, color='red', ls='--', lw=1, label='Darwin')
    ax1.axhline(0.3510, color='green', ls='--', lw=1, label='V1')
    ax1.axvline(200, color='gray', ls=':', lw=1, label='SWA')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MAE (eV)')
    ax1.set_title('Train/Val Curves'); ax1.legend(fontsize=6, ncol=2)
    ax1.grid(alpha=0.2)
    
    # Panel 2: Gap tracker (latest model)
    ax2 = axes[1]
    for cname, hists in all_hists.items():
        col = colors.get(cname, '#888')
        for fi, h in enumerate(hists):
            lbl = f'{cname}' if fi == 0 else None
            ax2.plot(h['gap'], alpha=0.5, lw=1.0, color=col, label=lbl)
    ax2.axhline(0, color='black', ls='-', lw=0.5)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val - Train (eV)')
    ax2.set_title('Generalization Gap (lower = better)'); ax2.legend(fontsize=6)
    ax2.grid(alpha=0.2)
    
    # Panel 3: Model comparison bars
    ax3 = axes[2]
    if fold_table:
        completed = {mn: np.mean(folds) for mn, folds in fold_table.items() if len(folds) >= 1}
        if completed:
            names = list(completed.keys())
            vals = [completed[n] for n in names]
            cols = [colors.get(n, '#888') for n in names]
            bars = ax3.bar(range(len(names)), vals, color=cols, alpha=0.8, edgecolor='white')
            ax3.set_xticks(range(len(names)))
            ax3.set_xticklabels(names, fontsize=7, rotation=15)
            ax3.axhline(0.2865, color='red', ls='--', lw=1, label='Darwin')
            ax3.axhline(0.3510, color='green', ls='--', lw=1, label='V1')
            for bar, v in zip(bars, vals):
                ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                        f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
            ax3.set_ylabel('MAE (eV)')
            ax3.set_title(f'Model Comparison ({n_done}/4 done)')
            ax3.legend(fontsize=7)
            ax3.grid(axis='y', alpha=0.2)
    
    fig.suptitle(f'TRIADS V2 Live Dashboard | {n_done}/4 models complete',
                fontweight='bold')
    fig.tight_layout()
    fig.savefig('expt_gap_live_v2.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    log.info("Live plot saved: expt_gap_live_v2.png")


# ======================================================================
# MAIN
# ======================================================================

def run_benchmark():
    t0 = time.time()
    print("""
  +============================================================+
  |  TRIADS V2 -- 100K Sweet Spot Hunt                         |
  |  4 Models: Steps(16,20) x Dropout(0.15,0.20)              |
  |  GPU: FastTensorDataLoader + batch_size=256                |
  |  Telemetry: Live race, gap tracker, drift alarm            |
  +============================================================+
    """)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gn = torch.cuda.get_device_name(0)
        try:
            gm = torch.cuda.get_device_properties(0).total_memory / 1e9
        except AttributeError:
            gm = 0
        log.info(f"GPU: {gn} ({gm:.1f} GB)")
        log.info(f"Strategy: Full VRAM + batch_size={BATCH_SIZE}")
        log.info(f"Expected: CPU ~5-10%  |  GPU ~60-80%")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        log.info("No GPU. Running on CPU.")

    # ---- LOAD DATA ----
    log.info("Loading matbench_expt_gap...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_expt_gap")
    comps_raw = df['composition'].tolist()
    targets_all = np.array(df['gap expt'].tolist(), np.float32)
    comps_all = [Composition(c) for c in comps_raw]
    log.info(f"Dataset: {len(comps_all)} samples | "
             f"mean={targets_all.mean():.3f}, std={targets_all.std():.3f}")

    # ---- FEATURIZE ----
    feat = ExpandedFeaturizer()
    X_all = feat.featurize_all(comps_all)
    n_extra = feat.n_extra
    log.info(f"Features: {X_all.shape} (n_extra={n_extra})")

    fact()  # Fun fact while we set up!

    # ---- SPLITS ----
    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    for fi, (tv, te) in enumerate(folds):
        assert len(set(tv) & set(te)) == 0
    log.info("5 folds verified: zero leakage")

    os.makedirs('expt_gap_models_v2', exist_ok=True)

    # ---- CONFIGS (all ~100K params) ----
    base = dict(n_props=22, stat_dim=6, n_extra=n_extra, mat2vec_dim=200,
                d_attn=36, nhead=4, d_hidden=72, ff_dim=112)

    configs = {
        '100K-S16-D15': {**base, 'max_steps': 16, 'dropout': 0.15},
        '100K-S16-D20': {**base, 'max_steps': 16, 'dropout': 0.20},
        '100K-S20-D15': {**base, 'max_steps': 20, 'dropout': 0.15},
        '100K-S20-D20': {**base, 'max_steps': 20, 'dropout': 0.20},
    }

    print(f"\n  {'Config':<20} {'Params':>10} {'Steps':>6} {'Drop':>6} {'Batch':>6}")
    print(f"  {'='*54}")
    param_counts = {}
    for cn, kw in configs.items():
        m = DeepHybridTRM(**kw); np_ = m.count_parameters(); del m
        param_counts[cn] = np_
        print(f"  {cn:<20} {np_:>10,} {kw['max_steps']:>6} {kw['dropout']:>6.2f} {BATCH_SIZE:>6}")
    print()

    # ---- TRAIN ALL ----
    all_results = {}
    all_hists = {}
    fold_table = {}  # {model_name: [fold1_mae, fold2_mae, ...]}
    model_names = list(configs.keys())

    for ci, (cname, model_kw) in enumerate(configs.items()):
        print(f"\n  {'='*60}")
        print(f"  [{ci+1}/4] {cname}  ({param_counts[cname]:,} params)")
        print(f"  {'='*60}")
        if ci > 0: fact()

        seed = SEEDS[0]
        fold_maes = []
        fold_hists = []
        fold_table[cname] = []

        for fi, (tv_i, te_i) in enumerate(folds):
            print(f"\n  -- [{cname}] Fold {fi+1}/5 " + "-"*30)

            tri, vli = strat_split(targets_all[tv_i], 0.15, seed + fi)
            feat.fit_scaler(X_all[tv_i][tri])

            # FULL VRAM LOADING
            tr_x = torch.tensor(feat.transform(X_all[tv_i][tri]), dtype=torch.float32).to(device)
            tr_y = torch.tensor(targets_all[tv_i][tri], dtype=torch.float32).to(device)
            vl_x = torch.tensor(feat.transform(X_all[tv_i][vli]), dtype=torch.float32).to(device)
            vl_y = torch.tensor(targets_all[tv_i][vli], dtype=torch.float32).to(device)
            te_x = torch.tensor(feat.transform(X_all[te_i]), dtype=torch.float32).to(device)
            te_y = torch.tensor(targets_all[te_i], dtype=torch.float32).to(device)

            tr_dl = FastTensorDataLoader(tr_x, tr_y, batch_size=BATCH_SIZE, shuffle=True)
            vl_dl = FastTensorDataLoader(vl_x, vl_y, batch_size=BATCH_SIZE, shuffle=False)
            te_dl = FastTensorDataLoader(te_x, te_y, batch_size=BATCH_SIZE, shuffle=False)

            log.info(f"  train={len(tr_x)} val={len(vl_x)} test={len(te_x)} "
                     f"| batches/epoch={len(tr_dl)} | ALL ON {device}")

            torch.manual_seed(seed + fi)
            np.random.seed(seed + fi)
            if device.type == 'cuda': torch.cuda.manual_seed(seed + fi)

            model = DeepHybridTRM(**model_kw).to(device)
            bv, model, hist = train_fold(
                model, tr_dl, vl_dl, device,
                epochs=300, swa_start=200, fold=fi+1, name=cname)
            fold_hists.append(hist)

            # Test evaluation
            pred = predict(model, te_dl)
            te_tgt = te_y.cpu()
            mae = F.l1_loss(pred, te_tgt).item()
            log.info(f"  Fold {fi+1} TEST MAE = {mae:.4f} eV  (val best = {bv:.4f})")

            # Recursion sensitivity (only on last fold to save time)
            if fi == 4:
                rec = recursion_sensitivity(model, te_dl, te_tgt, model_kw['max_steps'])
                print_recursion_analysis(rec, cname)

            fold_maes.append(mae)
            fold_table[cname].append(mae)

            torch.save({
                'model_state': model.state_dict(),
                'test_mae': mae, 'config': cname, 'seed': seed,
                'fold': fi+1, 'n_extra': n_extra,
            }, f'expt_gap_models_v2/{cname}_s{seed}_f{fi+1}.pt')

            del model, tr_x, tr_y, vl_x, vl_y, te_x, te_y
            if device.type == 'cuda': torch.cuda.empty_cache()

        avg = float(np.mean(fold_maes))
        std = float(np.std(fold_maes))
        all_results[cname] = {'avg': avg, 'std': std, 'folds': fold_maes,
                              'params': param_counts[cname]}
        all_hists[cname] = fold_hists

        print(f"\n  === {cname} DONE ===")
        print(f"      5-Fold Avg MAE: {avg:.4f} +/- {std:.4f} eV")
        print(f"      Per-fold: {[f'{m:.4f}' for m in fold_maes]}")

        # Live scoreboard + fold difficulty
        print_scoreboard(fold_table, model_names, 5)
        print_fold_difficulty(fold_table, 5)

        # Save live plot after each model
        save_live_plot(all_hists, fold_table, model_names, ci+1)

    # ======== FINAL RESULTS ========
    tt = time.time() - t0
    print(f"\n{'='*72}")
    print(f"  FINAL STANDINGS -- matbench_expt_gap V2 (5-Fold Avg MAE)")
    print(f"{'='*72}")
    print(f"  {'Model':<24} {'Params':>10} {'MAE':>10} {'Std':>8}  Notes")
    print(f"  {'-'*72}")

    for n, r in sorted(all_results.items(), key=lambda x: x[1]['avg']):
        tag = (" <-- #1 DARWIN BEATEN!" if r['avg'] < 0.2865 else
               " <-- Top 3!"           if r['avg'] < 0.3327 else
               " <-- Beats V1!"        if r['avg'] < 0.3510 else
               " <-- Beats AMMExpress"  if r['avg'] < 0.4161 else "")
        print(f"  {n:<24} {r['params']:>9,} {r['avg']:>10.4f} {r['std']:>8.4f}{tag}")

    print(f"  {'-'*72}")
    for vn, vm in sorted(V1_BEST.items(), key=lambda x: x[1]):
        print(f"  {vn:<24} {'(V1)':>10} {vm:>10.4f}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<24} {'baseline':>10} {bv:>10.4f}")

    print(f"\n  Total time: {tt/60:.1f} min  |  "
          f"Time per model: {tt/60/len(configs):.1f} min")

    # Per-fold table
    print(f"\n{'='*72}")
    print(f"  PER-FOLD BREAKDOWN")
    print(f"{'='*72}")
    hdr = f"  {'Fold':<6}"
    for cn in model_names: hdr += f"  {cn:>14}"
    print(hdr)
    print(f"  {'-'*72}")
    for fi in range(5):
        row = f"  F{fi+1:<5}"
        for cn in model_names:
            row += f"  {all_results[cn]['folds'][fi]:>14.4f}"
        print(row)

    # Heatmap
    print(f"\n  HYPERPARAMETER GRID (Steps x Dropout):")
    print(f"  {'':>12} {'D=0.15':>12} {'D=0.20':>12}")
    for s in [16, 20]:
        d15 = all_results.get(f'100K-S{s}-D15', {}).get('avg', 0)
        d20 = all_results.get(f'100K-S{s}-D20', {}).get('avg', 0)
        best_mark15 = " *" if d15 == min(r['avg'] for r in all_results.values()) else ""
        best_mark20 = " *" if d20 == min(r['avg'] for r in all_results.values()) else ""
        print(f"  S={s:>2}       {d15:>10.4f}{best_mark15:>2}  {d20:>10.4f}{best_mark20:>2}")
    print(f"  (* = best)")

    fact()  # One last fun fact!

    # Final plots
    generate_final_plots(all_results, all_hists)
    save_summary(all_results, all_hists, tt, n_extra)
    return all_results


# ======================================================================
# FINAL PLOTS
# ======================================================================

PAL = {'100K-S16-D15': '#1565C0', '100K-S16-D20': '#1E88E5',
       '100K-S20-D15': '#E65100', '100K-S20-D20': '#FB8C00'}

def generate_final_plots(all_results, all_hists):
    names = list(all_results.keys())
    avgs = [all_results[n]['avg'] for n in names]
    stds = [all_results[n]['std'] for n in names]
    cols = [PAL.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # 1: Bars vs baselines
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(names)), avgs, 0.5, yerr=stds, capsize=6,
                   color=cols, alpha=0.88, edgecolor='white', linewidth=1.5)
    for bv, c, ls, lb in [(0.2865,'#F44336','--','Darwin'),
                           (0.3327,'#F57F17',':','MODNet'),
                           (0.3510,'#4CAF50','--','V1')]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    for bar, m, s in zip(bars, avgs, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.003,
                 f'{m:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, fontsize=7, rotation=15)
    ax1.legend(fontsize=7); ax1.set_ylabel('MAE (eV)'); ax1.set_ylim(0, max(avgs)*1.5)
    ax1.set_title('V2 vs Baselines', fontweight='bold'); ax1.grid(axis='y', alpha=0.3)

    # 2: Per-fold
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(1, 6); w = 0.18
    for i, (n, col) in enumerate(zip(names, cols)):
        ax2.bar(x + (i-1.5)*w, all_results[n]['folds'], w, color=col, alpha=0.8,
                label=n, edgecolor='white')
    ax2.axhline(0.2865, color='red', ls='--', lw=1.5); ax2.axhline(0.3510, color='green', ls='--')
    ax2.set_xlabel('Fold'); ax2.set_ylabel('MAE')
    ax2.set_xticks(x); ax2.set_xticklabels([f'F{i}' for i in range(1,6)])
    ax2.set_title('Per-Fold', fontweight='bold'); ax2.legend(fontsize=6); ax2.grid(axis='y', alpha=0.2)

    # 3: Train curves
    ax3 = fig.add_subplot(gs[0, 2])
    for cn, col in PAL.items():
        if cn not in all_hists: continue
        for fi, h in enumerate(all_hists[cn]):
            ax3.plot(h['train'], alpha=0.3, lw=0.7, color=col,
                    label=f'{cn} tr' if fi==0 else None)
            ax3.plot(h['val'], alpha=0.7, lw=1.0, color=col, ls='--',
                    label=f'{cn} val' if fi==0 else None)
    ax3.axhline(0.2865, color='red', ls='--', lw=1); ax3.axvline(200, color='gray', ls=':')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE')
    ax3.set_title('Training Curves', fontweight='bold'); ax3.legend(fontsize=5, ncol=2)
    ax3.grid(alpha=0.2)

    # 4: Gap tracker
    ax4 = fig.add_subplot(gs[1, 0])
    for cn, col in PAL.items():
        if cn not in all_hists: continue
        for fi, h in enumerate(all_hists[cn]):
            ax4.plot(h['gap'], alpha=0.5, lw=1.0, color=col,
                    label=f'{cn}' if fi==0 else None)
    ax4.axhline(0, color='black', ls='-', lw=0.5)
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('Val - Train')
    ax4.set_title('Generalization Gap', fontweight='bold'); ax4.legend(fontsize=6)
    ax4.grid(alpha=0.2)

    # 5: Heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    heat = np.zeros((2, 2))
    for si, s in enumerate([16, 20]):
        for di, d in enumerate([0.15, 0.20]):
            key = f'100K-S{s}-D{int(d*100):02d}'
            if key in all_results: heat[di, si] = all_results[key]['avg']
    im = ax5.imshow(heat, cmap='RdYlGn_r', aspect='auto',
                    vmin=min(avgs)*0.97, vmax=max(avgs)*1.03)
    ax5.set_xticks([0,1]); ax5.set_xticklabels(['16 Steps', '20 Steps'])
    ax5.set_yticks([0,1]); ax5.set_yticklabels(['Drop 0.15', 'Drop 0.20'])
    for si in range(2):
        for di in range(2):
            ax5.text(si, di, f'{heat[di,si]:.4f}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
    plt.colorbar(im, ax=ax5, label='MAE (eV)')
    ax5.set_title('HP Grid (green=better)', fontweight='bold')

    # 6: Fold difficulty
    ax6 = fig.add_subplot(gs[1, 2])
    fold_avgs = []
    for fi in range(5):
        vals = [all_results[cn]['folds'][fi] for cn in names]
        fold_avgs.append(np.mean(vals))
    colors_fd = ['#E53935' if v == max(fold_avgs) else
                 '#43A047' if v == min(fold_avgs) else '#1E88E5' for v in fold_avgs]
    ax6.bar(range(1, 6), fold_avgs, color=colors_fd, alpha=0.8, edgecolor='white')
    for i, v in enumerate(fold_avgs):
        ax6.text(i+1, v+0.002, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Fold'); ax6.set_ylabel('Avg MAE across models')
    ax6.set_title('Fold Difficulty (red=hardest, green=easiest)', fontweight='bold')
    ax6.grid(axis='y', alpha=0.2)

    fig.suptitle('TRIADS V2 | matbench_expt_gap | 100K Sweet Spot',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('expt_gap_results_v2.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("Saved: expt_gap_results_v2.png")


def save_summary(all_results, all_hists, total_s, n_extra):
    s = {
        'version': 'EG-V2', 'task': 'matbench_expt_gap',
        'dataset_size': 4604, 'target': 'gap expt (eV)',
        'strategy': '100K sweet spot: Steps(16,20) x Dropout(0.15,0.20)',
        'batch_size': BATCH_SIZE,
        'gpu_optimization': 'FastTensorDataLoader + large batch',
        'seeds': SEEDS, 'n_extra': n_extra,
        'total_min': round(total_s/60, 1),
        'kfold': {'n_splits': 5, 'shuffle': True, 'random_state': 18012019},
        'models': {}, 'baselines': BASELINES, 'v1': V1_BEST,
    }
    for n, r in all_results.items():
        s['models'][n] = {
            'avg_mae': round(r['avg'], 4), 'std_mae': round(r['std'], 4),
            'folds': [round(x, 4) for x in r['folds']], 'params': r['params'],
        }
    with open('expt_gap_summary_v2.json', 'w') as f:
        json.dump(s, f, indent=2)
    log.info("Saved: expt_gap_summary_v2.json")


if __name__ == '__main__':
    run_benchmark()
