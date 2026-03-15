"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V10 — Adaptive Recursion with Deep Supervision          ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV          ║
║                                                                      ║
║  KEY INNOVATION: Deep supervision — loss at every recursion step     ║
║  so all 20 steps produce calibrated predictions. Then:               ║
║                                                                      ║
║  V10A: Fixed 20 steps (deep supervision only)                        ║
║  V10B: Adaptive halting — stop each sample when prediction converges ║
║        (delta < ε=1.0 MPa, min_steps=12)                            ║
║                                                                      ║
║  Same model trained once per fold, evaluated twice.                  ║
║  Architecture: V7B Hybrid-TRM (SA → FF → CA → pool → MLP-TRM)      ║
║  Target: Beat Darwin (123.29 MPa)                                    ║
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
log = logging.getLogger("TRM10")
SEED = 42

BASELINES = {
    'TPOT-Mat':       79.9468,
    'AutoML-Mat':     82.3043,
    'MODNet':         87.7627,
    'RF-SCM/Magpie': 103.5125,
    'CrabNet':       107.3160,
    'Darwin':        123.2932,
    'V7B Hybrid-L':  127.0782,
    'V5A MLP-SWA':   128.9836,
}


# ══════════════════════════════════════════════════════════════════════
# 1. FEATURIZER + DATASET
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
# 2. MODEL: HYBRID-TRM with DEEP SUPERVISION
# ══════════════════════════════════════════════════════════════════════

class HybridTRM(nn.Module):
    """V7B Hybrid-TRM with deep supervision support.

    Architecture (unchanged from V7B):
      Magpie [22 × 6] → tok_proj → [22, d_attn]
      SA → FF → CA(Mat2Vec) → mean pool → MLP-TRM (max_steps recursive)

    New forward modes:
      - forward(x): standard — returns final prediction (for inference)
      - forward(x, deep_supervision=True): returns list of predictions at
        every step (for training with per-step loss)
      - forward(x, return_trajectory=True): returns final pred + per-step
        preds (for convergence analysis)
      - adaptive_forward(x, ...): runs with adaptive halting per sample
    """
    def __init__(self, n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                 dropout=0.2, max_steps=20, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.arch = f'Hybrid-DS-{max_steps}s'

        # ── Attention feature extractor (unchanged from V7B) ─────────
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

        # ── MLP-TRM recursive reasoning ─────────────────────────────
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
        """Shared attention feature extraction."""
        B = x.size(0)
        mg = x[:, :self.n_props * self.stat_dim]
        m2v = x[:, self.n_props * self.stat_dim:]

        tok = self.tok_proj(mg.view(B, self.n_props, self.stat_dim))
        ctx = self.m2v_proj(m2v).unsqueeze(1)

        tok = self.sa_n(tok + self.sa(tok, tok, tok)[0])
        tok = self.sa_fn(tok + self.sa_ff(tok))
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx)[0])

        return self.pool(tok.mean(dim=1))  # [B, d_hidden]

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
            return step_preds  # list of [B] tensors, one per step
        elif return_trajectory:
            return step_preds[-1], step_preds
        else:
            return step_preds[-1]

    def adaptive_forward(self, x, min_steps=12, epsilon=1.0):
        """Inference with per-sample adaptive halting.

        Each sample stops when |pred_t - pred_{t-1}| < epsilon.
        All samples run at least min_steps.
        Returns final predictions and the step each sample halted at.
        """
        B = x.size(0)
        xp = self._attention(x)

        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)

        final_pred = torch.zeros(B, device=x.device)
        halted = torch.zeros(B, dtype=torch.bool, device=x.device)
        halt_step = torch.full((B,), self.max_steps, dtype=torch.long,
                               device=x.device)
        prev_pred = None

        for step in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            pred = self.head(y).squeeze(1)

            if step >= min_steps and prev_pred is not None:
                delta = (pred - prev_pred).abs()
                newly_halted = (~halted) & (delta < epsilon)
                if newly_halted.any():
                    final_pred[newly_halted] = pred[newly_halted]
                    halt_step[newly_halted] = step + 1
                    halted = halted | newly_halted

            prev_pred = pred.clone()

        # Samples that never halted use the final step prediction
        still_running = ~halted
        if still_running.any():
            final_pred[still_running] = pred[still_running]

        return final_pred, halt_step

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# 3. DEEP SUPERVISION LOSS
# ══════════════════════════════════════════════════════════════════════

def deep_supervision_loss(step_preds, targets, weighting='linear'):
    """Compute weighted L1 loss across all recursion steps.

    Args:
        step_preds: list of [B] tensors, one per step
        targets: [B] tensor of true values
        weighting: 'linear' (later steps weighted more) or 'uniform'

    Linear weighting: step 1 gets w=1, step 20 gets w=20.
    This incentivizes good predictions at every step while optimizing
    most aggressively for the final, most-refined prediction.
    """
    n = len(step_preds)
    if weighting == 'linear':
        weights = [(i + 1) for i in range(n)]
    else:
        weights = [1.0] * n

    total_w = sum(weights)
    loss = 0.0
    for pred, w in zip(step_preds, weights):
        loss += (w / total_w) * F.l1_loss(pred, targets)
    return loss


# ══════════════════════════════════════════════════════════════════════
# 4. UTILS + TRAINING
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
    """Training with deep supervision loss."""
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
        # ── Train with deep supervision ──────────────────────────────
        model.train(); tl = 0.0
        for bx, by in tr_dl:
            bx, by = bx.to(device), by.to(device)
            step_preds = model(bx, deep_supervision=True)
            loss = deep_supervision_loss(step_preds, by, weighting='linear')
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # Track final step MAE for display
            tl += F.l1_loss(step_preds[-1], by).item() * len(by)
        tl /= len(tr_dl.dataset)

        # ── Validate on final step prediction ────────────────────────
        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)  # standard forward, final step only
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


def eval_traj(model, dl, device):
    """Get per-step MAE trajectory on test data."""
    model.eval(); sp, tg = None, []
    with torch.no_grad():
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            _, tr = model(bx, return_trajectory=True)
            if sp is None: sp = [[] for _ in range(len(tr))]
            for i, s in enumerate(tr): sp[i].extend(s.cpu().numpy().tolist())
            tg.extend(by.cpu().numpy().tolist())
    t = np.array(tg)
    return [float(np.mean(np.abs(np.array(p) - t))) for p in sp]


def predict_fixed(model, dl, device):
    """Standard prediction using all max_steps."""
    model.eval(); preds = []
    with torch.no_grad():
        for bx, _ in dl:
            preds.append(model(bx.to(device)).cpu())
    return torch.cat(preds)


def predict_adaptive(model, dl, device, min_steps=12, epsilon=1.0):
    """Adaptive prediction — each sample halts independently."""
    model.eval()
    all_preds, all_steps = [], []
    with torch.no_grad():
        for bx, _ in dl:
            pred, steps = model.adaptive_forward(
                bx.to(device), min_steps=min_steps, epsilon=epsilon)
            all_preds.append(pred.cpu())
            all_steps.append(steps.cpu())
    return torch.cat(all_preds), torch.cat(all_steps)


def get_targets(dl):
    tgts = []
    for _, by in dl: tgts.append(by)
    return torch.cat(tgts)


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark():
    t0 = time.time()
    print("\n" + "═"*72)
    print("  TRM-MatSci V10 │ Adaptive Recursion │ matbench_steels")
    print("  Deep supervision (loss at every step) + adaptive halting")
    print("  V10A: Fixed 20 steps │ V10B: Adaptive halt (ε=1.0, min=12)")
    print("  Same model weights — train once, evaluate twice")
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
    os.makedirs('trm_models_v10', exist_ok=True)

    # ── CONFIG ────────────────────────────────────────────────────────
    MODEL_KW = dict(n_props=22, stat_dim=6, mat2vec_dim=200,
                    d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                    dropout=0.2, max_steps=20)

    _m = HybridTRM(**MODEL_KW); n_params = _m.count_parameters(); del _m
    dl_kw = dict(batch_size=32, num_workers=0)

    # Storage
    fixed_maes, adapt_maes = [], []
    all_hist, all_traj = [], []
    fold_details = []
    all_halt_stats = []

    print(f"\n  Model: Hybrid-DS-20 ({n_params:,} params)")
    print(f"  d_attn=48, nhead=2, d_hidden=64, ff_dim=100, max_steps=20")
    print(f"  Training: deep supervision (linear weights) + SWA@200")
    print(f"{'─'*72}")

    for fi, (tv_i, te_i) in enumerate(folds):
        print(f"\n  ── Fold {fi+1}/5 {'─'*56}")

        tri, vli = strat_split(targets_all[tv_i], 0.15, SEED+fi)
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

        # ── TRAIN (one model, deep supervision) ──────────────────────
        torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
        model = HybridTRM(**MODEL_KW).to(device)
        bv, model, hist = train_fold(model, tr_dl, vl_dl, device,
                                      fold=fi+1, name="V10-DS")
        all_hist.append(hist)

        # ── EVALUATE: V10A — Fixed 20 steps ──────────────────────────
        pred_fixed = predict_fixed(model, te_dl, device)
        mae_fixed = F.l1_loss(pred_fixed, te_tgt).item()
        fixed_maes.append(mae_fixed)

        # ── EVALUATE: V10B — Adaptive halting ─────────────────────────
        pred_adapt, halt_steps = predict_adaptive(
            model, te_dl, device, min_steps=12, epsilon=1.0)
        mae_adapt = F.l1_loss(pred_adapt, te_tgt).item()
        adapt_maes.append(mae_adapt)

        # Halt statistics
        hs = halt_steps.float()
        halt_info = {
            'mean': float(hs.mean()),
            'min': int(hs.min()),
            'max': int(hs.max()),
            'pct_early': float((hs < 20).float().mean() * 100),
        }
        all_halt_stats.append(halt_info)

        # Convergence trajectory
        traj = eval_traj(model, te_dl, device)
        all_traj.append(traj)

        log.info(f"  V10A fixed:    {mae_fixed:.2f}  (val {bv:.2f})")
        log.info(f"  V10B adaptive: {mae_adapt:.2f}  "
                 f"(avg halt={halt_info['mean']:.1f}, "
                 f"{halt_info['pct_early']:.0f}% early)")

        torch.save({'model_state': model.state_dict(),
                    'test_mae_fixed': mae_fixed,
                    'test_mae_adaptive': mae_adapt,
                    'halt_stats': halt_info},
                   f'trm_models_v10/V10_fold{fi+1}.pt')

        fold_details.append({
            'fixed_mae': mae_fixed,
            'adapt_mae': mae_adapt,
            'halt_stats': halt_info,
        })

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════

    results = {
        'V10A-Fixed-20': {
            'avg': float(np.mean(fixed_maes)), 'std': float(np.std(fixed_maes)),
            'folds': fixed_maes, 'params': n_params},
        'V10B-Adaptive': {
            'avg': float(np.mean(adapt_maes)), 'std': float(np.std(adapt_maes)),
            'folds': adapt_maes, 'params': n_params},
    }

    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V10 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'Params':>10} {'MAE(MPa)':>10} {'±Std':>8}")
    print(f"  {'─'*52}")
    for n, r in sorted(results.items(), key=lambda x: x[1]['avg']):
        tag = ("  ← BEATS DARWIN 🏆" if r['avg'] < 123.29 else
               "  ← Beats V7B ✓"    if r['avg'] < 127.08 else "")
        print(f"  {n:<22} {r['params']:>9,} "
              f"{r['avg']:>10.4f} {r['std']:>8.4f}{tag}")
    print(f"  {'─'*52}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<22} {'baseline':>10} {bv:>10.4f}")
    print(f"\n  Total time: {tt/60:.1f} minutes")

    # Per-fold details
    print(f"\n{'═'*72}")
    print(f"  PER-FOLD BREAKDOWN")
    print(f"{'═'*72}")
    print(f"  {'Fold':<6} {'V10A-Fix':>10} {'V10B-Adp':>10} "
          f"{'AvgHalt':>8} {'%Early':>7}")
    print(f"  {'─'*43}")
    for fi, fd in enumerate(fold_details):
        hs = fd['halt_stats']
        print(f"  {fi+1:<6} {fd['fixed_mae']:>10.2f} {fd['adapt_mae']:>10.2f} "
              f"{hs['mean']:>8.1f} {hs['pct_early']:>6.0f}%")

    # Halt distribution summary
    print(f"\n  Adaptive Halting Summary:")
    avg_halt = np.mean([h['mean'] for h in all_halt_stats])
    avg_early = np.mean([h['pct_early'] for h in all_halt_stats])
    print(f"    Average halt step: {avg_halt:.1f} / 20")
    print(f"    Samples stopping early: {avg_early:.0f}%")
    print()

    generate_plots(results, all_traj, all_hist, all_halt_stats)
    save_summary(results, fold_details, all_traj, all_halt_stats, tt)
    return results


# ══════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════

PAL = {'V10A-Fixed-20': '#1565C0', 'V10B-Adaptive': '#E65100'}

def generate_plots(results, all_traj, all_hist, halt_stats):
    names = list(results.keys())
    maes = [results[n]['avg'] for n in names]
    stds = [results[n]['std'] for n in names]
    cols = [PAL.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    # Bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(names, maes, yerr=stds, capsize=5, color=cols,
                   alpha=0.88, edgecolor='white', linewidth=1.2)
    for bv, c, ls, lb in [
        (87.76, '#F57F17', '--', 'MODNet (87.76)'),
        (107.32, '#9E9E9E', ':', 'CrabNet (107.32)'),
        (123.29, '#FF5722', ':', 'Darwin (123.29)'),
        (127.08, '#90CAF9', ':', 'V7B (127.08)'),
    ]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    for bar, m, s in zip(bars, maes, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+1,
                 f'{m:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_ylabel('MAE (MPa)'); ax1.set_ylim(0, max(maes)*1.4)
    ax1.set_title('V10 Results', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Recursion convergence (20 steps, deep supervised)
    ax2 = fig.add_subplot(gs[0, 1])
    avg_traj = np.mean(all_traj, axis=0)
    steps = range(1, len(avg_traj)+1)
    ax2.plot(steps, avg_traj, color=PAL['V10A-Fixed-20'], lw=2.5,
             label='Deep-supervised convergence')
    ax2.axhline(107.32, color='#9E9E9E', ls=':', lw=1.5, label='CrabNet')
    ax2.axhline(123.29, color='#FF5722', ls=':', lw=1.5, label='Darwin')
    ax2.axhline(127.08, color='#90CAF9', ls=':', lw=1.5, label='V7B')
    ax2.axvline(16, color='#4CAF50', ls='--', lw=1, alpha=0.5, label='V7B cutoff')
    avg_halt = np.mean([h['mean'] for h in halt_stats])
    ax2.axvline(avg_halt, color=PAL['V10B-Adaptive'], ls='--', lw=1.5,
                label=f'Avg halt ({avg_halt:.1f})')
    ax2.set_xlabel('Recursion Step'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_title('Deep-Supervised Convergence (20 steps)', fontweight='bold')
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # Training curves
    ax3 = fig.add_subplot(gs[1, 0])
    for fi, h in enumerate(all_hist):
        lb = 'Val MAE' if fi == 0 else None
        ax3.plot(h['val'], alpha=0.7, lw=1, color=PAL['V10A-Fixed-20'], label=lb)
    ax3.axhline(107.32, color='#9E9E9E', ls=':', lw=1.2, label='CrabNet')
    ax3.axvline(200, color='#4CAF50', ls='--', lw=1.2, alpha=0.6, label='SWA')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE (MPa)')
    ax3.set_title('Training Curves', fontweight='bold')
    ax3.legend(fontsize=7); ax3.grid(alpha=0.2)

    # Halt step distribution
    ax4 = fig.add_subplot(gs[1, 1])
    halt_means = [h['mean'] for h in halt_stats]
    pct_early = [h['pct_early'] for h in halt_stats]
    x = range(1, 6)
    ax4.bar(x, halt_means, color=PAL['V10B-Adaptive'], alpha=0.8, label='Avg halt step')
    ax4.axhline(20, color='#9E9E9E', ls=':', lw=1.5, label='Max steps')
    ax4.axhline(16, color='#4CAF50', ls='--', lw=1, alpha=0.5, label='V7B steps')
    ax4.set_xlabel('Fold'); ax4.set_ylabel('Average Halt Step')
    ax4.set_title('Adaptive Halting per Fold', fontweight='bold')
    ax4.set_ylim(0, 22)
    # Add % early as text
    for i, (hm, pe) in enumerate(zip(halt_means, pct_early)):
        ax4.text(i+1, hm+0.5, f'{pe:.0f}%\nearly', ha='center', fontsize=8)
    ax4.legend(fontsize=7); ax4.grid(axis='y', alpha=0.2)

    fig.suptitle('TRM-MatSci V10 │ Adaptive Recursion │ matbench_steels',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v10.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: trm_results_v10.png")


def save_summary(results, details, all_traj, halt_stats, total_s):
    s = {
        'version': 'V10', 'task': 'matbench_steels',
        'innovation': 'Deep supervision + adaptive halting',
        'configs': {
            'V10A': 'Fixed 20 steps with deep supervision (linear weights)',
            'V10B': 'Adaptive halt (epsilon=1.0, min_steps=12)',
        },
        'total_min': round(total_s/60, 1),
        'models': {n: {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()}
                   for n, r in results.items()},
        'fold_details': details,
        'convergence': np.mean(all_traj, axis=0).round(4).tolist(),
        'halt_stats': halt_stats,
    }
    with open('trm_models_v10/summary_v10.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    log.info("✓ Saved: summary_v10.json")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v10_all", "zip", "trm_models_v10")
    log.info("✓ Created trm_v10_all.zip")
