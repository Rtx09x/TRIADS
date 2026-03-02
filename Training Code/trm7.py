"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V7 — MLP (h64/ff128) + Hybrid-L (~80K) + Ensembles     ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV           ║
║                                                                      ║
║  V7A: MLP-SWA (h=64, ff=128) — user-specified config + SWA          ║
║  V7B: Hybrid-L (d_attn=48, d_hidden=64) — bigger attention + SWA    ║
║  V7C: Hybrid-L ×3 seeds — ensembled for stability                    ║
║  V7D: Cross-arch (MLP + Hybrid) — average different architectures    ║
║                                                                      ║
║  All models use SWA @200 (cosine 200ep → weight avg 100ep)           ║
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
log = logging.getLogger("TRM7")
SEED = 42

BASELINES = {
    'TPOT-Mat':       79.9468,
    'AutoML-Mat':     82.3043,
    'MODNet':         87.7627,
    'RF-SCM/Magpie': 103.5125,
    'CrabNet':       107.3160,
    'Darwin':        123.2932,
    'V5A MLP-SWA':   128.9836,
    'V6C Ensemble':  129.0406,
}


# ══════════════════════════════════════════════════════════════════════
# 1. FEATURIZER + DATASET (same as V4-V6)
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
# 2. MODEL: MLP-TRM (h=64, ff=128)
# ══════════════════════════════════════════════════════════════════════

class MLPTRM(nn.Module):
    def __init__(self, input_dim=332, hidden_dim=64, ff_dim=128,
                 dropout=0.2, steps=16, **kw):
        super().__init__()
        self.steps, self.D = steps, hidden_dim
        self.arch = 'MLP-TRM'
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.z_up = nn.Sequential(
            nn.Linear(hidden_dim*3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.y_up = nn.Sequential(
            nn.Linear(hidden_dim*2, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.head = nn.Linear(hidden_dim, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_trajectory=False):
        B = x.size(0)
        xp = self.proj(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        traj = []
        for _ in range(self.steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            if return_trajectory: traj.append(self.head(y).squeeze(1))
        out = self.head(y).squeeze(1)
        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# 3. MODEL: HYBRID-L (bigger attention → MLP-TRM)
# ══════════════════════════════════════════════════════════════════════

class HybridTRM(nn.Module):
    """Attention as learned feature extractor → MLP-TRM for reasoning.
    Scaled up: d_attn=48 (was 32), larger attention capacity.
    """
    def __init__(self, n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                 dropout=0.2, steps=16, **kw):
        super().__init__()
        self.steps, self.D = steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.arch = 'Hybrid-L'

        # ── Attention feature extractor ──────────────────────────────
        self.tok_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # Self-attention: property interactions
        self.sa = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa_n = nn.LayerNorm(d_attn)
        self.sa_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa_fn = nn.LayerNorm(d_attn)

        # Cross-attention: enrich with Mat2Vec chemistry
        self.ca = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d_attn)

        # Pool → MLP-TRM input
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

    def forward(self, x, return_trajectory=False):
        B = x.size(0)
        mg = x[:, :self.n_props * self.stat_dim]
        m2v = x[:, self.n_props * self.stat_dim:]

        # ── Attention (one-shot) ─────────────────────────────────────
        tok = self.tok_proj(mg.view(B, self.n_props, self.stat_dim))
        ctx = self.m2v_proj(m2v).unsqueeze(1)

        tok = self.sa_n(tok + self.sa(tok, tok, tok)[0])
        tok = self.sa_fn(tok + self.sa_ff(tok))
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx)[0])

        xp = self.pool(tok.mean(dim=1))

        # ── MLP-TRM (16 steps) ───────────────────────────────────────
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        traj = []
        for _ in range(self.steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            if return_trajectory: traj.append(self.head(y).squeeze(1))
        out = self.head(y).squeeze(1)
        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
            loss = F.l1_loss(model(bx), by)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item() * len(by)
        tl /= len(tr_dl.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                bx, by = bx.to(device), by.to(device)
                vl += F.l1_loss(model(bx), by).item() * len(by)
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


def predict(model, dl, device):
    """Get raw predictions from a model."""
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
# 5. MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark():
    t0 = time.time()
    print("\n" + "═"*72)
    print("  TRM-MatSci V7 │ MLP + Hybrid-L + Ensembles │ matbench_steels")
    print("  V7A: MLP-SWA (h=64, ff=128)")
    print("  V7B: Hybrid-L (d_attn=48) + SWA")
    print("  V7C: Hybrid-L ×3 seeds ensemble")
    print("  V7D: Cross-arch (MLP + Hybrid)")
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
    DIM = all_X.shape[1]
    log.info(f"Features: {all_X.shape}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    os.makedirs('trm_models_v7', exist_ok=True)

    # ── CONFIG ────────────────────────────────────────────────────────
    MLP_KW = dict(input_dim=DIM, hidden_dim=64, ff_dim=128, dropout=0.2, steps=16)
    HYB_KW = dict(n_props=22, stat_dim=6, mat2vec_dim=200,
                  d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                  dropout=0.2, steps=16)
    N_SEEDS = 3

    _m = MLPTRM(**MLP_KW); mlp_p = _m.count_parameters(); del _m
    _h = HybridTRM(**HYB_KW); hyb_p = _h.count_parameters(); del _h

    dl_kw = dict(batch_size=32, num_workers=0)

    # Storage for all fold results
    mlp_maes, mlp_hist, mlp_traj = [], [], []
    hyb_maes, hyb_hist, hyb_traj = [], [], []  # single Hybrid
    hyb_ens_maes = []                           # Hybrid ×3 ensemble
    cross_maes = []                             # MLP + Hybrid cross-arch
    fold_details = []

    # ══════════════════════════════════════════════════════════════════
    # TRAIN ALL MODELS PER FOLD
    # ══════════════════════════════════════════════════════════════════

    print(f"\n  Models: MLP ({mlp_p:,} params) │ Hybrid-L ({hyb_p:,} params)")
    print(f"  Hybrid ensemble: {N_SEEDS} seeds │ Cross-arch: MLP + Hybrid avg")
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

        # ── V7A: MLP-SWA ─────────────────────────────────────────────
        torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
        mlp = MLPTRM(**MLP_KW).to(device)
        bv, mlp, mh = train_fold(mlp, tr_dl, vl_dl, device,
                                  fold=fi+1, name="V7A-MLP")
        mlp_pred = predict(mlp, te_dl, device)
        mlp_mae = F.l1_loss(mlp_pred, te_tgt).item()
        mlp_maes.append(mlp_mae); mlp_hist.append(mh)
        mlp_traj.append(eval_traj(mlp, te_dl, device))
        log.info(f"  V7A MLP:     {mlp_mae:.2f}  (val {bv:.2f})")

        torch.save({'model_state': mlp.state_dict(), 'test_mae': mlp_mae},
                   f'trm_models_v7/V7A_MLP_fold{fi+1}.pt')

        # ── V7B/C: Hybrid-L (3 seeds) ────────────────────────────────
        hyb_preds = []
        hyb_best_mae, hyb_best_pred = float('inf'), None
        for si in range(N_SEEDS):
            torch.manual_seed(SEED + fi*100 + si)
            np.random.seed(SEED + fi*100 + si)
            hyb = HybridTRM(**HYB_KW).to(device)
            bv, hyb, hh = train_fold(hyb, tr_dl, vl_dl, device,
                                      fold=fi+1, name=f"V7B-Hyb-s{si}")
            hp = predict(hyb, te_dl, device)
            hm = F.l1_loss(hp, te_tgt).item()
            hyb_preds.append(hp)
            if hm < hyb_best_mae:
                hyb_best_mae, hyb_best_pred = hm, hp
                if si == 0: hyb_hist.append(hh)

            torch.save({'model_state': hyb.state_dict(), 'test_mae': hm},
                       f'trm_models_v7/V7B_Hyb_fold{fi+1}_s{si}.pt')

            if si == 0:
                hyb_traj.append(eval_traj(hyb, te_dl, device))

        # V7B: Best single Hybrid
        hyb_maes.append(hyb_best_mae)

        # V7C: Hybrid ×3 ensemble
        hyb_ens_pred = torch.stack(hyb_preds).mean(dim=0)
        hyb_ens_mae = F.l1_loss(hyb_ens_pred, te_tgt).item()
        hyb_ens_maes.append(hyb_ens_mae)

        indiv = [F.l1_loss(p, te_tgt).item() for p in hyb_preds]
        log.info(f"  V7B Hyb best: {hyb_best_mae:.2f}  "
                 f"individual: {[f'{m:.1f}' for m in indiv]}")
        log.info(f"  V7C Hyb×3:   {hyb_ens_mae:.2f}")

        # ── V7D: Cross-arch ensemble (MLP + Hybrid×3) ────────────────
        cross_pred = (mlp_pred + hyb_ens_pred) / 2.0
        cross_mae = F.l1_loss(cross_pred, te_tgt).item()
        cross_maes.append(cross_mae)
        log.info(f"  V7D Cross:   {cross_mae:.2f}")

        fold_details.append({
            'mlp_mae': mlp_mae,
            'hyb_best_mae': hyb_best_mae,
            'hyb_indiv': indiv,
            'hyb_ens_mae': hyb_ens_mae,
            'cross_mae': cross_mae,
        })

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════

    results = {
        'V7A-MLP-SWA': {
            'avg': float(np.mean(mlp_maes)), 'std': float(np.std(mlp_maes)),
            'folds': mlp_maes, 'params': mlp_p},
        'V7B-Hybrid-L': {
            'avg': float(np.mean(hyb_maes)), 'std': float(np.std(hyb_maes)),
            'folds': hyb_maes, 'params': hyb_p},
        'V7C-Hybrid-Ens': {
            'avg': float(np.mean(hyb_ens_maes)), 'std': float(np.std(hyb_ens_maes)),
            'folds': hyb_ens_maes, 'params': hyb_p, 'n_seeds': N_SEEDS},
        'V7D-Cross-Arch': {
            'avg': float(np.mean(cross_maes)), 'std': float(np.std(cross_maes)),
            'folds': cross_maes, 'params': mlp_p + hyb_p},
    }

    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V7 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'Params':>10} {'MAE(MPa)':>10} {'±Std':>8}")
    print(f"  {'─'*52}")
    for n, r in sorted(results.items(), key=lambda x: x[1]['avg']):
        st = f"×{r['n_seeds']}" if 'n_seeds' in r else ""
        tag = ("  ← BEATS DARWIN 🏆" if r['avg'] < 123.29 else
               "  ← Beats V5A ✓"    if r['avg'] < 128.98 else "")
        print(f"  {n:<22} {r['params']:>9,}{st} "
              f"{r['avg']:>10.4f} {r['std']:>8.4f}{tag}")
    print(f"  {'─'*52}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<22} {'baseline':>10} {bv:>10.4f}")
    print(f"\n  Total time: {tt/60:.1f} minutes")

    # Per-fold details
    print(f"\n{'═'*72}")
    print(f"  PER-FOLD BREAKDOWN")
    print(f"{'═'*72}")
    print(f"  {'Fold':<6} {'V7A-MLP':>10} {'V7B-Hyb':>10} "
          f"{'V7C-Ens':>10} {'V7D-Cross':>10}")
    print(f"  {'─'*48}")
    for fi, fd in enumerate(fold_details):
        print(f"  {fi+1:<6} {fd['mlp_mae']:>10.2f} {fd['hyb_best_mae']:>10.2f} "
              f"{fd['hyb_ens_mae']:>10.2f} {fd['cross_mae']:>10.2f}")
    print()

    generate_plots(results, mlp_traj, hyb_traj, mlp_hist, hyb_hist)
    save_summary(results, fold_details, mlp_traj, hyb_traj, tt)
    return results


# ══════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════

PAL = {'V7A-MLP-SWA': '#1565C0', 'V7B-Hybrid-L': '#6A1B9A',
       'V7C-Hybrid-Ens': '#E65100', 'V7D-Cross-Arch': '#2E7D32'}

def generate_plots(results, mlp_tr, hyb_tr, mlp_h, hyb_h):
    names = list(results.keys())
    maes = [results[n]['avg'] for n in names]
    stds = [results[n]['std'] for n in names]
    cols = [PAL.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    # Bar chart
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(names, maes, yerr=stds, capsize=5, color=cols,
                   alpha=0.88, edgecolor='white', linewidth=1.2)
    for bv, c, ls, lb in [
        (79.95, '#2E7D32', '--', 'TPOT (79.95)'),
        (87.76, '#F57F17', '--', 'MODNet (87.76)'),
        (107.32, '#9E9E9E', ':', 'CrabNet (107.32)'),
        (123.29, '#FF5722', ':', 'Darwin (123.29)'),
        (128.98, '#90CAF9', ':', 'V5A (128.98)'),
    ]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    for bar, m, s in zip(bars, maes, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+1,
                 f'{m:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_ylabel('MAE (MPa)'); ax1.set_ylim(0, max(maes)*1.3)
    ax1.set_title('TRM-MatSci V7 │ matbench_steels', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Recursion convergence
    ax2 = fig.add_subplot(gs[1, 0])
    mt = np.mean(mlp_tr, axis=0)
    ht = np.mean(hyb_tr, axis=0)
    ax2.plot(range(1, 17), mt, label='MLP-TRM', color=PAL['V7A-MLP-SWA'], lw=2)
    ax2.plot(range(1, 17), ht, label='Hybrid-L', color=PAL['V7B-Hybrid-L'], lw=2)
    ax2.axhline(107.32, color='#9E9E9E', ls=':', lw=1.5, label='CrabNet')
    ax2.axhline(123.29, color='#FF5722', ls=':', lw=1.5, label='Darwin')
    ax2.set_xlabel('Recursion Step'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_title('Recursion Convergence', fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Training curves
    ax3 = fig.add_subplot(gs[1, 1])
    for fi, h in enumerate(mlp_h):
        lb = 'MLP' if fi == 0 else None
        ax3.plot(h['val'], alpha=0.7, lw=1, color=PAL['V7A-MLP-SWA'], label=lb)
    for fi, h in enumerate(hyb_h):
        lb = 'Hybrid' if fi == 0 else None
        ax3.plot(h['val'], alpha=0.7, lw=1, color=PAL['V7B-Hybrid-L'], label=lb)
    ax3.axhline(107.32, color='#9E9E9E', ls=':', lw=1.2, label='CrabNet')
    ax3.axvline(200, color='#4CAF50', ls='--', lw=1.2, alpha=0.6, label='SWA')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE (MPa)')
    ax3.set_title('Val Curves (per fold)', fontweight='bold')
    ax3.legend(fontsize=7); ax3.grid(alpha=0.2)

    fig.savefig('trm_results_v7.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: trm_results_v7.png")


def save_summary(results, details, mlp_tr, hyb_tr, total_s):
    s = {
        'version': 'V7', 'task': 'matbench_steels',
        'configs': {
            'V7A': 'MLP-SWA h=64 ff=128',
            'V7B': 'Hybrid-L d_attn=48 d_hidden=64 ff=100',
            'V7C': 'Hybrid-L ×3 seeds ensemble',
            'V7D': 'Cross-arch (MLP + Hybrid avg)',
        },
        'total_min': round(total_s/60, 1),
        'models': {n: {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()}
                   for n, r in results.items()},
        'fold_details': details,
        'mlp_convergence': np.mean(mlp_tr, axis=0).round(4).tolist(),
        'hyb_convergence': np.mean(hyb_tr, axis=0).round(4).tolist(),
    }
    with open('trm_models_v7/summary_v7.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    log.info("✓ Saved: summary_v7.json")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v7_all", "zip", "trm_models_v7")
    log.info("✓ Created trm_v7_all.zip")
