"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V8 — Hybrid-M2V (SA injection) + Hybrid-XL             ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV          ║
║                                                                      ║
║  V8A: Hybrid-M2V (d_attn=48) — Mat2Vec as 23rd self-attention token ║
║       22 Magpie property tokens + 1 Mat2Vec chemistry token         ║
║       Mutual SA over all 23 → no separate cross-attention needed    ║
║       Fewer layers, richer interaction (early fusion > late fusion)  ║
║                                                                      ║
║  V8B: Hybrid-XL (d_attn=64, nhead=2) — wider attention             ║
║       Scale up attention from V7B's d_attn=48 → 64                  ║
║       32 dims/head (safe zone from V1 findings)                      ║
║                                                                      ║
║  Both use SWA @200 (cosine 200ep → weight avg 100ep)                ║
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
log = logging.getLogger("TRM8")
SEED = 42

BASELINES = {
    'TPOT-Mat':       79.9468,
    'AutoML-Mat':     82.3043,
    'MODNet':         87.7627,
    'RF-SCM/Magpie': 103.5125,
    'CrabNet':       107.3160,
    'Darwin':        123.2932,
    'V5A MLP-SWA':   128.9836,
    'V7B Hybrid-L':  127.0782,
}


# ══════════════════════════════════════════════════════════════════════
# 1. FEATURIZER + DATASET (same as V4-V7)
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
# 2. MODEL: HYBRID-M2V (d_attn=48 + Mat2Vec as SA token)
# ══════════════════════════════════════════════════════════════════════

class HybridM2V(nn.Module):
    """Hybrid-TRM with Mat2Vec injected as a self-attention token.

    V7B used Mat2Vec ONLY as cross-attention context (1 token, 22→1).
    V8A makes Mat2Vec a FULL PARTICIPANT in self-attention:
      - 22 Magpie property tokens + 1 Mat2Vec chemistry token = 23 tokens
      - All 23 tokens mutually attend to each other
      - Mat2Vec can learn: "given this chemical signature, electronegativity
        range matters more than atomic radius stats"
      - Property tokens can learn: "this composition's chemistry suggests
        carbide formation patterns"

    This removes the separate cross-attention layer entirely — Mat2Vec
    already participates fully in self-attention. Fewer layers, richer
    interaction. Follows the project's core lesson: early interaction
    beats late fusion (V1→V3 proved this).

    Architecture:
      Magpie [22, 6] → tok_proj → [22, d_attn]
      Mat2Vec [200]  → m2v_proj → [1, d_attn]
      Concat → [23, d_attn] → SA → FF → pool → MLP-TRM (16 steps)
    """
    def __init__(self, n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                 dropout=0.2, steps=16, **kw):
        super().__init__()
        self.steps, self.D = steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.arch = 'Hybrid-M2V'

        # ── Token projections ────────────────────────────────────────
        # Magpie: 22 property tokens, each 6-dim (stats) → d_attn
        self.tok_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        # Mat2Vec: 1 chemistry token, 200-dim → d_attn
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # ── Self-attention over ALL 23 tokens (Magpie + Mat2Vec) ─────
        self.sa = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa_n = nn.LayerNorm(d_attn)
        self.sa_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa_fn = nn.LayerNorm(d_attn)

        # No cross-attention needed — Mat2Vec is already in SA

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

        # ── Build 23-token sequence ──────────────────────────────────
        prop_tok = self.tok_proj(mg.view(B, self.n_props, self.stat_dim))  # [B, 22, d_attn]
        m2v_tok  = self.m2v_proj(m2v).unsqueeze(1)                         # [B, 1, d_attn]
        tok = torch.cat([prop_tok, m2v_tok], dim=1)                        # [B, 23, d_attn]

        # ── Self-attention (all 23 tokens interact mutually) ─────────
        tok = self.sa_n(tok + self.sa(tok, tok, tok)[0])
        tok = self.sa_fn(tok + self.sa_ff(tok))

        # ── Pool all 23 tokens → single vector ──────────────────────
        xp = self.pool(tok.mean(dim=1))  # [B, d_hidden]

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
# 3. MODEL: HYBRID-XL (d_attn=64, nhead=2 → 32 dims/head)
# ══════════════════════════════════════════════════════════════════════

class HybridXL(nn.Module):
    """Hybrid-TRM with wider attention (d_attn=64 vs V7B's 48).

    d_attn 32→48 gave 7.9 MPa improvement (V6B→V7B).
    V8B tests whether 48→64 continues the trend.
    nhead=2 gives 32 dims/head (safe zone confirmed from V1).
    """
    def __init__(self, n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_attn=64, nhead=2, d_hidden=64, ff_dim=100,
                 dropout=0.2, steps=16, **kw):
        super().__init__()
        self.steps, self.D = steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.arch = 'Hybrid-XL'

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
# 4. UTILS + TRAINING (same as V7)
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
    print("  TRM-MatSci V8 │ Hybrid-M2V + Hybrid-XL │ matbench_steels")
    print("  V8A: Hybrid-M2V (d_attn=48) — Mat2Vec as 23rd SA token")
    print("  V8B: Hybrid-XL  (d_attn=64) — wider attention")
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
    os.makedirs('trm_models_v8', exist_ok=True)

    # ── CONFIG ────────────────────────────────────────────────────────
    M2V_KW = dict(n_props=22, stat_dim=6, mat2vec_dim=200,
                  d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                  dropout=0.2, steps=16)
    XL_KW = dict(n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_attn=64, nhead=2, d_hidden=64, ff_dim=100,
                 dropout=0.2, steps=16)

    _a = HybridM2V(**M2V_KW); m2v_p = _a.count_parameters(); del _a
    _b = HybridXL(**XL_KW);   xl_p  = _b.count_parameters(); del _b

    dl_kw = dict(batch_size=32, num_workers=0)

    # Storage
    m2v_maes, m2v_hist, m2v_traj = [], [], []
    xl_maes,  xl_hist,  xl_traj  = [], [], []
    fold_details = []

    # ══════════════════════════════════════════════════════════════════
    # TRAIN BOTH MODELS PER FOLD
    # ══════════════════════════════════════════════════════════════════

    print(f"\n  Models: Hybrid-M2V ({m2v_p:,} params) │ Hybrid-XL ({xl_p:,} params)")
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

        # ── V8A: Hybrid-M2V ──────────────────────────────────────────
        torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
        m2v_model = HybridM2V(**M2V_KW).to(device)
        bv, m2v_model, mh = train_fold(m2v_model, tr_dl, vl_dl, device,
                                        fold=fi+1, name="V8A-M2V")
        m2v_pred = predict(m2v_model, te_dl, device)
        m2v_mae = F.l1_loss(m2v_pred, te_tgt).item()
        m2v_maes.append(m2v_mae); m2v_hist.append(mh)
        m2v_traj.append(eval_traj(m2v_model, te_dl, device))
        log.info(f"  V8A M2V:  {m2v_mae:.2f}  (val {bv:.2f})")

        torch.save({'model_state': m2v_model.state_dict(), 'test_mae': m2v_mae},
                   f'trm_models_v8/V8A_M2V_fold{fi+1}.pt')

        # ── V8B: Hybrid-XL ───────────────────────────────────────────
        torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
        xl_model = HybridXL(**XL_KW).to(device)
        bv, xl_model, xh = train_fold(xl_model, tr_dl, vl_dl, device,
                                       fold=fi+1, name="V8B-XL")
        xl_pred = predict(xl_model, te_dl, device)
        xl_mae = F.l1_loss(xl_pred, te_tgt).item()
        xl_maes.append(xl_mae); xl_hist.append(xh)
        xl_traj.append(eval_traj(xl_model, te_dl, device))
        log.info(f"  V8B XL:   {xl_mae:.2f}  (val {bv:.2f})")

        torch.save({'model_state': xl_model.state_dict(), 'test_mae': xl_mae},
                   f'trm_models_v8/V8B_XL_fold{fi+1}.pt')

        fold_details.append({
            'm2v_mae': m2v_mae,
            'xl_mae': xl_mae,
        })

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════

    results = {
        'V8A-Hybrid-M2V': {
            'avg': float(np.mean(m2v_maes)), 'std': float(np.std(m2v_maes)),
            'folds': m2v_maes, 'params': m2v_p},
        'V8B-Hybrid-XL': {
            'avg': float(np.mean(xl_maes)), 'std': float(np.std(xl_maes)),
            'folds': xl_maes, 'params': xl_p},
    }

    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V8 (5-Fold Avg MAE)")
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
    print(f"  {'Fold':<6} {'V8A-M2V':>10} {'V8B-XL':>10}")
    print(f"  {'─'*28}")
    for fi, fd in enumerate(fold_details):
        print(f"  {fi+1:<6} {fd['m2v_mae']:>10.2f} {fd['xl_mae']:>10.2f}")
    print()

    generate_plots(results, m2v_traj, xl_traj, m2v_hist, xl_hist)
    save_summary(results, fold_details, m2v_traj, xl_traj, tt)
    return results


# ══════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════

PAL = {'V8A-Hybrid-M2V': '#1565C0', 'V8B-Hybrid-XL': '#6A1B9A'}

def generate_plots(results, m2v_tr, xl_tr, m2v_h, xl_h):
    names = list(results.keys())
    maes = [results[n]['avg'] for n in names]
    stds = [results[n]['std'] for n in names]
    cols = [PAL.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(18, 12))
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
        (127.08, '#90CAF9', ':', 'V7B (127.08)'),
        (128.98, '#B0BEC5', ':', 'V5A (128.98)'),
    ]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    for bar, m, s in zip(bars, maes, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+1,
                 f'{m:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_ylabel('MAE (MPa)'); ax1.set_ylim(0, max(maes)*1.3)
    ax1.set_title('TRM-MatSci V8 │ matbench_steels', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Recursion convergence
    ax2 = fig.add_subplot(gs[1, 0])
    mt = np.mean(m2v_tr, axis=0)
    xt = np.mean(xl_tr, axis=0)
    ax2.plot(range(1, 17), mt, label='Hybrid-M2V', color=PAL['V8A-Hybrid-M2V'], lw=2)
    ax2.plot(range(1, 17), xt, label='Hybrid-XL', color=PAL['V8B-Hybrid-XL'], lw=2)
    ax2.axhline(107.32, color='#9E9E9E', ls=':', lw=1.5, label='CrabNet')
    ax2.axhline(123.29, color='#FF5722', ls=':', lw=1.5, label='Darwin')
    ax2.set_xlabel('Recursion Step'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_title('Recursion Convergence', fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Training curves
    ax3 = fig.add_subplot(gs[1, 1])
    for fi, h in enumerate(m2v_h):
        lb = 'Hybrid-M2V' if fi == 0 else None
        ax3.plot(h['val'], alpha=0.7, lw=1, color=PAL['V8A-Hybrid-M2V'], label=lb)
    for fi, h in enumerate(xl_h):
        lb = 'Hybrid-XL' if fi == 0 else None
        ax3.plot(h['val'], alpha=0.7, lw=1, color=PAL['V8B-Hybrid-XL'], label=lb)
    ax3.axhline(107.32, color='#9E9E9E', ls=':', lw=1.2, label='CrabNet')
    ax3.axvline(200, color='#4CAF50', ls='--', lw=1.2, alpha=0.6, label='SWA')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE (MPa)')
    ax3.set_title('Val Curves (per fold)', fontweight='bold')
    ax3.legend(fontsize=7); ax3.grid(alpha=0.2)

    fig.savefig('trm_results_v8.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: trm_results_v8.png")


def save_summary(results, details, m2v_tr, xl_tr, total_s):
    s = {
        'version': 'V8', 'task': 'matbench_steels',
        'configs': {
            'V8A': 'Hybrid-M2V d_attn=48 + Mat2Vec as 23rd SA token (no cross-attn)',
            'V8B': 'Hybrid-XL d_attn=64 nhead=2 (32 dims/head)',
        },
        'total_min': round(total_s/60, 1),
        'models': {n: {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()}
                   for n, r in results.items()},
        'fold_details': details,
        'm2v_convergence': np.mean(m2v_tr, axis=0).round(4).tolist(),
        'xl_convergence': np.mean(xl_tr, axis=0).round(4).tolist(),
    }
    with open('trm_models_v8/summary_v8.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    log.info("✓ Saved: summary_v8.json")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v8_all", "zip", "trm_models_v8")
    log.info("✓ Created trm_v8_all.zip")
