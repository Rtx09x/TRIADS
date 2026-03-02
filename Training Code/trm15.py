"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V15 — HTRM: Hierarchical Tiny Reasoning Model          ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV          ║
║                                                                      ║
║  Architecture: HRM-inspired hierarchical reasoning                   ║
║                                                                      ║
║  H-MODULE (Attention Transformer — slow, abstract planning):         ║
║    • Tokenized features: Magpie(22) + DEML(13) + Alloy(23)         ║
║    • + 1 "state token" projecting L-module's current y-state        ║
║    • Shared-weight Self-Attention (re-runs every H-cycle)           ║
║    • Cross-Attention to Mat2Vec context                              ║
║    • Sees and STEERS the L-module's evolving predictions            ║
║                                                                      ║
║  L-MODULE (MLP-TRM loop — fast, detailed computation):               ║
║    • Shared-weight MLP reasoning (z_up, y_up)                       ║
║    • Conditioned on H-module's output zH                            ║
║    • 5 fast steps per H-cycle                                        ║
║                                                                      ║
║  Structure: 4 H-cycles × 5 L-steps = 20 total steps                ║
║  Gradient detach between H-cycles (HRM one-step approximation)      ║
║  Deep supervision at every L-step (all 20 steps)                    ║
║  Single seed (42) — experimental run                                 ║
║  Baseline: V13A ensemble = 91.20 MPa                                 ║
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
log = logging.getLogger("TRM15")

SEED = 42

BASELINES = {
    'TPOT-Mat':       79.9468,
    'AutoML-Mat':     82.3043,
    'MODNet':         87.7627,
    'V13A (ens)':     91.2028,
    'V14A-Flat':      94.9386,
    'V12A':           95.9900,
    'V14B-Token':     96.1497,
    'V11B':          102.3003,
    'RF-SCM/Magpie': 103.5125,
    'CrabNet':       107.3160,
    'Darwin':        123.2932,
}


# ══════════════════════════════════════════════════════════════════════
# 1. MEGA-FEATURIZER (same as V14)
# ══════════════════════════════════════════════════════════════════════

class MegaFeaturizer:
    """Mega feature set with tracked block sizes for tokenization."""
    GCS = "https://storage.googleapis.com/mat2vec/"
    FILES = ["pretrained_embeddings",
             "pretrained_embeddings.wv.vectors.npy",
             "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache="mat2vec_cache"):
        from matminer.featurizers.composition import (
            ElementFraction, Stoichiometry, ValenceOrbital,
            IonProperty, BandCenter
        )
        from matminer.featurizers.composition.alloy import (
            WenAlloys, Miedema, YangSolidSolution
        )
        from matminer.featurizers.composition.element import TMetalFraction
        from matminer.featurizers.composition.composite import Meredig
        from matminer.featurizers.base import MultipleFeaturizer

        self.ep_magpie = ElementProperty.from_preset("magpie")
        self.n_mg = len(self.ep_magpie.feature_labels())
        self.mg_n_props = len(self.ep_magpie.features)
        self.mg_stat_dim = len(self.ep_magpie.stats)

        self.ep_deml = ElementProperty.from_preset("deml")
        self.n_deml = len(self.ep_deml.feature_labels())
        self.deml_n_props = len(self.ep_deml.features)
        self.deml_stat_dim = len(self.ep_deml.stats)

        self.alloy_feats = MultipleFeaturizer([
            WenAlloys(), Miedema(), YangSolidSolution(), TMetalFraction()])
        self.n_alloy = None

        self.other_feats = MultipleFeaturizer([
            Meredig(), ElementFraction(), Stoichiometry(),
            ValenceOrbital(), IonProperty(), BandCenter()])
        self.n_other = None

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
        for c in tqdm(comps, desc="  Featurizing (mega)", leave=False):
            try: mg = np.array(self.ep_magpie.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)
            try: deml = np.array(self.ep_deml.featurize(c), np.float32)
            except: deml = np.zeros(self.n_deml, np.float32)
            try: alloy = np.array(self.alloy_feats.featurize(c), np.float32)
            except: alloy = np.zeros(self.n_alloy or 23, np.float32)
            if self.n_alloy is None: self.n_alloy = len(alloy)
            try: other = np.array(self.other_feats.featurize(c), np.float32)
            except: other = np.zeros(self.n_other or 241, np.float32)
            if self.n_other is None: self.n_other = len(other)

            out.append(np.concatenate([
                np.nan_to_num(mg, nan=0.0), np.nan_to_num(deml, nan=0.0),
                np.nan_to_num(alloy, nan=0.0), np.nan_to_num(other, nan=0.0),
                self._pool(c)]))

        total_d = self.n_mg + self.n_deml + self.n_alloy + self.n_other + 200
        log.info(f"MEGA features: Magpie={self.n_mg} + DEML={self.n_deml} + "
                 f"Alloy={self.n_alloy} + Other={self.n_other} + M2V=200 = {total_d}d")
        return np.array(out)

    def fit_scaler(self, X): self.scaler = StandardScaler().fit(X)
    def transform(self, X):
        if not self.scaler: return X
        return np.nan_to_num(self.scaler.transform(X), nan=0.0).astype(np.float32)


class DSData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y, np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════
# 2. HTRM — Hierarchical Tiny Reasoning Model
# ══════════════════════════════════════════════════════════════════════

class HTRM(nn.Module):
    """Hierarchical Tiny Reasoning Model (HRM-inspired).

    H-MODULE (Attention Transformer — slow, abstract planner):
      - Tokenizes features: Magpie(22×6) + DEML(13×5) + Alloy(23×1)
      - Adds a "state token" from L-module's current y-state
      - Shared-weight Self-Attention over all 59 tokens
      - Cross-Attention to Mat2Vec context (1 token)
      - Mean-pool + flat extras → zH (guidance vector for L-module)
      - Re-runs every H-cycle, seeing updated L-state each time

    L-MODULE (MLP-TRM — fast, detailed computation):
      - Shared-weight MLP reasoning loop (z_up, y_up)
      - Conditioned on zH from H-module
      - Runs n_l_steps per H-cycle
      - Deep supervision loss at every step

    Gradient detach between H-cycles (HRM one-step approximation).
    Total steps = n_h_cycles × n_l_steps.
    """
    def __init__(self, mg_n_props=22, mg_stat_dim=6,
                 deml_n_props=13, deml_stat_dim=5,
                 n_alloy=23, n_other=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, n_h_cycles=4, n_l_steps=5, **kw):
        super().__init__()
        self.D = d_hidden
        self.n_h_cycles = n_h_cycles
        self.n_l_steps = n_l_steps

        # Store sizes for input splitting
        self.mg_n_props, self.mg_stat_dim = mg_n_props, mg_stat_dim
        self.mg_dim = mg_n_props * mg_stat_dim
        self.deml_n_props, self.deml_stat_dim = deml_n_props, deml_stat_dim
        self.deml_dim = deml_n_props * deml_stat_dim
        self.n_alloy = n_alloy
        self.n_other = n_other

        # ═══════════════════════════════════════════════════════════════
        # H-MODULE: Attention Transformer (slow, abstract)
        # ═══════════════════════════════════════════════════════════════

        # Token projections
        self.mg_proj = nn.Sequential(
            nn.Linear(mg_stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.deml_proj = nn.Sequential(
            nn.Linear(deml_stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.alloy_proj = nn.Sequential(
            nn.Linear(1, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # State token: project L-module's y-state to attention space
        self.state_proj = nn.Sequential(
            nn.Linear(d_hidden, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # Shared-weight Self-Attention (re-used every H-cycle)
        self.sa = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa_n = nn.LayerNorm(d_attn)
        self.sa_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa_fn = nn.LayerNorm(d_attn)

        # Cross-Attention to Mat2Vec chemistry context
        self.ca = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d_attn)

        # Pool: attention output + flat extras → d_hidden
        pool_in = d_attn + (n_other if n_other > 0 else 0)
        self.h_pool = nn.Sequential(
            nn.Linear(pool_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())

        # ═══════════════════════════════════════════════════════════════
        # L-MODULE: MLP-TRM loop (fast, detailed)
        # ═══════════════════════════════════════════════════════════════

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

    def _h_module(self, x, y_state):
        """H-MODULE: Run attention over tokenized features + current L-state.

        Returns zH (d_hidden) — the high-level guidance vector.
        """
        B = x.size(0)

        # ── Split input into blocks ───────────────────────────────────
        idx = 0
        mg_flat = x[:, idx:idx+self.mg_dim];        idx += self.mg_dim
        deml_flat = x[:, idx:idx+self.deml_dim];     idx += self.deml_dim
        alloy_flat = x[:, idx:idx+self.n_alloy];     idx += self.n_alloy
        other_flat = x[:, idx:idx+self.n_other];     idx += self.n_other
        m2v_flat = x[:, idx:]                         # Mat2Vec (200d)

        # ── Create tokens ─────────────────────────────────────────────
        mg_tok = self.mg_proj(mg_flat.view(B, self.mg_n_props, self.mg_stat_dim))
        deml_tok = self.deml_proj(deml_flat.view(B, self.deml_n_props, self.deml_stat_dim))
        alloy_tok = self.alloy_proj(alloy_flat.unsqueeze(-1))

        # State token: L-module's current prediction state
        state_tok = self.state_proj(y_state).unsqueeze(1)   # [B, 1, d_attn]

        # All tokens: [B, 59, d_attn] (22 + 13 + 23 + 1 state)
        tok = torch.cat([mg_tok, deml_tok, alloy_tok, state_tok], dim=1)

        # Mat2Vec context: [B, 1, d_attn]
        ctx = self.m2v_proj(m2v_flat).unsqueeze(1)

        # ── Self-Attention (shared weights) ───────────────────────────
        tok = self.sa_n(tok + self.sa(tok, tok, tok)[0])
        tok = self.sa_fn(tok + self.sa_ff(tok))

        # ── Cross-Attention to Mat2Vec ────────────────────────────────
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx)[0])

        # ── Pool → zH ────────────────────────────────────────────────
        pooled = tok.mean(dim=1)   # [B, d_attn]

        if self.n_other > 0:
            pooled = torch.cat([pooled, other_flat], dim=-1)

        return self.h_pool(pooled)  # [B, d_hidden] = zH

    def _l_step(self, zH, y, z):
        """L-MODULE: One step of MLP-TRM reasoning, conditioned on zH."""
        z = z + self.z_up(torch.cat([zH, y, z], dim=-1))
        y = y + self.y_up(torch.cat([y, z], dim=-1))
        pred = self.head(y).squeeze(1)
        return y, z, pred

    def forward(self, x, deep_supervision=False):
        B = x.size(0)
        device = x.device

        y = torch.zeros(B, self.D, device=device)
        z = torch.zeros(B, self.D, device=device)
        step_preds = []

        for h in range(self.n_h_cycles):
            # ── Gradient detach between H-cycles (HRM one-step approx) ─
            if h > 0:
                y = y.detach()
                z = z.detach()

            # ── H-MODULE: Attention sees features + current L-state ────
            zH = self._h_module(x, y)

            # ── L-MODULE: MLP-TRM loop × n_l_steps ────────────────────
            for t in range(self.n_l_steps):
                y, z, pred = self._l_step(zH, y, z)
                step_preds.append(pred)

        return step_preds if deep_supervision else step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# 3. LOSS + UTILS + TRAINING
# ══════════════════════════════════════════════════════════════════════

def deep_supervision_loss(step_preds, targets):
    """Linear-weighted L1 loss across all steps."""
    n = len(step_preds)
    weights = [(i + 1) for i in range(n)]
    total_w = sum(weights)
    loss = 0.0
    for pred, w in zip(step_preds, weights):
        loss += (w / total_w) * F.l1_loss(pred, targets)
    return loss


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
    """Training with standard deep supervision + SWA."""
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
# 4. MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark():
    t0 = time.time()
    print("\n" + "═"*72)
    print("  TRM-MatSci V15 │ HTRM: Hierarchical Tiny Reasoning Model")
    print("  H-MODULE: Shared-weight Attention Transformer (59 tokens)")
    print("  L-MODULE: MLP-TRM reasoning (conditioned on H-state)")
    print("  Structure: 4 H-cycles × 5 L-steps = 20 total steps")
    print(f"  Seed: {SEED} │ Standard DS │ Experimental run")
    print("═"*72 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log.info("Loading matbench_steels...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_steels")
    comps_raw = df['composition'].tolist()
    targets_all = np.array(df['yield strength'].tolist(), np.float32)
    comps_all = [Composition(c) for c in comps_raw]

    # ── FEATURIZE ─────────────────────────────────────────────────────
    log.info("Computing MEGA features...")
    feat = MegaFeaturizer()
    X_all = feat.featurize_all(comps_all)
    log.info(f"Features: {X_all.shape}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    os.makedirs('trm_models_v15', exist_ok=True)
    dl_kw = dict(batch_size=32, num_workers=0)

    # ── MODEL CONFIG ──────────────────────────────────────────────────
    model_kw = dict(
        mg_n_props=feat.mg_n_props, mg_stat_dim=feat.mg_stat_dim,
        deml_n_props=feat.deml_n_props, deml_stat_dim=feat.deml_stat_dim,
        n_alloy=feat.n_alloy, n_other=feat.n_other, mat2vec_dim=200,
        d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
        dropout=0.2, n_h_cycles=4, n_l_steps=5)

    _m = HTRM(**model_kw)
    n_params = _m.count_parameters(); del _m
    n_tok = feat.mg_n_props + feat.deml_n_props + feat.n_alloy + 1  # +1 state
    print(f"\n  HTRM: {n_params:,} params")
    print(f"  Tokens: {n_tok} (Magpie={feat.mg_n_props} + DEML={feat.deml_n_props}"
          f" + Alloy={feat.n_alloy} + 1 state)")
    print(f"  Structure: {model_kw['n_h_cycles']} H-cycles × "
          f"{model_kw['n_l_steps']} L-steps = "
          f"{model_kw['n_h_cycles'] * model_kw['n_l_steps']} total steps")
    print(f"  Flat extras: {feat.n_other} features (after pool)")
    print()

    # ── TRAIN + EVALUATE ──────────────────────────────────────────────
    cname = 'V15-HTRM'
    fold_maes = []
    fold_hists = []

    for fi, (tv_i, te_i) in enumerate(folds):
        print(f"\n  ── [{cname}] Fold {fi+1}/5 {'─'*40}")

        tri, vli = strat_split(targets_all[tv_i], 0.15, SEED+fi)
        feat.fit_scaler(X_all[tv_i][tri])
        tr_s = feat.transform(X_all[tv_i][tri])
        vl_s = feat.transform(X_all[tv_i][vli])
        te_s = feat.transform(X_all[te_i])

        pin = device.type == 'cuda'
        tr_dl = DataLoader(DSData(tr_s, targets_all[tv_i][tri]), shuffle=True,
                           pin_memory=pin, **dl_kw)
        vl_dl = DataLoader(DSData(vl_s, targets_all[tv_i][vli]), shuffle=False,
                           pin_memory=pin, **dl_kw)
        te_dl = DataLoader(DSData(te_s, targets_all[te_i]), shuffle=False,
                           pin_memory=pin, **dl_kw)
        te_tgt = get_targets(te_dl)

        torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
        if device.type == 'cuda': torch.cuda.manual_seed(SEED + fi)

        model = HTRM(**model_kw).to(device)
        bv, model, hist = train_fold(model, tr_dl, vl_dl, device,
                                      fold=fi+1, name=cname)
        fold_hists.append(hist)

        pred = predict(model, te_dl, device)
        mae = F.l1_loss(pred, te_tgt).item()
        fold_maes.append(mae)
        log.info(f"  [{cname}] F{fi+1}: MAE={mae:.2f}  (val {bv:.2f})")

        torch.save({'model_state': model.state_dict(), 'test_mae': mae,
                    'config': cname},
                   f'trm_models_v15/{cname}_fold{fi+1}.pt')

        del model; torch.cuda.empty_cache() if device.type == 'cuda' else None

    avg = float(np.mean(fold_maes))
    std = float(np.std(fold_maes))
    print(f"\n  ═══ {cname}: {avg:.4f} ±{std:.4f} MPa ═══")

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════
    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V15 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'Params':>10} {'MAE(MPa)':>10} {'±Std':>8}")
    print(f"  {'─'*54}")
    tag = ("  ← BEATS MODNet 🏆" if avg < 87.76 else
           "  ← BEATS V13A ✓"  if avg < 91.20 else
           "  ← BEATS V14A ✓"  if avg < 94.94 else
           "  ← BEATS V12A ✓"  if avg < 95.99 else "")
    print(f"  {cname:<22} {n_params:>9,} {avg:>10.4f} {std:>8.4f}{tag}")
    print(f"  {'─'*54}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<22} {'baseline':>10} {bv:>10.4f}")
    print(f"\n  Total time: {tt/60:.1f} minutes")

    # Per-fold breakdown
    print(f"\n{'═'*72}")
    print(f"  PER-FOLD BREAKDOWN")
    print(f"{'═'*72}")
    print(f"  {'Fold':<6} {'V15-HTRM':>12} {'V14A-Flat':>12} {'V14B-Token':>12}")
    print(f"  {'─'*46}")
    v14a_folds = [122.25, 82.77, 85.37, 94.27, 90.04]
    v14b_folds = [113.30, 86.51, 81.56, 107.18, 92.20]
    for fi in range(5):
        print(f"  {fi+1:<6} {fold_maes[fi]:>12.2f} {v14a_folds[fi]:>12.2f} "
              f"{v14b_folds[fi]:>12.2f}")
    print()

    results = {'avg': avg, 'std': std, 'folds': fold_maes, 'params': n_params}
    generate_plots(results, fold_hists, avg, std, fold_maes,
                   v14a_folds, v14b_folds)
    save_summary(results, tt, model_kw)
    return results


# ══════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ══════════════════════════════════════════════════════════════════════

COL_V15 = '#D32F2F'
COL_V14A = '#1565C0'
COL_V14B = '#E65100'

def generate_plots(results, fold_hists, avg, std, fold_maes,
                   v14a_folds, v14b_folds):
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ── Plot 1: Bar vs baselines ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['V15-HTRM', 'V14A-Flat', 'V14B-Token']
    vals = [avg, 94.94, 96.15]
    cols = [COL_V15, COL_V14A, COL_V14B]
    bars = ax1.bar(models, vals, color=cols, alpha=0.88,
                   edgecolor='white', linewidth=1.5)
    for bv, c, ls, lb in [
        (79.95, '#2E7D32', '--', 'TPOT-Mat (79.95)'),
        (87.76, '#F57F17', '--', 'MODNet (87.76)'),
        (91.20, '#9C27B0', '-.', 'V13A ens (91.20)'),
        (95.99, '#4CAF50', '-.', 'V12A (95.99)'),
    ]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    for bar, m in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{m:.1f}', ha='center', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_ylabel('MAE (MPa)'); ax1.set_ylim(0, max(vals)*1.5)
    ax1.set_title('V15 HTRM vs Previous (single seed)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # ── Plot 2: Per-fold comparison ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(1, 6)
    w = 0.25
    ax2.bar(x - w, fold_maes, w, color=COL_V15, alpha=0.85, label='V15-HTRM')
    ax2.bar(x, v14a_folds, w, color=COL_V14A, alpha=0.85, label='V14A-Flat')
    ax2.bar(x + w, v14b_folds, w, color=COL_V14B, alpha=0.85, label='V14B-Token')
    ax2.axhline(91.20, color='#9C27B0', ls='-.', lw=1.5, label='V13A ens (91.20)')
    ax2.axhline(87.76, color='#F57F17', ls='--', lw=1.5, label='MODNet (87.76)')
    ax2.set_xlabel('Fold'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_xticks(x); ax2.set_xticklabels([f'F{i}' for i in range(1,6)])
    ax2.set_title('Per-Fold Comparison', fontweight='bold')
    ax2.legend(fontsize=7); ax2.grid(axis='y', alpha=0.2)

    # ── Plot 3: Training curves ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for fi, h in enumerate(fold_hists):
        lb_tr = 'train' if fi == 0 else None
        lb_vl = 'val'   if fi == 0 else None
        ax3.plot(h['train'], alpha=0.3, lw=0.8, color=COL_V15, label=lb_tr)
        ax3.plot(h['val'],   alpha=0.7, lw=1.2, color=COL_V15, label=lb_vl,
                 linestyle='--')
    ax3.axhline(91.20, color='#9C27B0', ls='-.', lw=1.2, label='V13A ens (91.20)')
    ax3.axvline(200, color='#4CAF50', ls='--', lw=1.2, alpha=0.6, label='SWA start')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE (MPa)')
    ax3.set_title('HTRM Training Curves (all folds)', fontweight='bold')
    ax3.legend(fontsize=7, ncol=2); ax3.grid(alpha=0.2)
    ax3.set_ylim(0, 300)

    # ── Plot 4: Architecture diagram ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10); ax4.set_ylim(0, 10)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('HTRM Architecture', fontweight='bold', fontsize=13)

    # Draw H-cycles
    for i, yy in enumerate([8.0, 5.5, 3.0, 0.5]):
        # H-module box
        rect = plt.Rectangle((0.5, yy), 3.5, 1.5, fill=True,
                             facecolor='#BBDEFB', edgecolor=COL_V14A, lw=2)
        ax4.add_patch(rect)
        ax4.text(2.25, yy+0.75, f'H-Module\n(Attention)', ha='center',
                va='center', fontsize=8, fontweight='bold', color=COL_V14A)

        # Arrow
        ax4.annotate('', xy=(5.0, yy+0.75), xytext=(4.2, yy+0.75),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax4.text(4.6, yy+1.0, 'zH', fontsize=7, ha='center')

        # L-module box
        rect2 = plt.Rectangle((5.2, yy), 4.0, 1.5, fill=True,
                              facecolor='#FFCCBC', edgecolor=COL_V15, lw=2)
        ax4.add_patch(rect2)
        ax4.text(7.2, yy+0.75, f'L-Module ×5\n(MLP-TRM)', ha='center',
                va='center', fontsize=8, fontweight='bold', color=COL_V15)

        # Cycle label
        ax4.text(0.2, yy+0.75, f'C{i+1}', fontsize=9, fontweight='bold',
                ha='center', va='center')

        # Detach arrow between cycles
        if i < 3:
            ax4.annotate('', xy=(7.2, yy), xytext=(7.2, yy-0.8),
                        arrowprops=dict(arrowstyle='->', color='gray',
                                       lw=1.5, ls='--'))
            ax4.text(8.0, yy-0.4, 'detach', fontsize=6, color='gray')

            # Feedback arrow (y_state → H-module)
            ax4.annotate('', xy=(2.25, yy), xytext=(2.25, yy-0.8),
                        arrowprops=dict(arrowstyle='->', color=COL_V15,
                                       lw=1.2, ls='-'))

    fig.suptitle('TRM-MatSci V15 │ HTRM │ matbench_steels',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v15.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: trm_results_v15.png")


def save_summary(results, total_s, model_kw):
    s = {
        'version': 'V15', 'task': 'matbench_steels',
        'architecture': 'HTRM (Hierarchical Tiny Reasoning Model)',
        'strategy': 'H-module(Attention) + L-module(MLP-TRM), HRM-inspired',
        'seed': SEED,
        'n_h_cycles': model_kw['n_h_cycles'],
        'n_l_steps': model_kw['n_l_steps'],
        'total_steps': model_kw['n_h_cycles'] * model_kw['n_l_steps'],
        'total_min': round(total_s/60, 1),
        'avg': round(results['avg'], 4),
        'std': round(results['std'], 4),
        'folds': [round(x, 4) for x in results['folds']],
        'params': results['params'],
    }
    with open('trm_models_v15/summary_v15.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    log.info("✓ Saved: summary_v15.json")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v15_all", "zip", "trm_models_v15")
    log.info("✓ Created trm_v15_all.zip")
