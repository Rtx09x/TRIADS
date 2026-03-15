"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V11 — The Push for #1                                    ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV          ║
║                                                                      ║
║  Three models, one goal: beat MODNet (87.76 MPa)                    ║
║                                                                      ║
║  V11A  Feature Expansion  — more chemical descriptors               ║
║        + Magpie + Mat2Vec + ElementFraction + Stoichiometry          ║
║        + ValenceOrbital + IonProperty + BandCenter                   ║
║        Same V10A arch (d_attn=48, 20 steps, Deep Supervision)       ║
║                                                                      ║
║  V11B  Scaled + Deep Supervision — bigger model now safe            ║
║        d_attn=64, nhead=4, d_hidden=96, ff_dim=150, 20 steps       ║
║        Deep Supervision regularizes away V8's overfitting            ║
║                                                                      ║
║  V11C  Learned Adaptive Halting (ACT-style)                          ║
║        Model learns halt probability per step per sample             ║
║        min=12, max=22 steps, ponder cost penalty                     ║
║        Same base arch as V10A + halt_head                            ║
║                                                                      ║
║  All models: Deep Supervision + SWA + AdamW + 300 epochs            ║
║  Baseline: V10A = 103.28 MPa (current best)                         ║
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
log = logging.getLogger("TRM11")
SEED = 42

BASELINES = {
    'TPOT-Mat':       79.9468,
    'AutoML-Mat':     82.3043,
    'MODNet':         87.7627,
    'RF-SCM/Magpie': 103.5125,
    'V10A (best)':   103.2867,
    'CrabNet':       107.3160,
    'Darwin':        123.2932,
}


# ══════════════════════════════════════════════════════════════════════
# 1. FEATURIZERS
# ══════════════════════════════════════════════════════════════════════

class CombinedFeaturizer:
    """Original V10 featurizer: 22 Magpie props × 6 stats + 200 Mat2Vec."""
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
        for c in tqdm(comps, desc="  Featurizing (standard)", leave=False):
            try: mg = np.array(self.ep.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)
            out.append(np.concatenate([np.nan_to_num(mg, nan=0.0), self._pool(c)]))
        return np.array(out)

    def fit_scaler(self, X): self.scaler = StandardScaler().fit(X)
    def transform(self, X):
        if not self.scaler: return X
        return np.nan_to_num(self.scaler.transform(X), nan=0.0).astype(np.float32)


class ExpandedFeaturizer:
    """V11A featurizer: Magpie + additional matminer descriptors + Mat2Vec.

    Adds ElementFraction, Stoichiometry, ValenceOrbital, IonProperty, BandCenter
    to the existing Magpie stats. All extra features are flat (not property × stats),
    so they are concatenated AFTER the Magpie block.

    Layout: [Magpie: n_props × stat_dim] [Extra: n_extra flat] [Mat2Vec: 200]
    """
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
            ElementFraction(),
            Stoichiometry(),
            ValenceOrbital(),
            IonProperty(),
            BandCenter(),
        ])
        self.n_extra = None  # Will be detected at featurize time

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
        for c in tqdm(comps, desc="  Featurizing (expanded)", leave=False):
            # Magpie block (n_props × stat_dim structure)
            try: mg = np.array(self.ep_magpie.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)

            # Extra features (flat vector)
            try:
                ex = np.array(self.extra_feats.featurize(c), np.float32)
            except:
                # If we already know n_extra, use it; else use placeholder
                ex = np.zeros(self.n_extra or 200, np.float32)

            if self.n_extra is None:
                self.n_extra = len(ex)
                log.info(f"Expanded features: {self.n_mg} Magpie + {self.n_extra} Extra + 200 Mat2Vec = {self.n_mg + self.n_extra + 200}d")

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


class DSData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y, np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════
# 2. MODELS
# ══════════════════════════════════════════════════════════════════════

class HybridTRM(nn.Module):
    """Standard Hybrid-TRM with Deep Supervision. Used by V11A and V11B."""
    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                 dropout=0.2, max_steps=20, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.n_extra = n_extra

        # Attention feature extractor
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

        # Pool + optional extra feature injection
        pool_in = d_attn + (n_extra if n_extra > 0 else 0)
        self.pool = nn.Sequential(
            nn.Linear(pool_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())

        # MLP-TRM recursive reasoning
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
        mg = x[:, :mg_dim]

        if self.n_extra > 0:
            extra = x[:, mg_dim:mg_dim + self.n_extra]
            m2v = x[:, mg_dim + self.n_extra:]
        else:
            extra = None
            m2v = x[:, mg_dim:]

        tok = self.tok_proj(mg.view(B, self.n_props, self.stat_dim))
        ctx = self.m2v_proj(m2v).unsqueeze(1)

        tok = self.sa_n(tok + self.sa(tok, tok, tok)[0])
        tok = self.sa_fn(tok + self.sa_ff(tok))
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx)[0])

        pooled = tok.mean(dim=1)  # [B, d_attn]

        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=-1)  # [B, d_attn + n_extra]

        return self.pool(pooled)  # [B, d_hidden]

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


class ACTHybridTRM(nn.Module):
    """V11C: Hybrid-TRM with Adaptive Computation Time (ACT).

    Instead of fixed steps or hand-tuned thresholds, the model LEARNS
    when to stop thinking via a halt_head that outputs a halt probability.

    Training: All steps run (for deep supervision), but predictions are
    weighted by learned halting probabilities. A ponder cost encourages
    efficiency (fewer steps = lower penalty).

    Inference: Sample halts when cumulative halt probability >= 1.0.
    """
    def __init__(self, n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                 dropout=0.2, max_steps=22, min_steps=12,
                 ponder_cost=0.01, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.min_steps = min_steps
        self.ponder_cost = ponder_cost
        self.n_props, self.stat_dim = n_props, stat_dim

        # Attention feature extractor (identical to HybridTRM)
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

        # MLP-TRM recursive reasoning
        self.z_up = nn.Sequential(
            nn.Linear(d_hidden*3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.y_up = nn.Sequential(
            nn.Linear(d_hidden*2, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))

        self.head = nn.Linear(d_hidden, 1)

        # ── ACT: Learned halting ─────────────────────────────────────
        self.halt_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, 1), nn.Sigmoid())

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        # Initialize halt_head bias negative so model starts by NOT halting
        # (sigmoid(-2) ≈ 0.12, so early halt probs are low)
        nn.init.constant_(self.halt_head[-2].bias, -2.0)

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

    def forward(self, x, deep_supervision=False):
        """Training forward pass with ACT.

        Returns:
            If deep_supervision=True: (step_preds, halt_probs, remainders)
                step_preds: list of [B] predictions at each step
                halt_probs: [B, max_steps] tensor of halt probabilities
                remainders: [B] tensor - remainder for ponder cost
            Else: final weighted prediction [B]
        """
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)

        step_preds = []
        all_halts = []
        cum_halt = torch.zeros(B, device=x.device)
        remainders = torch.zeros(B, device=x.device)
        active = torch.ones(B, dtype=torch.bool, device=x.device)

        for step in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))

            pred = self.head(y).squeeze(1)
            step_preds.append(pred)

            # Halt probability (suppressed for min_steps)
            if step < self.min_steps:
                h = torch.zeros(B, device=x.device)
            else:
                h = self.halt_head(y).squeeze(1)  # [B], in [0, 1]

            all_halts.append(h)

            # Track cumulative halting (for the remainder penalty)
            still_active = (cum_halt + h < 1.0)
            # Samples that would exceed 1.0 this step get remainder
            newly_halted = active & ~still_active
            if newly_halted.any():
                remainders[newly_halted] = 1.0 - cum_halt[newly_halted]
            cum_halt = cum_halt + h
            active = active & still_active

        # Force-halt remaining samples at the last step
        remainders[active] = 1.0 - cum_halt[active]

        halt_probs = torch.stack(all_halts, dim=1)  # [B, max_steps]

        if deep_supervision:
            return step_preds, halt_probs, remainders
        else:
            # Weighted prediction: each step's prediction weighted by halt prob
            # Normalize halt_probs to sum to 1
            hp_norm = halt_probs / (halt_probs.sum(dim=1, keepdim=True) + 1e-8)
            preds_stack = torch.stack(step_preds, dim=1)  # [B, max_steps]
            weighted_pred = (preds_stack * hp_norm).sum(dim=1)
            return weighted_pred

    def inference_forward(self, x):
        """Inference: stop each sample when cumulative halt prob >= 1.0."""
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)

        final_pred = torch.zeros(B, device=x.device)
        halted = torch.zeros(B, dtype=torch.bool, device=x.device)
        halt_step = torch.full((B,), self.max_steps, dtype=torch.long,
                               device=x.device)
        cum_halt = torch.zeros(B, device=x.device)

        for step in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            pred = self.head(y).squeeze(1)

            if step < self.min_steps:
                h = torch.zeros(B, device=x.device)
            else:
                h = self.halt_head(y).squeeze(1)

            cum_halt = cum_halt + h
            newly_halted = (~halted) & (cum_halt >= 1.0)
            if newly_halted.any():
                final_pred[newly_halted] = pred[newly_halted]
                halt_step[newly_halted] = step + 1
                halted = halted | newly_halted

            if halted.all():
                break

        # Samples that never halted use final prediction
        still_running = ~halted
        if still_running.any():
            final_pred[still_running] = pred[still_running]

        return final_pred, halt_step

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def deep_supervision_loss(step_preds, targets):
    """Linear-weighted L1 loss across all recursion steps."""
    n = len(step_preds)
    weights = [(i + 1) for i in range(n)]
    total_w = sum(weights)
    loss = 0.0
    for pred, w in zip(step_preds, weights):
        loss += (w / total_w) * F.l1_loss(pred, targets)
    return loss


def act_loss(step_preds, targets, halt_probs, remainders, ponder_cost=0.01):
    """ACT loss = deep supervision + halt-weighted prediction loss + ponder cost.

    Components:
    1. Deep supervision: standard linear-weighted L1 across all steps
    2. Halt-weighted: L1 of the halt-probability-weighted prediction
    3. Ponder cost: penalizes the number of steps used (via remainders)
    """
    # Deep supervision on all steps
    ds_loss = deep_supervision_loss(step_preds, targets)

    # Halt-probability-weighted prediction
    hp_norm = halt_probs / (halt_probs.sum(dim=1, keepdim=True) + 1e-8)
    preds_stack = torch.stack(step_preds, dim=1)
    weighted_pred = (preds_stack * hp_norm).sum(dim=1)
    weighted_loss = F.l1_loss(weighted_pred, targets)

    # Ponder cost: encourage efficiency
    ponder = ponder_cost * remainders.mean()

    return ds_loss + weighted_loss + ponder


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


def train_fold_standard(model, tr_dl, vl_dl, device,
                        epochs=300, swa_start=200, fold=1, name=""):
    """Training with standard deep supervision (V11A, V11B)."""
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


def train_fold_act(model, tr_dl, vl_dl, device,
                   epochs=300, swa_start=200, fold=1, name="V11C"):
    """Training with ACT loss (V11C)."""
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
            step_preds, halt_probs, remainders = model(bx, deep_supervision=True)
            loss = act_loss(step_preds, by, halt_probs, remainders,
                           ponder_cost=model.ponder_cost)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += F.l1_loss(step_preds[-1], by).item() * len(by)
        tl /= len(tr_dl.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                bx, by = bx.to(device), by.to(device)
                pred, _ = model.inference_forward(bx)
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


def predict_act(model, dl, device):
    """Predict using ACT model's learned halting."""
    model.eval()
    all_preds, all_steps = [], []
    with torch.no_grad():
        for bx, _ in dl:
            pred, steps = model.inference_forward(bx.to(device))
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
    print("  TRM-MatSci V11 │ The Push for #1 │ matbench_steels")
    print("  V11A: Feature Expansion (more chemical descriptors)")
    print("  V11B: Scaled Architecture (d_attn=64) + Deep Supervision")
    print("  V11C: Learned Adaptive Halting (ACT-style, 12-22 steps)")
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

    # ── FEATURIZE ─────────────────────────────────────────────────────
    log.info("Computing STANDARD features (V11B, V11C)...")
    feat_std = CombinedFeaturizer()
    X_std = feat_std.featurize_all(comps_all)
    log.info(f"Standard features: {X_std.shape}")

    log.info("Computing EXPANDED features (V11A)...")
    feat_exp = ExpandedFeaturizer()
    X_exp = feat_exp.featurize_all(comps_all)
    n_extra = feat_exp.n_extra
    log.info(f"Expanded features: {X_exp.shape} (n_extra={n_extra})")

    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    os.makedirs('trm_models_v11', exist_ok=True)
    dl_kw = dict(batch_size=32, num_workers=0)

    # ── CONFIGS ───────────────────────────────────────────────────────
    configs = {
        'V11A-FeatExp': {
            'model_cls': HybridTRM,
            'model_kw': dict(n_props=22, stat_dim=6, n_extra=n_extra,
                             mat2vec_dim=200, d_attn=48, nhead=2,
                             d_hidden=64, ff_dim=100, dropout=0.2,
                             max_steps=20),
            'X': X_exp, 'feat': feat_exp, 'train_fn': train_fold_standard,
            'predict_fn': predict, 'is_act': False,
        },
        'V11B-Scaled': {
            'model_cls': HybridTRM,
            'model_kw': dict(n_props=22, stat_dim=6, n_extra=0,
                             mat2vec_dim=200, d_attn=64, nhead=4,
                             d_hidden=96, ff_dim=150, dropout=0.2,
                             max_steps=20),
            'X': X_std, 'feat': feat_std, 'train_fn': train_fold_standard,
            'predict_fn': predict, 'is_act': False,
        },
        'V11C-ACT': {
            'model_cls': ACTHybridTRM,
            'model_kw': dict(n_props=22, stat_dim=6, mat2vec_dim=200,
                             d_attn=48, nhead=2, d_hidden=64, ff_dim=100,
                             dropout=0.2, max_steps=22, min_steps=12,
                             ponder_cost=0.01),
            'X': X_std, 'feat': feat_std, 'train_fn': train_fold_act,
            'predict_fn': None, 'is_act': True,
        },
    }

    # Print param counts
    print(f"\n  {'Config':<20} {'Params':>10} {'Steps':>8}")
    print(f"  {'─'*42}")
    for cname, cfg in configs.items():
        _m = cfg['model_cls'](**cfg['model_kw'])
        np_ = _m.count_parameters(); del _m
        cfg['n_params'] = np_
        steps = cfg['model_kw'].get('max_steps', 20)
        print(f"  {cname:<20} {np_:>10,} {steps:>8}")
    print()

    # ── TRAIN + EVALUATE ──────────────────────────────────────────────
    all_results = {}
    all_halt_stats = {}

    for cname, cfg in configs.items():
        print(f"\n{'▓'*72}")
        print(f"  {cname}")
        print(f"{'▓'*72}")

        fold_maes = []
        fold_halt_info = []

        for fi, (tv_i, te_i) in enumerate(folds):
            print(f"\n  ── [{cname}] Fold {fi+1}/5 {'─'*40}")

            X = cfg['X']
            feat = cfg['feat']
            tri, vli = strat_split(targets_all[tv_i], 0.15, SEED+fi)
            feat.fit_scaler(X[tv_i][tri])
            tr_s = feat.transform(X[tv_i][tri])
            vl_s = feat.transform(X[tv_i][vli])
            te_s = feat.transform(X[te_i])

            pin = device.type == 'cuda'
            tr_dl = DataLoader(DSData(tr_s, targets_all[tv_i][tri]), shuffle=True,
                               pin_memory=pin, **dl_kw)
            vl_dl = DataLoader(DSData(vl_s, targets_all[tv_i][vli]), shuffle=False,
                               pin_memory=pin, **dl_kw)
            te_dl = DataLoader(DSData(te_s, targets_all[te_i]), shuffle=False,
                               pin_memory=pin, **dl_kw)
            te_tgt = get_targets(te_dl)

            torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
            model = cfg['model_cls'](**cfg['model_kw']).to(device)
            bv, model, hist = cfg['train_fn'](model, tr_dl, vl_dl, device,
                                               fold=fi+1, name=cname)

            # Predict
            if cfg['is_act']:
                pred, halt_steps = predict_act(model, te_dl, device)
                hs = halt_steps.float()
                halt_info = {
                    'mean': float(hs.mean()),
                    'min': int(hs.min()),
                    'max': int(hs.max()),
                    'pct_early': float((hs < cfg['model_kw']['max_steps']).float().mean() * 100),
                }
                fold_halt_info.append(halt_info)
                log.info(f"  [{cname}] F{fi+1}: MAE={F.l1_loss(pred, te_tgt).item():.2f}  "
                         f"(avg halt={halt_info['mean']:.1f}, "
                         f"{halt_info['pct_early']:.0f}% early)")
            else:
                pred = cfg['predict_fn'](model, te_dl, device)
                log.info(f"  [{cname}] F{fi+1}: MAE={F.l1_loss(pred, te_tgt).item():.2f}  "
                         f"(val {bv:.2f})")

            mae = F.l1_loss(pred, te_tgt).item()
            fold_maes.append(mae)

            torch.save({'model_state': model.state_dict(), 'test_mae': mae,
                        'config': cname},
                       f'trm_models_v11/{cname}_fold{fi+1}.pt')

        avg = float(np.mean(fold_maes))
        std = float(np.std(fold_maes))
        all_results[cname] = {
            'avg': avg, 'std': std, 'folds': fold_maes,
            'params': cfg['n_params'],
        }
        if fold_halt_info:
            all_halt_stats[cname] = fold_halt_info

        print(f"\n  ═══ {cname}: {avg:.4f} ±{std:.4f} MPa ═══")

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════

    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V11 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'Params':>10} {'MAE(MPa)':>10} {'±Std':>8}")
    print(f"  {'─'*52}")
    for n, r in sorted(all_results.items(), key=lambda x: x[1]['avg']):
        tag = ("  ← BEATS MODNet 🏆" if r['avg'] < 87.76 else
               "  ← BEATS RF-SCM ✓"  if r['avg'] < 103.51 else
               "  ← BEATS DARWIN ✓"  if r['avg'] < 123.29 else "")
        print(f"  {n:<22} {r['params']:>9,} "
              f"{r['avg']:>10.4f} {r['std']:>8.4f}{tag}")
    print(f"  {'─'*52}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<22} {'baseline':>10} {bv:>10.4f}")
    print(f"\n  Total time: {tt/60:.1f} minutes")

    # Per-fold breakdown
    print(f"\n{'═'*72}")
    print(f"  PER-FOLD BREAKDOWN")
    print(f"{'═'*72}")
    cnames = list(all_results.keys())
    header = f"  {'Fold':<6}"
    for cn in cnames:
        header += f" {cn:>12}"
    print(header)
    print(f"  {'─'*54}")
    for fi in range(5):
        row = f"  {fi+1:<6}"
        for cn in cnames:
            row += f" {all_results[cn]['folds'][fi]:>12.2f}"
        print(row)

    # ACT halt stats
    if all_halt_stats:
        print(f"\n  ACT Halting Summary (V11C):")
        for cn, stats in all_halt_stats.items():
            avg_halt = np.mean([h['mean'] for h in stats])
            avg_early = np.mean([h['pct_early'] for h in stats])
            print(f"    {cn}: avg halt step={avg_halt:.1f}, early halting={avg_early:.0f}%")
    print()

    generate_plots(all_results, all_halt_stats)
    save_summary(all_results, all_halt_stats, tt)
    return all_results


# ══════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════

PAL = {'V11A-FeatExp': '#1565C0', 'V11B-Scaled': '#E65100', 'V11C-ACT': '#2E7D32'}

def generate_plots(all_results, halt_stats):
    names = list(all_results.keys())
    avgs = [all_results[n]['avg'] for n in names]
    stds = [all_results[n]['std'] for n in names]
    cols = [PAL.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)

    # Bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(names, avgs, yerr=stds, capsize=6, color=cols,
                   alpha=0.88, edgecolor='white', linewidth=1.5)
    for bv, c, ls, lb in [
        (87.76, '#F57F17', '--', 'MODNet (87.76)'),
        (103.29, '#9E9E9E', '-.', 'V10A (103.29)'),
        (103.51, '#B0BEC5', ':', 'RF-SCM (103.51)'),
        (107.32, '#FF9800', ':', 'CrabNet (107.32)'),
        (123.29, '#FF5722', ':', 'Darwin (123.29)'),
    ]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    for bar, m, s in zip(bars, avgs, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+1,
                 f'{m:.1f}', ha='center', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_ylabel('MAE (MPa)'); ax1.set_ylim(0, max(avgs)*1.5)
    ax1.set_title('V11 Results vs Baselines', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Per-fold comparison
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(1, 6)
    w = 0.25
    for i, (n, col) in enumerate(zip(names, cols)):
        fold_vals = all_results[n]['folds']
        ax2.bar(x + (i - 1) * w, fold_vals, w, color=col, alpha=0.8,
                label=n, edgecolor='white')
    ax2.axhline(103.29, color='#9E9E9E', ls='-.', lw=1.5, label='V10A')
    ax2.axhline(103.51, color='#B0BEC5', ls=':', lw=1.5, label='RF-SCM')
    ax2.axhline(123.29, color='#FF5722', ls=':', lw=1.5, label='Darwin')
    ax2.set_xlabel('Fold'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_xticks(x); ax2.set_xticklabels([f'F{i}' for i in range(1,6)])
    ax2.set_title('Per-Fold Breakdown', fontweight='bold')
    ax2.legend(fontsize=7); ax2.grid(axis='y', alpha=0.2)

    fig.suptitle('TRM-MatSci V11 │ Push for #1 │ matbench_steels',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v11.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: trm_results_v11.png")


def save_summary(all_results, halt_stats, total_s):
    s = {
        'version': 'V11', 'task': 'matbench_steels',
        'strategy': 'Feature Expansion + Scaling + ACT Halting',
        'total_min': round(total_s/60, 1),
        'models': {n: {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()}
                   for n, r in all_results.items()},
        'halt_stats': halt_stats,
    }
    with open('trm_models_v11/summary_v11.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    log.info("✓ Saved: summary_v11.json")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v11_all", "zip", "trm_models_v11")
    log.info("✓ Created trm_v11_all.zip")
