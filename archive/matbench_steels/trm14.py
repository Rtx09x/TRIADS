"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V14 — Mega Features + Tokenized Attention               ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV          ║
║                                                                      ║
║  V14A  Flat features + 2-Layer SA + Standard DS (control)            ║
║        Same architecture as V13A but with mega-expanded features     ║
║        All extra features concatenated flat after attention          ║
║                                                                      ║
║  V14B  TOKENIZED features + Shared-Weight SA (TRM attention loop)    ║
║        Magpie → 22 tokens × 6d → project to 64d                    ║
║        DEML → 13 tokens × 5d → project to 64d                      ║
║        Alloy (Wen+Miedema+Yang+TMetal) → 23 tokens × 1d → proj 64d ║
║        All tokens attend to each other via SHARED-weight SA ×2       ║
║        (= 2-step TRM loop in the attention feature extractor)       ║
║        Cross-attention to Mat2Vec context                            ║
║        Remaining features as flat extra after pooling                ║
║                                                                      ║
║  Both: Standard DS + SWA + AdamW + 300 epochs                       ║
║  Single seed (42) — experimental comparison run                      ║
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
log = logging.getLogger("TRM14")

SEED = 42

BASELINES = {
    'TPOT-Mat':       79.9468,
    'AutoML-Mat':     82.3043,
    'MODNet':         87.7627,
    'RF-SCM/Magpie': 103.5125,
    'V13A (ens)':     91.2028,
    'V12A':           95.9900,
    'V11B':          102.3003,
    'CrabNet':       107.3160,
    'Darwin':        123.2932,
}


# ══════════════════════════════════════════════════════════════════════
# 1. MEGA-FEATURIZER — with block-level size tracking
# ══════════════════════════════════════════════════════════════════════

class MegaFeaturizer:
    """Massively expanded features with tracked block sizes.

    Output layout (flat concat):
      [Magpie | DEML | Alloy | Other | Mat2Vec]
       132      65     ~23    ~241    200

    Block sizes stored as attributes for V14B's tokenization.
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
        from matminer.featurizers.composition.alloy import (
            WenAlloys, Miedema, YangSolidSolution
        )
        from matminer.featurizers.composition.element import TMetalFraction
        from matminer.featurizers.composition.composite import Meredig
        from matminer.featurizers.base import MultipleFeaturizer

        # ── Block 1: Magpie (22 props × 6 stats) ─────────────────────
        self.ep_magpie = ElementProperty.from_preset("magpie")
        self.n_mg = len(self.ep_magpie.feature_labels())
        self.mg_n_props = len(self.ep_magpie.features)    # 22
        self.mg_stat_dim = len(self.ep_magpie.stats)      # 6

        # ── Block 2: DEML (13 props × 5 stats) ───────────────────────
        self.ep_deml = ElementProperty.from_preset("deml")
        self.n_deml = len(self.ep_deml.feature_labels())
        self.deml_n_props = len(self.ep_deml.features)    # 13
        self.deml_stat_dim = len(self.ep_deml.stats)      # 5

        # ── Block 3: Alloy features (tokenizable scalars) ────────────
        self.alloy_feats = MultipleFeaturizer([
            WenAlloys(),
            Miedema(),
            YangSolidSolution(),
            TMetalFraction(),
        ])
        self.n_alloy = None  # detected at featurize time (~23)

        # ── Block 4: Other extras (flat) ──────────────────────────────
        self.other_feats = MultipleFeaturizer([
            Meredig(),
            ElementFraction(),
            Stoichiometry(),
            ValenceOrbital(),
            IonProperty(),
            BandCenter(),
        ])
        self.n_other = None  # detected at featurize time (~241)

        # ── Block 5: Mat2Vec ──────────────────────────────────────────
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
            # Block 1: Magpie
            try: mg = np.array(self.ep_magpie.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)

            # Block 2: DEML
            try: deml = np.array(self.ep_deml.featurize(c), np.float32)
            except: deml = np.zeros(self.n_deml, np.float32)

            # Block 3: Alloy
            try:
                alloy = np.array(self.alloy_feats.featurize(c), np.float32)
            except:
                alloy = np.zeros(self.n_alloy or 23, np.float32)

            if self.n_alloy is None:
                self.n_alloy = len(alloy)

            # Block 4: Other
            try:
                other = np.array(self.other_feats.featurize(c), np.float32)
            except:
                other = np.zeros(self.n_other or 241, np.float32)

            if self.n_other is None:
                self.n_other = len(other)

            out.append(np.concatenate([
                np.nan_to_num(mg, nan=0.0),
                np.nan_to_num(deml, nan=0.0),
                np.nan_to_num(alloy, nan=0.0),
                np.nan_to_num(other, nan=0.0),
                self._pool(c)
            ]))

        total_d = self.n_mg + self.n_deml + self.n_alloy + self.n_other + 200
        log.info(f"MEGA features: "
                 f"Magpie={self.n_mg} ({self.mg_n_props}×{self.mg_stat_dim}) + "
                 f"DEML={self.n_deml} ({self.deml_n_props}×{self.deml_stat_dim}) + "
                 f"Alloy={self.n_alloy} + Other={self.n_other} + "
                 f"Mat2Vec=200 = {total_d}d")

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
# 2A. MODEL — V14A: Flat features + 2-Layer SA (same as V13A)
# ══════════════════════════════════════════════════════════════════════

class DeepHybridTRM(nn.Module):
    """V14A: Flat-feature 2-Layer SA (control).

    Same as V13A — only Magpie tokens attend. DEML/Alloy/Other are flat
    extras concatenated after pooling. Separate weights per SA layer.
    """
    def __init__(self, mg_n_props=22, mg_stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, max_steps=20, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.mg_n_props, self.mg_stat_dim = mg_n_props, mg_stat_dim
        self.n_extra = n_extra

        # Projections
        self.tok_proj = nn.Sequential(
            nn.Linear(mg_stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # SA Layer 1
        self.sa1 = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa1_n = nn.LayerNorm(d_attn)
        self.sa1_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa1_fn = nn.LayerNorm(d_attn)

        # SA Layer 2
        self.sa2 = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa2_n = nn.LayerNorm(d_attn)
        self.sa2_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa2_fn = nn.LayerNorm(d_attn)

        # Cross-Attention
        self.ca = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d_attn)

        # Pool
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
        mg_dim = self.mg_n_props * self.mg_stat_dim
        mg = x[:, :mg_dim]

        if self.n_extra > 0:
            extra = x[:, mg_dim:mg_dim + self.n_extra]
            m2v = x[:, mg_dim + self.n_extra:]
        else:
            extra = None
            m2v = x[:, mg_dim:]

        tok = self.tok_proj(mg.view(B, self.mg_n_props, self.mg_stat_dim))
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

    def forward(self, x, deep_supervision=False):
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        step_preds = []
        for _ in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            step_preds.append(self.head(y).squeeze(1))
        return step_preds if deep_supervision else step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# 2B. MODEL — V14B: Tokenized Attention with Shared-Weight SA TRM Loop
# ══════════════════════════════════════════════════════════════════════

class TokenizedTRM(nn.Module):
    """V14B: ALL features tokenized + Shared-Weight SA (attention TRM loop).

    Token groups (all attend to each other):
      - Magpie:  22 tokens × 6d  → project(6  → d_attn) = 22 × 64
      - DEML:    13 tokens × 5d  → project(5  → d_attn) = 13 × 64
      - Alloy:   23 tokens × 1d  → project(1  → d_attn) = 23 × 64
      Total: 58 tokens × 64d

    Shared-weight SA (TRM attention loop):
      The self-attention layer is applied TWICE with the SAME weights.
      - Pass 1: learns pairwise property interactions
      - Pass 2: refines higher-order interactions using same transform
      This is a 2-step TRM recursion in the attention feature extractor.

    Cross-attention: all tokens attend to Mat2Vec context (1 × 64d).
    Pool: mean-pool tokens → [64], concat flat "other" extras → pool → [96].
    MLP recursive reasoning: standard 20-step TRM loop.
    """
    def __init__(self, mg_n_props=22, mg_stat_dim=6,
                 deml_n_props=13, deml_stat_dim=5,
                 n_alloy=23, n_other=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, max_steps=20, sa_passes=2, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.sa_passes = sa_passes

        # Store sizes for input splitting
        self.mg_n_props, self.mg_stat_dim = mg_n_props, mg_stat_dim
        self.mg_dim = mg_n_props * mg_stat_dim
        self.deml_n_props, self.deml_stat_dim = deml_n_props, deml_stat_dim
        self.deml_dim = deml_n_props * deml_stat_dim
        self.n_alloy = n_alloy
        self.n_other = n_other

        # ── Token projections (each group → d_attn) ──────────────────
        self.mg_proj = nn.Sequential(
            nn.Linear(mg_stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.deml_proj = nn.Sequential(
            nn.Linear(deml_stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.alloy_proj = nn.Sequential(
            nn.Linear(1, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # ── Shared-Weight Self-Attention (applied sa_passes times) ────
        self.sa = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa_n = nn.LayerNorm(d_attn)
        self.sa_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa_fn = nn.LayerNorm(d_attn)

        # ── Cross-Attention to Mat2Vec context ────────────────────────
        self.ca = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d_attn)

        # ── Pool: attention output + flat extras → d_hidden ───────────
        pool_in = d_attn + (n_other if n_other > 0 else 0)
        self.pool = nn.Sequential(
            nn.Linear(pool_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())

        # ── MLP-TRM recursive reasoning (shared weights) ─────────────
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

    def _tokenize_and_attend(self, x):
        B = x.size(0)

        # ── Split input into blocks ───────────────────────────────────
        idx = 0
        mg_flat = x[:, idx:idx+self.mg_dim];        idx += self.mg_dim
        deml_flat = x[:, idx:idx+self.deml_dim];     idx += self.deml_dim
        alloy_flat = x[:, idx:idx+self.n_alloy];     idx += self.n_alloy
        other_flat = x[:, idx:idx+self.n_other];     idx += self.n_other
        m2v_flat = x[:, idx:]                         # remaining = Mat2Vec (200d)

        # ── Create tokens ─────────────────────────────────────────────
        # Magpie: [B, 22, 6] → project → [B, 22, 64]
        mg_tok = self.mg_proj(mg_flat.view(B, self.mg_n_props, self.mg_stat_dim))

        # DEML: [B, 13, 5] → project → [B, 13, 64]
        deml_tok = self.deml_proj(deml_flat.view(B, self.deml_n_props, self.deml_stat_dim))

        # Alloy: [B, 23, 1] → project → [B, 23, 64]
        alloy_tok = self.alloy_proj(alloy_flat.unsqueeze(-1))

        # Concatenate all tokens: [B, 58, 64]
        tok = torch.cat([mg_tok, deml_tok, alloy_tok], dim=1)

        # Mat2Vec context: [B, 1, 64]
        ctx = self.m2v_proj(m2v_flat).unsqueeze(1)

        # ── Shared-Weight SA TRM Loop (same weights, multiple passes) ─
        for _ in range(self.sa_passes):
            tok = self.sa_n(tok + self.sa(tok, tok, tok)[0])
            tok = self.sa_fn(tok + self.sa_ff(tok))

        # ── Cross-Attention to Mat2Vec chemistry context ──────────────
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx)[0])

        # ── Pool ──────────────────────────────────────────────────────
        pooled = tok.mean(dim=1)  # [B, d_attn]

        if self.n_other > 0:
            pooled = torch.cat([pooled, other_flat], dim=-1)

        return self.pool(pooled)  # [B, d_hidden]

    def forward(self, x, deep_supervision=False):
        B = x.size(0)
        xp = self._tokenize_and_attend(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        step_preds = []
        for _ in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            step_preds.append(self.head(y).squeeze(1))
        return step_preds if deep_supervision else step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# 3. LOSS + UTILS + TRAINING
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
# 4. MAIN BENCHMARK — V14A + V14B (single seed each)
# ══════════════════════════════════════════════════════════════════════

def run_benchmark():
    t0 = time.time()
    print("\n" + "═"*72)
    print("  TRM-MatSci V14 │ Mega Features │ Experimental Comparison")
    print("  V14A: Flat features + 2-Layer SA (control)")
    print("  V14B: Tokenized features + Shared-Weight SA TRM loop")
    print(f"  Seed: {SEED} │ Standard DS │ 20 steps")
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
    os.makedirs('trm_models_v14', exist_ok=True)
    dl_kw = dict(batch_size=32, num_workers=0)

    # ── CONFIGS ───────────────────────────────────────────────────────
    # V14A: Flat features (DEML + alloy + other all concatenated as extra)
    n_extra_a = feat.n_deml + feat.n_alloy + feat.n_other

    # V14B: Tokenized (alloy/DEML as tokens, "other" as flat extra)
    configs = {
        'V14A-Flat': {
            'model_cls': DeepHybridTRM,
            'model_kw': dict(
                mg_n_props=feat.mg_n_props, mg_stat_dim=feat.mg_stat_dim,
                n_extra=n_extra_a, mat2vec_dim=200,
                d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                dropout=0.2, max_steps=20),
        },
        'V14B-Tokenized': {
            'model_cls': TokenizedTRM,
            'model_kw': dict(
                mg_n_props=feat.mg_n_props, mg_stat_dim=feat.mg_stat_dim,
                deml_n_props=feat.deml_n_props, deml_stat_dim=feat.deml_stat_dim,
                n_alloy=feat.n_alloy, n_other=feat.n_other, mat2vec_dim=200,
                d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                dropout=0.2, max_steps=20, sa_passes=2),
        },
    }

    # Print specs
    print(f"\n  {'Config':<20} {'Params':>10} {'SA Layers':>12} {'Tokens':>8}")
    print(f"  {'─'*54}")
    for cname, cfg in configs.items():
        _m = cfg['model_cls'](**cfg['model_kw'])
        np_ = _m.count_parameters(); del _m
        cfg['n_params'] = np_
        if cname == 'V14A-Flat':
            print(f"  {cname:<20} {np_:>10,} {'2 (separate)':>12} {'22':>8}")
        else:
            n_tok = feat.mg_n_props + feat.deml_n_props + feat.n_alloy
            print(f"  {cname:<20} {np_:>10,} {'2 (shared)':>12} {n_tok:>8}")
    print()

    # ── TRAIN + EVALUATE ──────────────────────────────────────────────
    all_results = {}
    all_hists = {}

    for cname, cfg in configs.items():
        print(f"\n{'▓'*72}")
        print(f"  {cname}")
        print(f"{'▓'*72}")

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

            model = cfg['model_cls'](**cfg['model_kw']).to(device)
            bv, model, hist = train_fold(model, tr_dl, vl_dl, device,
                                          fold=fi+1, name=cname)
            fold_hists.append(hist)

            pred = predict(model, te_dl, device)
            mae = F.l1_loss(pred, te_tgt).item()
            fold_maes.append(mae)
            log.info(f"  [{cname}] F{fi+1}: MAE={mae:.2f}  (val {bv:.2f})")

            torch.save({'model_state': model.state_dict(), 'test_mae': mae,
                        'config': cname},
                       f'trm_models_v14/{cname}_fold{fi+1}.pt')

            del model; torch.cuda.empty_cache() if device.type == 'cuda' else None

        avg = float(np.mean(fold_maes))
        std = float(np.std(fold_maes))
        all_results[cname] = {
            'avg': avg, 'std': std, 'folds': fold_maes,
            'params': cfg['n_params'],
        }
        all_hists[cname] = fold_hists
        print(f"\n  ═══ {cname}: {avg:.4f} ±{std:.4f} MPa ═══")

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════
    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V14 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'Params':>10} {'MAE(MPa)':>10} {'±Std':>8}")
    print(f"  {'─'*54}")
    for n, r in sorted(all_results.items(), key=lambda x: x[1]['avg']):
        tag = ("  ← BEATS MODNet 🏆" if r['avg'] < 87.76 else
               "  ← BEATS V13A ✓"  if r['avg'] < 91.20 else
               "  ← BEATS V12A ✓"  if r['avg'] < 95.99 else
               "  ← BEATS V11B ✓"  if r['avg'] < 102.30 else "")
        print(f"  {n:<22} {r['params']:>9,} "
              f"{r['avg']:>10.4f} {r['std']:>8.4f}{tag}")
    print(f"  {'─'*54}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<22} {'baseline':>10} {bv:>10.4f}")
    print(f"\n  Total time: {tt/60:.1f} minutes")

    # Per-fold breakdown
    print(f"\n{'═'*72}")
    print(f"  PER-FOLD BREAKDOWN")
    print(f"{'═'*72}")
    cnames = list(all_results.keys())
    header = f"  {'Fold':<6}"
    for cn in cnames: header += f" {cn:>18}"
    print(header)
    print(f"  {'─'*46}")
    for fi in range(5):
        row = f"  {fi+1:<6}"
        for cn in cnames: row += f" {all_results[cn]['folds'][fi]:>18.2f}"
        print(row)
    print()

    generate_plots(all_results, all_hists)
    save_summary(all_results, tt)
    return all_results


# ══════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ══════════════════════════════════════════════════════════════════════

PAL = {'V14A-Flat': '#1565C0', 'V14B-Tokenized': '#E65100'}

def generate_plots(all_results, all_hists):
    names = list(all_results.keys())
    avgs = [all_results[n]['avg'] for n in names]
    stds = [all_results[n]['std'] for n in names]
    cols = [PAL.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ── Plot 1: Bar chart vs baselines ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(names, avgs, yerr=stds, capsize=6, color=cols,
                   alpha=0.88, edgecolor='white', linewidth=1.5)
    for bv, c, ls, lb in [
        (79.95, '#2E7D32', '--', 'TPOT-Mat (79.95)'),
        (87.76, '#F57F17', '--', 'MODNet (87.76)'),
        (91.20, '#9C27B0', '-.', 'V13A ens (91.20)'),
        (95.99, '#4CAF50', '-.', 'V12A (95.99)'),
        (102.30, '#9E9E9E', ':', 'V11B (102.30)'),
    ]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    for bar, m, s in zip(bars, avgs, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+1,
                 f'{m:.1f}', ha='center', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_ylabel('MAE (MPa)'); ax1.set_ylim(0, max(avgs)*1.5)
    ax1.set_title('V14 Results vs Baselines', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # ── Plot 2: Per-fold grouped bars ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(1, 6)
    w = 0.35
    for i, (n, col) in enumerate(zip(names, cols)):
        ax2.bar(x + (i - 0.5) * w, all_results[n]['folds'], w,
                color=col, alpha=0.8, label=n, edgecolor='white')
    ax2.axhline(91.20, color='#9C27B0', ls='-.', lw=1.5, label='V13A ens (91.20)')
    ax2.axhline(87.76, color='#F57F17', ls='--', lw=1.5, label='MODNet (87.76)')
    ax2.set_xlabel('Fold'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_xticks(x); ax2.set_xticklabels([f'F{i}' for i in range(1,6)])
    ax2.set_title('Per-Fold Comparison', fontweight='bold')
    ax2.legend(fontsize=7); ax2.grid(axis='y', alpha=0.2)

    # ── Plot 3: Training curves V14A ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for cname, col in PAL.items():
        if cname not in all_hists: continue
        for fi, h in enumerate(all_hists[cname]):
            lb_tr = f'{cname} train' if fi == 0 else None
            lb_vl = f'{cname} val'   if fi == 0 else None
            ax3.plot(h['train'], alpha=0.3, lw=0.8, color=col, label=lb_tr)
            ax3.plot(h['val'],   alpha=0.7, lw=1.2, color=col, label=lb_vl,
                     linestyle='--')
    ax3.axhline(91.20, color='#9C27B0', ls='-.', lw=1.2, label='V13A ens (91.20)')
    ax3.axvline(200, color='#4CAF50', ls='--', lw=1.2, alpha=0.6, label='SWA start')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE (MPa)')
    ax3.set_title('Training Curves (all folds)', fontweight='bold')
    ax3.legend(fontsize=6, ncol=2); ax3.grid(alpha=0.2)
    ax3.set_ylim(0, 300)

    # ── Plot 4: Fold-level delta ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if len(names) == 2:
        a_folds = all_results[names[0]]['folds']
        b_folds = all_results[names[1]]['folds']
        deltas = [b - a for a, b in zip(a_folds, b_folds)]
        colors4 = ['#2E7D32' if d < 0 else '#D32F2F' for d in deltas]
        ax4.bar(x, deltas, color=colors4, alpha=0.8, edgecolor='white')
        ax4.axhline(0, color='black', lw=1)
        for i, d in enumerate(deltas):
            ax4.text(i+1, d + (1 if d >= 0 else -2),
                    f'{d:+.1f}', ha='center', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('Δ MAE (V14B − V14A)')
        ax4.set_xticks(x); ax4.set_xticklabels([f'F{i}' for i in range(1,6)])
        ax4.set_title('V14B vs V14A: Per-Fold Difference', fontweight='bold')
        ax4.grid(axis='y', alpha=0.2)

    fig.suptitle('TRM-MatSci V14 │ Mega Features + Tokenized Attention │ matbench_steels',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v14.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: trm_results_v14.png")


def save_summary(all_results, total_s):
    s = {
        'version': 'V14', 'task': 'matbench_steels',
        'strategy': 'Mega Features: Flat vs Tokenized Attention',
        'seed': SEED,
        'total_min': round(total_s/60, 1),
        'models': {n: {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()}
                   for n, r in all_results.items()},
    }
    with open('trm_models_v14/summary_v14.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    log.info("✓ Saved: summary_v14.json")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v14_all", "zip", "trm_models_v14")
    log.info("✓ Created trm_v14_all.zip")
