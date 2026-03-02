"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V12 — Scaled + Expanded + Advanced Deep Supervision     ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV          ║
║                                                                      ║
║  V12A  Scaled + Expanded + Standard Deep Supervision                 ║
║        d_attn=64, nhead=4, d_hidden=96, ff_dim=150, 20 steps       ║
║        Expanded features (Magpie + Mat2Vec + Extra descriptors)      ║
║        Standard linear-weighted deep supervision                     ║
║                                                                      ║
║  V12B  Same arch + Confidence-Weighted Step Selection                ║
║        22 steps, confidence_head learns which step to trust          ║
║        Final prediction = softmax(confidence) · step_preds           ║
║        No ponder cost, no halting — always runs all steps            ║
║                                                                      ║
║  All models: Deep Supervision + SWA + AdamW + 300 epochs            ║
║  Baseline: V11B = 102.30 MPa (current best)                         ║
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
log = logging.getLogger("TRM12")
SEED = 42

BASELINES = {
    'TPOT-Mat':       79.9468,
    'AutoML-Mat':     82.3043,
    'MODNet':         87.7627,
    'RF-SCM/Magpie': 103.5125,
    'V11B (best)':   102.3003,
    'V10A':          103.2867,
    'CrabNet':       107.3160,
    'Darwin':        123.2932,
}


# ══════════════════════════════════════════════════════════════════════
# 1. FEATURIZER + DATASET
# ══════════════════════════════════════════════════════════════════════

class ExpandedFeaturizer:
    """Magpie (22 props × 6 stats) + Extra matminer descriptors + Mat2Vec (200d).

    Extra descriptors: ElementFraction, Stoichiometry, ValenceOrbital,
    IonProperty, BandCenter — all concatenated as a flat vector between
    the Magpie block and Mat2Vec.
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
        self.n_extra = None   # detected at featurize time

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
            try: mg = np.array(self.ep_magpie.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)

            try:
                ex = np.array(self.extra_feats.featurize(c), np.float32)
            except:
                ex = np.zeros(self.n_extra or 200, np.float32)

            if self.n_extra is None:
                self.n_extra = len(ex)
                log.info(f"Expanded features: {self.n_mg} Magpie + "
                         f"{self.n_extra} Extra + 200 Mat2Vec = "
                         f"{self.n_mg + self.n_extra + 200}d")

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
    """Scaled Hybrid-TRM with Deep Supervision and optional extra features.

    Used by V12A (standard deep supervision).
    """
    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
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

        # Pool with optional extra feature injection
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
            pooled = torch.cat([pooled, extra], dim=-1)

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


class ConfidenceHybridTRM(nn.Module):
    """V12B: Hybrid-TRM with Confidence-Weighted Step Selection.

    Instead of always using the final step, the model learns a confidence
    score at each step. The final prediction is a softmax-weighted average
    of all per-step predictions, where the weights come from the confidence
    head. This lets the model dynamically trust earlier or later steps
    depending on the sample.

    Training:
        - Deep supervision loss on all steps (linear weights)
        - Plus L1 loss on the confidence-weighted prediction
        - No ponder cost — avoids V11C's failure

    Inference:
        - Run all steps, use confidence-weighted prediction
    """
    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, max_steps=22, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.n_extra = n_extra

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

        # Pool with optional extra feature injection
        pool_in = d_attn + (n_extra if n_extra > 0 else 0)
        self.pool = nn.Sequential(
            nn.Linear(pool_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())

        # MLP-TRM recursive reasoning (shared weights)
        self.z_up = nn.Sequential(
            nn.Linear(d_hidden*3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.y_up = nn.Sequential(
            nn.Linear(d_hidden*2, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.head = nn.Linear(d_hidden, 1)

        # ── Confidence head: learns which step to trust ──────────────
        self.confidence_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, 1))  # raw logit, softmaxed later

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        # Initialize confidence bias with Gaussian prior centered at step 16
        # This gives the model a soft inductive bias toward middle-to-late steps
        # while allowing training to shift it per-sample
        with torch.no_grad():
            center = 16.0
            sigma = 4.0
            # We can't set step-dependent bias directly since weights are shared,
            # but we initialize the final bias to 0 so all steps start equal.
            # The learned representations at each step will naturally differentiate.
            nn.init.zeros_(self.confidence_head[-1].bias)

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

        pooled = tok.mean(dim=1)

        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=-1)

        return self.pool(pooled)

    def forward(self, x, deep_supervision=False, return_confidence=False):
        """Forward pass.

        Returns:
            deep_supervision=True:  (step_preds, confidence_logits)
                step_preds: list of [B] predictions, one per step
                confidence_logits: [B, max_steps] raw logits
            deep_supervision=False, return_confidence=False:
                weighted_pred: [B] confidence-weighted prediction
            deep_supervision=False, return_confidence=True:
                (weighted_pred, confidence_weights): [B], [B, max_steps]
        """
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)

        step_preds = []
        conf_logits = []

        for _ in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            step_preds.append(self.head(y).squeeze(1))
            conf_logits.append(self.confidence_head(y).squeeze(1))

        conf_logits = torch.stack(conf_logits, dim=1)  # [B, max_steps]

        if deep_supervision:
            return step_preds, conf_logits

        # Confidence-weighted prediction
        conf_weights = F.softmax(conf_logits, dim=1)  # [B, max_steps]
        preds_stack = torch.stack(step_preds, dim=1)   # [B, max_steps]
        weighted_pred = (preds_stack * conf_weights).sum(dim=1)  # [B]

        if return_confidence:
            return weighted_pred, conf_weights
        return weighted_pred

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


def confidence_ds_loss(step_preds, targets, conf_logits):
    """Advanced Deep Supervision: standard DS + confidence-weighted L1.

    Components:
    1. Standard linear-weighted deep supervision on all steps
    2. L1 loss on the confidence-weighted final prediction
    """
    # Standard deep supervision
    ds = deep_supervision_loss(step_preds, targets)

    # Confidence-weighted prediction loss
    conf_weights = F.softmax(conf_logits, dim=1)  # [B, max_steps]
    preds_stack = torch.stack(step_preds, dim=1)   # [B, max_steps]
    weighted_pred = (preds_stack * conf_weights).sum(dim=1)
    conf_loss = F.l1_loss(weighted_pred, targets)

    return ds + conf_loss


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
    """Training with standard deep supervision (V12A)."""
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


def train_fold_confidence(model, tr_dl, vl_dl, device,
                          epochs=300, swa_start=200, fold=1, name=""):
    """Training with confidence-weighted deep supervision (V12B)."""
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
            step_preds, conf_logits = model(bx, deep_supervision=True)
            loss = confidence_ds_loss(step_preds, by, conf_logits)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # Track confidence-weighted MAE for display
            with torch.no_grad():
                cw = F.softmax(conf_logits, dim=1)
                ps = torch.stack(step_preds, dim=1)
                wp = (ps * cw).sum(dim=1)
                tl += F.l1_loss(wp, by).item() * len(by)
        tl /= len(tr_dl.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)  # uses confidence-weighted by default
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


def predict_confidence(model, dl, device):
    """Predict using confidence model, also return per-step weights."""
    model.eval()
    all_preds, all_weights = [], []
    with torch.no_grad():
        for bx, _ in dl:
            pred, weights = model(bx.to(device), return_confidence=True)
            all_preds.append(pred.cpu())
            all_weights.append(weights.cpu())
    return torch.cat(all_preds), torch.cat(all_weights)


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
    print("  TRM-MatSci V12 │ Scaled + Expanded + Advanced DS │ matbench_steels")
    print("  V12A: Scaled arch + expanded features + standard deep supervision")
    print("  V12B: Same + confidence-weighted step selection (22 steps)")
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
    log.info("Computing EXPANDED features...")
    feat = ExpandedFeaturizer()
    X_all = feat.featurize_all(comps_all)
    n_extra = feat.n_extra
    log.info(f"Features: {X_all.shape} (n_extra={n_extra})")

    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    os.makedirs('trm_models_v12', exist_ok=True)
    dl_kw = dict(batch_size=32, num_workers=0)

    # ── CONFIGS ───────────────────────────────────────────────────────
    shared_kw = dict(n_props=22, stat_dim=6, n_extra=n_extra,
                     mat2vec_dim=200, d_attn=64, nhead=4,
                     d_hidden=96, ff_dim=150, dropout=0.2)

    configs = {
        'V12A-StdDS': {
            'model_cls': HybridTRM,
            'model_kw': {**shared_kw, 'max_steps': 20},
            'train_fn': train_fold_standard,
            'predict_fn': predict,
            'is_confidence': False,
        },
        'V12B-ConfDS': {
            'model_cls': ConfidenceHybridTRM,
            'model_kw': {**shared_kw, 'max_steps': 22},
            'train_fn': train_fold_confidence,
            'predict_fn': None,  # uses predict_confidence
            'is_confidence': True,
        },
    }

    # Print param counts
    print(f"\n  {'Config':<20} {'Params':>10} {'Steps':>8}")
    print(f"  {'─'*42}")
    for cname, cfg in configs.items():
        _m = cfg['model_cls'](**cfg['model_kw'])
        np_ = _m.count_parameters(); del _m
        cfg['n_params'] = np_
        steps = cfg['model_kw']['max_steps']
        print(f"  {cname:<20} {np_:>10,} {steps:>8}")
    print()

    # ── TRAIN + EVALUATE ──────────────────────────────────────────────
    all_results = {}
    all_hists = {}
    all_conf_weights = {}

    for cname, cfg in configs.items():
        print(f"\n{'▓'*72}")
        print(f"  {cname}")
        print(f"{'▓'*72}")

        fold_maes = []
        fold_hists = []
        fold_conf_w = []

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
            model = cfg['model_cls'](**cfg['model_kw']).to(device)
            bv, model, hist = cfg['train_fn'](model, tr_dl, vl_dl, device,
                                               fold=fi+1, name=cname)
            fold_hists.append(hist)

            # Predict
            if cfg['is_confidence']:
                pred, conf_w = predict_confidence(model, te_dl, device)
                fold_conf_w.append(conf_w)
                avg_peak = conf_w.argmax(dim=1).float().mean().item() + 1
                log.info(f"  [{cname}] F{fi+1}: MAE={F.l1_loss(pred, te_tgt).item():.2f}  "
                         f"(val {bv:.2f}, avg peak step={avg_peak:.1f})")
            else:
                pred = cfg['predict_fn'](model, te_dl, device)
                log.info(f"  [{cname}] F{fi+1}: MAE={F.l1_loss(pred, te_tgt).item():.2f}  "
                         f"(val {bv:.2f})")

            mae = F.l1_loss(pred, te_tgt).item()
            fold_maes.append(mae)

            torch.save({'model_state': model.state_dict(), 'test_mae': mae,
                        'config': cname},
                       f'trm_models_v12/{cname}_fold{fi+1}.pt')

        avg = float(np.mean(fold_maes))
        std = float(np.std(fold_maes))
        all_results[cname] = {
            'avg': avg, 'std': std, 'folds': fold_maes,
            'params': cfg['n_params'],
        }
        all_hists[cname] = fold_hists
        if fold_conf_w:
            all_conf_weights[cname] = fold_conf_w

        print(f"\n  ═══ {cname}: {avg:.4f} ±{std:.4f} MPa ═══")

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════

    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V12 (5-Fold Avg MAE)")
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
        header += f" {cn:>14}"
    print(header)
    print(f"  {'─'*38}")
    for fi in range(5):
        row = f"  {fi+1:<6}"
        for cn in cnames:
            row += f" {all_results[cn]['folds'][fi]:>14.2f}"
        print(row)

    # Confidence stats
    if all_conf_weights:
        print(f"\n  Confidence Step Selection Summary:")
        for cn, fw_list in all_conf_weights.items():
            all_w = torch.cat(fw_list, dim=0)  # [N_total, max_steps]
            avg_w = all_w.mean(dim=0)
            peak_step = avg_w.argmax().item() + 1
            avg_peak = all_w.argmax(dim=1).float().mean().item() + 1
            print(f"    {cn}: avg peak step={avg_peak:.1f}, "
                  f"population peak=step {peak_step}")
    print()

    generate_plots(all_results, all_hists, all_conf_weights)
    save_summary(all_results, all_hists, all_conf_weights, tt)
    return all_results


# ══════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════

PAL = {'V12A-StdDS': '#1565C0', 'V12B-ConfDS': '#E65100'}

def generate_plots(all_results, all_hists, all_conf_weights):
    names = list(all_results.keys())
    avgs = [all_results[n]['avg'] for n in names]
    stds = [all_results[n]['std'] for n in names]
    cols = [PAL.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ── Plot 1: Bar chart vs baselines ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(names, avgs, yerr=stds, capsize=6, color=cols,
                   alpha=0.88, edgecolor='white', linewidth=1.5)
    for bv, c, ls, lb in [
        (87.76, '#F57F17', '--', 'MODNet (87.76)'),
        (102.30, '#9E9E9E', '-.', 'V11B (102.30)'),
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
    ax1.set_title('V12 Results vs Baselines', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # ── Plot 2: Per-fold grouped bars ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(1, 6)
    w = 0.35
    for i, (n, col) in enumerate(zip(names, cols)):
        fold_vals = all_results[n]['folds']
        ax2.bar(x + (i - 0.5) * w, fold_vals, w, color=col, alpha=0.8,
                label=n, edgecolor='white')
    ax2.axhline(102.30, color='#9E9E9E', ls='-.', lw=1.5, label='V11B (102.30)')
    ax2.axhline(103.51, color='#B0BEC5', ls=':', lw=1.5, label='RF-SCM')
    ax2.set_xlabel('Fold'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_xticks(x); ax2.set_xticklabels([f'F{i}' for i in range(1,6)])
    ax2.set_title('Per-Fold Breakdown', fontweight='bold')
    ax2.legend(fontsize=7); ax2.grid(axis='y', alpha=0.2)

    # ── Plot 3: Training/Val loss curves ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for cname, col in PAL.items():
        if cname not in all_hists: continue
        for fi, h in enumerate(all_hists[cname]):
            lb_tr = f'{cname} train' if fi == 0 else None
            lb_vl = f'{cname} val'   if fi == 0 else None
            ax3.plot(h['train'], alpha=0.3, lw=0.8, color=col, label=lb_tr)
            ax3.plot(h['val'],   alpha=0.7, lw=1.2, color=col, label=lb_vl,
                     linestyle='--')
    ax3.axhline(102.30, color='#9E9E9E', ls='-.', lw=1.2, label='V11B (102.30)')
    ax3.axvline(200, color='#4CAF50', ls='--', lw=1.2, alpha=0.6, label='SWA start')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE (MPa)')
    ax3.set_title('Training Curves (all folds)', fontweight='bold')
    ax3.legend(fontsize=6, ncol=2); ax3.grid(alpha=0.2)
    ax3.set_ylim(0, 300)

    # ── Plot 4: Confidence distribution ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if all_conf_weights:
        for cn, fw_list in all_conf_weights.items():
            all_w = torch.cat(fw_list, dim=0)  # [N_total, max_steps]
            avg_w = all_w.mean(dim=0).numpy()
            steps = np.arange(1, len(avg_w)+1)
            ax4.bar(steps, avg_w, color=PAL.get(cn, '#E65100'), alpha=0.8,
                    label=f'{cn} avg confidence', edgecolor='white')
            # Show std as error bars
            std_w = all_w.std(dim=0).numpy()
            ax4.errorbar(steps, avg_w, yerr=std_w, fmt='none',
                        ecolor='#333', capsize=2, alpha=0.5)
        ax4.set_xlabel('Recursion Step')
        ax4.set_ylabel('Confidence Weight (softmax)')
        ax4.set_title('V12B: Where the Model Trusts Its Predictions',
                      fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(axis='y', alpha=0.2)
    else:
        ax4.text(0.5, 0.5, 'No confidence model trained',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Confidence Distribution', fontweight='bold')

    fig.suptitle('TRM-MatSci V12 │ Scaled + Expanded + Advanced DS │ matbench_steels',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v12.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: trm_results_v12.png")


def save_summary(all_results, all_hists, all_conf_weights, total_s):
    # Prepare confidence info
    conf_info = {}
    for cn, fw_list in all_conf_weights.items():
        all_w = torch.cat(fw_list, dim=0)
        conf_info[cn] = {
            'avg_weights': all_w.mean(dim=0).numpy().round(4).tolist(),
            'avg_peak_step': float(all_w.argmax(dim=1).float().mean().item() + 1),
        }

    s = {
        'version': 'V12', 'task': 'matbench_steels',
        'strategy': 'Scaled + Expanded Features + Confidence DS',
        'total_min': round(total_s/60, 1),
        'models': {n: {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()}
                   for n, r in all_results.items()},
        'confidence': conf_info,
    }
    with open('trm_models_v12/summary_v12.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    log.info("✓ Saved: summary_v12.json")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v12_all", "zip", "trm_models_v12")
    log.info("✓ Created trm_v12_all.zip")
