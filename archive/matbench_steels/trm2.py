"""
╔══════════════════════════════════════════════════════════════════════╗
║     TRM-MatSci V2 │ Element-Token Input + Novel Dual-Ref Arch        ║
║     Dataset  : MatBench Steels │ 312 samples │ Yield Strength MPa    ║
║     Models   : 6 total — 2 MLP + 2 Trans-Normal + 2 Trans-Novel      ║
║     Hardware : Kaggle P100 (GPU must stay at ~100% — pre-computed)   ║
╚══════════════════════════════════════════════════════════════════════╝

INSTALL (run once):
  !pip install pymatgen torch tqdm matplotlib gensim scikit-learn matminer -q

KEY CHANGES FROM V1:
  1. Input: each element = separate 205-dim token (Mat2Vec 200 + 5 properties)
  2. Variable-length sequences (2-15 elements) with padding + attention mask
  3. GPU pipeline: features pre-computed at init, num_workers=4, pin_memory=True
  4. Trans-Novel: dual-reference architecture (fixed E0 + living Et)
  5. 300 epochs, dropout=0.2, batch=32, stratified val split
"""

import os, copy, json, math, time, logging, warnings, urllib.request
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from pymatgen.core import Composition, Element
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

# ─────────────────────────────────────────────────────────────────────────────
# SEEDS & LOGGING
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s │ %(levelname)s │ %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger("TRM2")

BASELINES = {
    'TPOT-Mat (best)':  79.9468, 'AutoML-Mat': 82.3043,
    'MODNet v0.1.12':   87.7627, 'RF-SCM/Magpie': 103.5125,
    'CrabNet':         107.3160, 'Dummy':          229.7445,
}

VALID_STEEL_ELEMENTS = {
    'Fe','C','Mn','Si','Cr','Ni','Mo','V','W','Co','Cu','Ti','Nb',
    'Al','N','S','P','B','Zr','Ta','Re','Hf','Se','Te','Ca','Mg','Sn'
}

MAX_ELEMENTS = 15   # max token sequence length
TOKEN_DIM    = 205  # Mat2Vec(200) + 5 elemental properties


# ══════════════════════════════════════════════════════════════════════════════
# 1. ELEMENT TOKEN FEATURIZER
# ══════════════════════════════════════════════════════════════════════════════

class ElementTokenFeaturizer:
    """
    Converts a pymatgen Composition into padded element token sequences.

    Each token = [mat2vec(200) | atomic_radius | electronegativity | group | period | fraction]
                = 205 dims total

    Output:
        tokens : np.array [MAX_ELEMENTS, 205]  — padded with zeros
        mask   : np.array [MAX_ELEMENTS]       — True = PAD (ignore in attention)
    """

    GCS_BASE = "https://storage.googleapis.com/mat2vec/"
    FILES    = ["pretrained_embeddings",
                "pretrained_embeddings.wv.vectors.npy",
                "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache_dir="mat2vec_cache"):
        self.cache_dir = cache_dir
        self.w2v       = self._download_and_load()
        self.embeddings = {}
        for el in Element:
            sym = el.symbol
            if sym in self.w2v.wv:
                self.embeddings[sym] = np.array(self.w2v.wv[sym], dtype=np.float32)
        log.info(f"✓ Mat2Vec: {len(self.embeddings)} elements, dim=200")
        assert len(self.embeddings) >= 20, "Mat2Vec download failed — delete mat2vec_cache/ and retry"
        self.scaler = None   # fitted per fold on training data only

    def _download_and_load(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        for fname in self.FILES:
            fpath = os.path.join(self.cache_dir, fname)
            if not os.path.exists(fpath):
                log.info(f"Downloading {fname}...")
                urllib.request.urlretrieve(self.GCS_BASE + fname, fpath)
        return Word2Vec.load(os.path.join(self.cache_dir, "pretrained_embeddings"))

    def _scalar_features(self, sym: str, frac: float) -> np.ndarray:
        el = Element(sym)
        return np.array([
            float(el.atomic_radius or 1.4),
            float(el.X or 1.5),
            float(el.group   if el.group   is not None else 8),
            float(el.row     if el.row     is not None else 4),
            float(frac),
        ], dtype=np.float32)

    def featurize(self, comp: Composition):
        """Returns (tokens [MAX_ELEMENTS, 205], mask [MAX_ELEMENTS])."""
        tokens = np.zeros((MAX_ELEMENTS, TOKEN_DIM), dtype=np.float32)
        mask   = np.ones(MAX_ELEMENTS, dtype=bool)   # True = ignore (PAD)
        items  = list(comp.get_el_amt_dict().items())[:MAX_ELEMENTS]
        # Normalize fractions to sum to 1
        total = sum(f for _, f in items)
        for i, (sym, frac) in enumerate(items):
            m2v     = self.embeddings.get(sym, np.zeros(200, dtype=np.float32))
            scalars = self._scalar_features(sym, frac / max(total, 1e-8))
            tokens[i, :200] = m2v
            tokens[i, 200:] = scalars
            mask[i]         = False
        return tokens, mask

    def fit_scaler(self, train_comps):
        """Fit StandardScaler on scalar features of training compositions only."""
        scalars = []
        for comp in train_comps:
            items = list(comp.get_el_amt_dict().items())
            total = sum(f for _, f in items)
            for sym, frac in items:
                scalars.append(self._scalar_features(sym, frac / max(total, 1e-8)))
        self.scaler = StandardScaler().fit(np.array(scalars))

    def apply_scaler(self, tokens: np.ndarray) -> np.ndarray:
        """Normalize scalar part (cols 200:205) using fitted scaler."""
        if self.scaler is None:
            return tokens
        out = tokens.copy()
        valid = ~np.all(tokens == 0, axis=1)   # non-PAD rows
        if valid.any():
            out[valid, 200:] = self.scaler.transform(tokens[valid, 200:])
        return out

    def validate_steel(self, comp: Composition) -> bool:
        syms = {str(e) for e in comp.elements}
        return ('Fe' in syms and
                comp.get_atomic_fraction('Fe') >= 0.50 and
                syms.issubset(VALID_STEEL_ELEMENTS))

    # Legacy: fraction-weighted sum for MLP models
    def featurize_pooled(self, comp: Composition) -> np.ndarray:
        vec, total = np.zeros(200, dtype=np.float32), 0.0
        for sym, frac in comp.get_el_amt_dict().items():
            if sym in self.embeddings:
                vec += frac * self.embeddings[sym]
                total += frac
        return vec / max(total, 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASETS — pre-computed at __init__ for zero runtime overhead
# ══════════════════════════════════════════════════════════════════════════════

class SteelsTokenDataset(Dataset):
    """Token dataset for Transformer models. All features pre-computed."""
    def __init__(self, comps, targets, featurizer: ElementTokenFeaturizer):
        self.data = []
        for comp, tgt in zip(comps, targets):
            tokens, mask = featurizer.featurize(comp)
            tokens = featurizer.apply_scaler(tokens)
            self.data.append((
                torch.tensor(tokens, dtype=torch.float32),
                torch.tensor(mask,   dtype=torch.bool),
                torch.tensor(float(tgt), dtype=torch.float32),
            ))
    def __len__(self):          return len(self.data)
    def __getitem__(self, i):   return self.data[i]


class SteelsVecDataset(Dataset):
    """Pooled-vector dataset for MLP models. All features pre-computed."""
    def __init__(self, comps, targets, featurizer: ElementTokenFeaturizer):
        self.data = []
        for comp, tgt in zip(comps, targets):
            vec = featurizer.featurize_pooled(comp)
            self.data.append((
                torch.tensor(vec,          dtype=torch.float32),
                torch.tensor(float(tgt),   dtype=torch.float32),
            ))
    def __len__(self):          return len(self.data)
    def __getitem__(self, i):   return self.data[i]


# ══════════════════════════════════════════════════════════════════════════════
# 3A. MLP-TRM  (attention-weighted pool → MLP recursive loop)
# ══════════════════════════════════════════════════════════════════════════════

class MLPTRM(nn.Module):
    """
    Dual-state TRM with MLP core.
    Input: fraction-weighted pooled 200-dim Mat2Vec vector.
    """
    def __init__(self, hidden_dim=64, ff_dim=128, dropout=0.2, steps=16):
        super().__init__()
        self.steps = steps
        self.D     = hidden_dim
        self.arch_type = 'MLP'

        self.input_proj = nn.Sequential(
            nn.Linear(200, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

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

    def forward(self, x, mask=None, return_trajectory=False):
        # x: [B, 200] (pooled vector from MLP dataset)
        B = x.size(0)
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
# 3B. TRANS-NORMAL-TRM  (pool → attention over artificial [x,y,z] tokens)
# ══════════════════════════════════════════════════════════════════════════════

class TransNormalTRM(nn.Module):
    """
    Trans-Normal: fraction-weighted mean pool of element tokens → single x_proj,
    then standard TRM loop: attention over synthetic [x_proj, y, z] triplet.
    Identical to V1 Transformer-TRM but with richer 205-dim pooled input.
    """
    def __init__(self, hidden_dim=256, ff_dim=256, nhead=4, dropout=0.2, steps=16):
        super().__init__()
        self.steps = steps
        self.D     = hidden_dim
        self.arch_type = 'Trans-Normal'

        self.input_proj = nn.Sequential(
            nn.Linear(TOKEN_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        self.z_attn  = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.z_ffn   = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(ff_dim, hidden_dim))
        self.z_norm1 = nn.LayerNorm(hidden_dim)
        self.z_norm2 = nn.LayerNorm(hidden_dim)

        self.y_attn  = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.y_ffn   = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(ff_dim, hidden_dim))
        self.y_norm1 = nn.LayerNorm(hidden_dim)
        self.y_norm2 = nn.LayerNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _pool(self, tokens, mask):
        # fraction-weighted mean pool: tokens [B,N,TOKEN_DIM], mask [B,N] (True=PAD)
        proj = self.input_proj(tokens)               # [B, N, D]
        valid = (~mask).float().unsqueeze(-1)        # [B, N, 1]
        pooled = (proj * valid).sum(1) / valid.sum(1).clamp(min=1)
        return pooled                                # [B, D]

    def forward(self, tokens, mask, return_trajectory=False):
        B   = tokens.size(0)
        xp  = self._pool(tokens, mask)               # [B, D]
        z   = torch.zeros(B, self.D, device=tokens.device)
        y   = torch.zeros(B, self.D, device=tokens.device)
        traj = []

        for _ in range(self.steps):
            # z update: attends over [xp, y, z]
            seq    = torch.stack([xp, y, z], dim=1)  # [B, 3, D]
            az, _  = self.z_attn(seq, seq, seq)      # [B, 3, D]
            z_res  = self.z_norm1(z + az[:, 2, :])   # take 3rd token (z)
            z      = self.z_norm2(z_res + self.z_ffn(z_res))

            # y update: attends over [xp, z, y] (richer context)
            seq_y  = torch.stack([xp, z, y], dim=1)  # [B, 3, D]
            ay, _  = self.y_attn(seq_y, seq_y, seq_y)
            y_res  = self.y_norm1(y + ay[:, 2, :])   # take 3rd token (y)
            y      = self.y_norm2(y_res + self.y_ffn(y_res))

            if return_trajectory:
                traj.append(self.head(y).squeeze(1))

        out = self.head(y).squeeze(1)
        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 3C. TRANS-NOVEL-TRM  (dual-reference: E0 fixed + Et living, both used by z)
# ══════════════════════════════════════════════════════════════════════════════

class TransNovelTRM(nn.Module):
    """
    Trans-Novel: the NEW architecture.

    TRM loop runs OVER all element tokens simultaneously (not over a pooled vector).
    Two parallel element representations:
      E0 = projected tokens, FIXED (original reference — stable gradient path)
      Et = copy of E0, EVOLVES via self-attention each step (living state)

    At each step:
      Et = Et + SelfAttn(Et)                     ← elements update each other
      z  = z  + CrossAttn(z→Et) + CrossAttn(z→E0) ← z reads both living + reference
      y  = y  + CrossAttn(y→z)                   ← y reads z as usual

    This lets z compare "where are we now" (Et) vs "where we started" (E0),
    enabling it to track how element understanding has evolved over 16 steps.
    """
    def __init__(self, hidden_dim=256, ff_dim=256, nhead=4, dropout=0.2, steps=16):
        super().__init__()
        self.steps = steps
        self.D     = hidden_dim
        self.arch_type = 'Trans-Novel'

        self.input_proj = nn.Sequential(
            nn.Linear(TOKEN_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        # Shared weights for all 16 steps:
        # 1. Element self-attention (Et evolves)
        self.el_self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.el_ffn       = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(ff_dim, hidden_dim))
        self.el_norm1     = nn.LayerNorm(hidden_dim)
        self.el_norm2     = nn.LayerNorm(hidden_dim)

        # 2. z cross-attends to Et (living)
        self.z_live_attn  = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.z_live_norm  = nn.LayerNorm(hidden_dim)

        # 3. z cross-attends to E0 (reference)
        self.z_ref_attn   = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.z_ref_norm   = nn.LayerNorm(hidden_dim)

        # 4. z FFN after both cross-attentions
        self.z_ffn        = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(ff_dim, hidden_dim))
        self.z_ffn_norm   = nn.LayerNorm(hidden_dim)

        # 5. y cross-attends to z
        self.y_attn       = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.y_ffn        = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(ff_dim, hidden_dim))
        self.y_norm1      = nn.LayerNorm(hidden_dim)
        self.y_norm2      = nn.LayerNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, tokens, mask, return_trajectory=False):
        """
        tokens : [B, MAX_ELEMENTS, TOKEN_DIM]
        mask   : [B, MAX_ELEMENTS]  True = PAD
        """
        B = tokens.size(0)
        D = self.D

        E0 = self.input_proj(tokens)              # [B, N, D] — fixed reference
        Et = E0.clone()                           # [B, N, D] — living state

        z  = torch.zeros(B, D, device=tokens.device)
        y  = torch.zeros(B, D, device=tokens.device)
        traj = []

        for _ in range(self.steps):
            # ── 1. Living elements self-attend ────────────────────────
            et_res, _ = self.el_self_attn(Et, Et, Et, key_padding_mask=mask)
            Et        = self.el_norm1(Et + et_res)
            Et        = self.el_norm2(Et + self.el_ffn(Et))

            # ── 2. z cross-attends to living Et ──────────────────────
            zq         = z.unsqueeze(1)                 # [B, 1, D]
            z_live, _  = self.z_live_attn(zq, Et, Et, key_padding_mask=mask)
            z          = self.z_live_norm(z + z_live.squeeze(1))

            # ── 3. z cross-attends to fixed E0 ───────────────────────
            z_ref, _   = self.z_ref_attn(z.unsqueeze(1), E0, E0, key_padding_mask=mask)
            z          = self.z_ref_norm(z + z_ref.squeeze(1))

            # ── 4. z FFN ─────────────────────────────────────────────
            z          = self.z_ffn_norm(z + self.z_ffn(z))

            # ── 5. y cross-attends to z ───────────────────────────────
            # Fix: y attends over the full [z, y] sequence instead of 1-to-1
            seq_y      = torch.stack([z, y], dim=1)     # [B, 2, D]
            ay, _      = self.y_attn(seq_y, seq_y, seq_y)
            y_res      = self.y_norm1(y + ay[:, 1, :])  # take 2nd token (y)
            y          = self.y_norm2(y_res + self.y_ffn(y_res))

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
    """Split a local targets array into train/val LOCAL indices (0..len-1).
    
    Args:
        targets: 1-D numpy array of target values (the fold's train+val targets).
        val_size: fraction to hold out for validation.
        seed: random seed.
    Returns:
        (train_indices, val_indices) — local indices into `targets`.
    """
    bins   = np.percentile(targets, [25, 50, 75])
    labels = np.digitize(targets, bins)
    train_idx, val_idx = [], []
    rng = np.random.RandomState(seed)
    for b in range(4):
        bin_mask = np.where(labels == b)[0]
        if len(bin_mask) == 0:
            continue
        n_val    = max(1, int(len(bin_mask) * val_size))
        chosen   = rng.choice(bin_mask, n_val, replace=False)
        val_idx.extend(chosen.tolist())
        train_idx.extend(np.setdiff1d(bin_mask, chosen).tolist())
    return np.array(train_idx), np.array(val_idx)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(model, train_loader, val_loader, device,
               epochs=300, fold_idx=1, config_name="", is_mlp=False):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    best_val  = float('inf')
    best_wts  = copy.deepcopy(model.state_dict())
    history   = {'train': [], 'val': []}
    patience, no_imp = 60, 0

    pbar = tqdm(range(epochs), desc=f"  [{config_name}] F{fold_idx}/5",
                leave=False, ncols=120)

    for epoch in pbar:
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            if is_mlp:
                bx, by = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                preds  = model(bx)
            else:
                bx, bmask, by = (batch[0].to(device, non_blocking=True),
                                 batch[1].to(device, non_blocking=True),
                                 batch[2].to(device, non_blocking=True))
                preds = model(bx, bmask)
            loss = F.l1_loss(preds, by)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(by)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if is_mlp:
                    bx, by = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                    preds  = model(bx)
                else:
                    bx, bmask, by = (batch[0].to(device, non_blocking=True),
                                     batch[1].to(device, non_blocking=True),
                                     batch[2].to(device, non_blocking=True))
                    preds = model(bx, bmask)
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
            'Best': f'{best_val:.1f}', 'LR': f'{scheduler.get_last_lr()[0]:.2e}'
        })

        if no_imp >= patience:
            log.info(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_wts)
    return best_val, model, history


# ══════════════════════════════════════════════════════════════════════════════
# 6. RECURSION TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

def eval_trajectory(model, loader, device, is_mlp):
    model.eval()
    step_preds, targets = None, []
    with torch.no_grad():
        for batch in loader:
            if is_mlp:
                bx, by = batch[0].to(device), batch[1].to(device)
                _, traj = model(bx, return_trajectory=True)
            else:
                bx, bmask, by = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                _, traj = model(bx, bmask, return_trajectory=True)
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
    print("  TRM-MatSci V2 │ 6-Model Benchmark │ Element-Token Input")
    print("  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV")
    print("═"*72 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        log.info("Device: CPU")

    log.info("Loading matbench_steels with OFFICIAL splits (seed=18012019)...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_steels")
    comps_raw   = df['composition'].tolist()
    targets_all = np.array(df['yield strength'].tolist(), dtype=np.float32)
    comps_all   = [Composition(c) for c in comps_raw]
    log.info(f"Dataset: {len(df)} samples")

    # Official MatBench splits — this seed is the published standard
    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    official_folds = list(kfold.split(comps_all))

    log.info("Loading Mat2Vec featurizer...")
    feat = ElementTokenFeaturizer()
    os.makedirs('trm_models_v2', exist_ok=True)

    CONFIGS = [
        (MLPTRM,         'MLP-S',          dict(hidden_dim=64,  ff_dim=128, dropout=0.2, steps=16),          True),
        (MLPTRM,         'MLP-L',          dict(hidden_dim=128, ff_dim=256, dropout=0.2, steps=16),          True),
        (TransNormalTRM, 'Trans-Normal-S', dict(hidden_dim=256, ff_dim=256, nhead=4, dropout=0.2, steps=16), False),
        (TransNovelTRM,  'Trans-Novel-S',  dict(hidden_dim=256, ff_dim=256, nhead=4, dropout=0.2, steps=16), False),
        (TransNormalTRM, 'Trans-Normal-L', dict(hidden_dim=256, ff_dim=512, nhead=4, dropout=0.2, steps=16), False),
        (TransNovelTRM,  'Trans-Novel-L',  dict(hidden_dim=256, ff_dim=512, nhead=4, dropout=0.2, steps=16), False),
    ]

    all_results, all_histories, all_trajectories = {}, {}, {}

    for cfg_idx, (ModelClass, cfg_name, cfg_kwargs, is_mlp) in enumerate(CONFIGS):
        t_cfg = time.time()
        print(f"\n{'═'*72}")
        print(f"  [{cfg_idx+1:02d}/06]  {cfg_name}")
        print(f"  hidden={cfg_kwargs.get('hidden_dim')}  ff={cfg_kwargs.get('ff_dim')}  "
              f"steps=16  dropout=0.2  epochs=300")
        print(f"{'─'*72}")
        _tmp = ModelClass(**cfg_kwargs)
        log.info(f"Params: {_tmp.count_parameters():,}")
        del _tmp

        fold_maes, fold_histories, fold_trajectories, fold_details = [], [], [], []

        for fold, (tv_idx, test_idx) in enumerate(official_folds):
            # ── Build local composition/target arrays from official indices ──
            tv_comps   = [comps_all[i] for i in tv_idx]
            tv_targets = targets_all[tv_idx]
            te_comps   = [comps_all[i] for i in test_idx]
            te_targets = targets_all[test_idx]

            torch.manual_seed(SEED + fold)

            # Stratified train/val split (LOCAL indices into tv_*)
            tr_idx, vl_idx = stratified_val_split(
                tv_targets, val_size=0.15, seed=SEED + fold)

            train_comps   = [tv_comps[i] for i in tr_idx]
            train_targets = tv_targets[tr_idx]
            val_comps     = [tv_comps[i] for i in vl_idx]
            val_targets   = tv_targets[vl_idx]

            # Fit scaler on training compositions ONLY
            feat.fit_scaler(train_comps)

            # Build datasets directly from local lists
            if is_mlp:
                train_ds = SteelsVecDataset(train_comps, train_targets, feat)
                val_ds   = SteelsVecDataset(val_comps,   val_targets,   feat)
                test_ds  = SteelsVecDataset(te_comps,    te_targets,    feat)
            else:
                train_ds = SteelsTokenDataset(train_comps, train_targets, feat)
                val_ds   = SteelsTokenDataset(val_comps,   val_targets,   feat)
                test_ds  = SteelsTokenDataset(te_comps,    te_targets,    feat)

            num_workers = 4 if device.type == 'cuda' else 0
            dl_kw = dict(batch_size=32, pin_memory=(device.type=='cuda'), num_workers=num_workers)
            if num_workers > 0:
                dl_kw.update(prefetch_factor=2, persistent_workers=True)
                
            train_dl = DataLoader(train_ds, shuffle=True,  **dl_kw)
            val_dl   = DataLoader(val_ds,   shuffle=False, **dl_kw)
            test_dl  = DataLoader(test_ds,  shuffle=False, **dl_kw)

            model = ModelClass(**cfg_kwargs).to(device)
            best_val, model, history = train_fold(
                model, train_dl, val_dl, device,
                epochs=300, fold_idx=fold+1,
                config_name=cfg_name, is_mlp=is_mlp)

            model.eval()
            te_loss = 0.0
            with torch.no_grad():
                for batch in test_dl:
                    if is_mlp:
                        bx, by = batch[0].to(device), batch[1].to(device)
                        preds  = model(bx)
                    else:
                        bx, bmask, by = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                        preds  = model(bx, bmask)
                    te_loss += F.l1_loss(preds, by).item() * len(by)
            te_mae = te_loss / len(test_dl.dataset)

            log.info(f"  Fold {fold+1}/5 → Test: {te_mae:.4f}  Val: {best_val:.4f}")
            fold_maes.append(te_mae)
            fold_histories.append(history)
            fold_details.append({'val_mae': best_val, 'test_mae': te_mae})
            fold_trajectories.append(eval_trajectory(model, test_dl, device, is_mlp))

            torch.save({
                'config_name': cfg_name, 'arch': ModelClass.__name__,
                'fold': fold+1, 'kwargs': cfg_kwargs,
                'test_mae': te_mae, 'val_mae': best_val,
                'model_state': model.state_dict(), 'history': history,
            }, f'trm_models_v2/{cfg_name}_fold{fold+1}.pt')

        avg_mae = float(np.mean(fold_maes))
        std_mae = float(np.std(fold_maes))
        t_cfg   = time.time() - t_cfg
        tag = ("  ← BEATS TPOT 🏆"   if avg_mae < 79.95  else
               "  ← Beats AutoML ✓"  if avg_mae < 82.30  else
               "  ← Beats MODNet ✓"  if avg_mae < 87.76  else
               "  ← Beats CrabNet ✓" if avg_mae < 107.32 else "")
        print(f"\n  ✓ {cfg_name}: {avg_mae:.4f} ± {std_mae:.4f} MPa{tag}"
              f"  (took {t_cfg/60:.1f} min)")

        all_results[cfg_name]      = {
            'arch': ModelClass.__name__, 'is_mlp': is_mlp,
            'avg_mae': avg_mae, 'std_mae': std_mae,
            'fold_maes': fold_maes, 'fold_details': fold_details,
            'params': ModelClass(**cfg_kwargs).count_parameters(),
        }
        all_histories[cfg_name]    = fold_histories
        all_trajectories[cfg_name] = np.mean(fold_trajectories, axis=0).tolist()

    t_total = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V2 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'Arch':<16} {'Params':>9} {'MAE(MPa)':>10} {'±Std':>8}")
    print(f"  {'─'*68}")
    for name, r in sorted(all_results.items(), key=lambda x: x[1]['avg_mae']):
        print(f"  {name:<22} {r['arch']:<16} {r['params']:>9,} "
              f"{r['avg_mae']:>10.4f} {r['std_mae']:>8.4f}")
    print(f"  {'─'*68}")
    for bname, bval in [('TPOT', 79.95), ('MODNet', 87.76), ('CrabNet', 107.32)]:
        print(f"  {bname:<22} {'baseline':<16} {'—':>9} {bval:>10.4f}")
    print(f"\n  Total time: {t_total/60:.1f} minutes\n")

    generate_plots(all_results, all_trajectories, all_histories)
    save_summary(all_results, all_trajectories, t_total)
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 8. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    'MLP-S':'#1565C0','MLP-L':'#42A5F5',
    'Trans-Normal-S':'#E53935','Trans-Normal-L':'#B71C1C',
    'Trans-Novel-S':'#F57F17','Trans-Novel-L':'#E65100',
}

def generate_plots(results, trajectories, histories):
    names  = list(results.keys())
    maes   = [results[n]['avg_mae'] for n in names]
    stds   = [results[n]['std_mae'] for n in names]
    params = [results[n]['params']  for n in names]
    colors = [PALETTE.get(n,'#888') for n in names]

    fig = plt.figure(figsize=(22, 20))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.32)

    # Panel 1: Main bar
    ax1  = fig.add_subplot(gs[0, :])
    bars = ax1.bar(names, maes, yerr=stds, capsize=5,
                   color=colors, alpha=0.88, edgecolor='white', linewidth=1.2)
    for bv, col, ls, lbl in [
        (79.95,'#2E7D32','--','TPOT (79.95)'),
        (87.76,'#F57F17','--','MODNet (87.76)'),
        (107.32,'#795548',':','CrabNet (107.32)'),
    ]:
        ax1.axhline(bv, color=col, linestyle=ls, linewidth=1.8, label=lbl, alpha=0.85)
    for bar, m, s in zip(bars, maes, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+2,
                 f'{m:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylabel('MAE (MPa)', fontsize=11)
    ax1.set_title('TRM-MatSci V2 — 6 Models × 5 Folds │ matbench_steels',
                  fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(maes)*1.35)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=10)

    # Panel 2: Params vs MAE
    ax2 = fig.add_subplot(gs[1, 0])
    for n, p, m, s in zip(names, params, maes, stds):
        c = PALETTE.get(n,'#888')
        ax2.scatter(p, m, s=100, color=c, zorder=5)
        ax2.errorbar(p, m, yerr=s, fmt='none', color=c, capsize=3)
        ax2.annotate(n, (p, m), textcoords='offset points', xytext=(4,3), fontsize=7)
    for bv, col, lbl in [(87.76,'#F57F17','MODNet'),(107.32,'#795548','CrabNet')]:
        ax2.axhline(bv, color=col, linestyle='--', linewidth=1.2, alpha=0.7, label=lbl)
    ax2.set_xlabel('Parameters', fontsize=10)
    ax2.set_ylabel('MAE (MPa)', fontsize=10)
    ax2.set_title('Scaling: Params vs MAE', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Panel 3: Normal vs Novel
    ax3 = fig.add_subplot(gs[1, 1])
    groups = [('Trans-Normal-S','Trans-Novel-S'),('Trans-Normal-L','Trans-Novel-L')]
    x = np.arange(len(groups)); w = 0.35
    ax3.bar(x-w/2, [results[g[0]]['avg_mae'] for g in groups], w,
            label='Trans-Normal', color='#E53935', alpha=0.85)
    ax3.bar(x+w/2, [results[g[1]]['avg_mae'] for g in groups], w,
            label='Trans-Novel',  color='#F57F17', alpha=0.85)
    ax3.axhline(107.32, color='#795548', linestyle=':', linewidth=1.5, label='CrabNet')
    ax3.axhline(87.76,  color='#2E7D32', linestyle='--',linewidth=1.5, label='MODNet', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Small (ff=256)','Large (ff=512)'])
    ax3.set_ylabel('MAE (MPa)', fontsize=10)
    ax3.set_title('Normal vs Novel Transformer\nHead-to-Head', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)

    # Panel 4: Recursion convergence
    ax4 = fig.add_subplot(gs[2, :])
    for n, traj in trajectories.items():
        ls = '-' if results[n]['is_mlp'] else ('--' if 'Normal' in n else ':')
        ax4.plot(range(1, len(traj)+1), traj, label=n,
                 color=PALETTE.get(n,'#888'), linewidth=2.0, linestyle=ls, alpha=0.9)
    ax4.axhline(107.32, color='#795548', linestyle=':', linewidth=2, label='CrabNet')
    ax4.axhline(87.76,  color='#F57F17', linestyle='--',linewidth=2, label='MODNet')
    ax4.axhline(79.95,  color='#2E7D32', linestyle='--',linewidth=2, label='TPOT')
    ax4.set_xlabel('Recursion Step (1 → 16)', fontsize=11)
    ax4.set_ylabel('MAE (MPa)', fontsize=11)
    ax4.set_title('Recursion Convergence — All 6 Models\n'
                  'Solid=MLP │ Dashed=Trans-Normal │ Dotted=Trans-Novel',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right', ncol=3); ax4.grid(alpha=0.3)

    plt.suptitle('TRM-MatSci V2 │ Full Results │ matbench_steels',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v2.png', dpi=150, bbox_inches='tight')
    log.info("✓ Saved: trm_results_v2.png")
    plt.close(fig)

    # Training curves (separate figure)
    fig2, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig2.suptitle('V2 Training Curves (Val MAE per Fold)', fontsize=13, fontweight='bold')
    for ax, n in zip(axes.flatten(), names):
        for fold_i, h in enumerate(histories[n]):
            ax.plot(h['train'], alpha=0.35, linewidth=1, color='#1565C0')
            ax.plot(h['val'],   alpha=0.85, linewidth=1.2,
                    color=PALETTE.get(n,'#888'), label=f'F{fold_i+1}')
        ax.axhline(107.32, color='#795548', linestyle=':', linewidth=1.2, label='CrabNet')
        ax.set_title(n, fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=8); ax.set_ylabel('MAE', fontsize=8)
        ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.2)
    plt.tight_layout()
    fig2.savefig('trm_curves_v2.png', dpi=130, bbox_inches='tight')
    plt.close(fig2)
    log.info("✓ Saved: trm_curves_v2.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(results, trajectories, total_s):
    summary = {
        'version': 'V2', 'task': 'matbench_steels',
        'input_type': 'element_tokens_205dim',
        'recursion_steps': 16, 'epochs': 300, 'dropout': 0.2,
        'total_train_min': round(total_s/60, 1),
        'baselines': BASELINES, 'models': {}
    }
    for name, r in results.items():
        summary['models'][name] = {
            'arch': r['arch'], 'params': r['params'],
            'avg_mae': round(r['avg_mae'],4), 'std_mae': round(r['std_mae'],4),
            'fold_maes': [round(m,4) for m in r['fold_maes']],
            'fold_details': r['fold_details'],
            'convergence': [round(v,4) for v in trajectories[name]],
            'beats_tpot':    r['avg_mae'] < 79.9468,
            'beats_modnet':  r['avg_mae'] < 87.7627,
            'beats_crabnet': r['avg_mae'] < 107.316,
        }
    with open('trm_models_v2/summary_v2.json','w') as f:
        json.dump(summary, f, indent=2)

    rows = []
    for name, r in results.items():
        for fi, (fm, fd) in enumerate(zip(r['fold_maes'], r['fold_details'])):
            rows.append({'model':name,'fold':fi+1,
                         'test_mae':round(fm,4),'val_mae':round(fd['val_mae'],4)})
    pd.DataFrame(rows).to_csv('trm_models_v2/fold_results_v2.csv', index=False)
    log.info("✓ Saved: trm_models_v2/summary_v2.json + fold_results_v2.csv")
    log.info(f"  → {6*5} checkpoints in trm_models_v2/")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_benchmark()
