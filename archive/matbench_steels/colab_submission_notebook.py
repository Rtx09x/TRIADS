"""
╔══════════════════════════════════════════════════════════════════════╗
║  RudraNet — Matbench Submission Notebook (Google Colab)              ║
║  Copy each section into a separate Colab cell and run sequentially.  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies (CRITICAL ORDER)
# ══════════════════════════════════════════════════════════════════════
# We must install the modern dependencies first, then matbench with --no-deps.
# This prevents pip from "helpfully" downgrading to the broken version 0.5.

# !pip install matminer gensim pymatgen
# !pip install matbench --no-deps

# ══════════════════════════════════════════════════════════════════════
# CELL 2: Extract your model checkpoints
# ══════════════════════════════════════════════════════════════════════
# Run this cell to extract the weights you just uploaded.

import zipfile
import os

zip_path = 'trm_v13_all.zip'
extract_path = 'trm_models_v13'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✅ Extracted to {extract_path}/")
    print(f"Files found: {len(os.listdir(extract_path))}")
else:
    print(f"❌ File {zip_path} not found in /content/ folder!")


# ══════════════════════════════════════════════════════════════════════
# CELL 3: Model Definition (DeepHybridTRM — copied from trm13.py)
# ══════════════════════════════════════════════════════════════════════

import os, json, time, warnings, urllib.request
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from gensim.models import Word2Vec


class DeepHybridTRM(nn.Module):
    """V13A: 2-Layer SA Hybrid-TRM with Standard Deep Supervision."""

    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, max_steps=20, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.n_extra = n_extra

        self.tok_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        self.sa1 = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa1_n = nn.LayerNorm(d_attn)
        self.sa1_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa1_fn = nn.LayerNorm(d_attn)

        self.sa2 = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa2_n = nn.LayerNorm(d_attn)
        self.sa2_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa2_fn = nn.LayerNorm(d_attn)

        self.ca = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
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
        mg = x[:, :mg_dim]

        if self.n_extra > 0:
            extra = x[:, mg_dim:mg_dim + self.n_extra]
            m2v = x[:, mg_dim + self.n_extra:]
        else:
            extra = None
            m2v = x[:, mg_dim:]

        tok = self.tok_proj(mg.view(B, self.n_props, self.stat_dim))
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
# CELL 4: Featurizer (same as trm13.py ExpandedFeaturizer)
# ══════════════════════════════════════════════════════════════════════

class ExpandedFeaturizer:
    """Magpie (22 props × 6 stats) + Extra matminer descriptors + Mat2Vec (200d)."""
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
        self.n_extra = None

        self.scaler = None
        os.makedirs(cache, exist_ok=True)
        for f in self.FILES:
            p = os.path.join(cache, f)
            if not os.path.exists(p):
                print(f"  Downloading {f}...")
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
                print(f"Expanded features: {self.n_mg} Magpie + "
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


# ══════════════════════════════════════════════════════════════════════
# CELL 5: Generate results.json.gz using the Matbench API
# ══════════════════════════════════════════════════════════════════════

from matbench.bench import MatbenchBenchmark
from sklearn.model_selection import KFold

SEEDS = [42, 123, 7, 0, 99]
MODEL_DIR = "trm_models_v13"  # <-- CHANGE THIS if your folder is named differently
CNAME = "V13A-2xSA-StdDS"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Load the dataset through matminer (same as training)
print("Loading matbench_steels dataset...")
from matminer.datasets import load_dataset
df = load_dataset("matbench_steels")
comps_raw = df['composition'].tolist()
targets_all = np.array(df['yield strength'].tolist(), np.float32)
comps_all = [Composition(c) for c in comps_raw]

# 2. Compute features (SAME featurizer as training)
print("Computing features...")
feat = ExpandedFeaturizer(cache="mat2vec_cache")
X_all_raw = feat.featurize_all(comps_all)
n_extra = feat.n_extra
print(f"Feature shape: {X_all_raw.shape}, n_extra: {n_extra}")

# 3. Recreate the EXACT same 5-fold split used during training
#    (KFold with random_state=18012019 is the matbench default)
kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
folds = list(kfold.split(comps_all))

# 4. Model config (must match training exactly)
model_kw = dict(n_props=22, stat_dim=6, n_extra=n_extra,
                mat2vec_dim=200, d_attn=64, nhead=4,
                d_hidden=96, ff_dim=150, dropout=0.2, max_steps=20)

# 5. Initialize Matbench and record predictions
mb = MatbenchBenchmark(autoload=False, subset=["matbench_steels"])
task = list(mb.tasks)[0]
task.load()

fold_maes = []
for fold_idx, (tv_i, te_i) in enumerate(folds):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx + 1}/5 — {len(te_i)} test samples")
    print(f"{'='*60}")

    seed_preds = []
    for seed in SEEDS:
        # Recreate exact scaler from training
        tri, vli = strat_split(targets_all[tv_i], 0.15, seed + fold_idx)
        feat.fit_scaler(X_all_raw[tv_i][tri])
        te_scaled = feat.transform(X_all_raw[te_i])

        # Load model
        ckpt_path = os.path.join(MODEL_DIR, f"{CNAME}_seed{seed}_fold{fold_idx+1}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  MISSING: {ckpt_path}")
            continue

        model = DeepHybridTRM(**model_kw).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state['model_state'])
        model.eval()

        # Run inference
        with torch.no_grad():
            te_tensor = torch.tensor(te_scaled, dtype=torch.float32).to(device)
            pred = model(te_tensor).cpu().numpy()
        seed_preds.append(pred)

        saved_mae = state.get('test_mae', '?')
        print(f"  Seed {seed:>3}: loaded ({saved_mae:.2f} MAE saved)")
        del model

    if not seed_preds:
        print(f"  ❌ No checkpoints found for fold {fold_idx+1}!")
        continue

    # Average across seeds
    ensemble_pred = np.mean(seed_preds, axis=0)

    # Compute MAE for verification
    true_vals = targets_all[te_i]
    mae = np.mean(np.abs(ensemble_pred - true_vals))
    fold_maes.append(mae)
    print(f"\n  ✅ Fold {fold_idx+1} Ensemble MAE: {mae:.2f} MPa "
          f"({len(seed_preds)} seeds averaged)")

    # Record with Matbench API
    task.record(fold_idx, ensemble_pred)

# 6. Print summary
avg_mae = np.mean(fold_maes)
print(f"\n{'='*60}")
print(f"  GRAND AVERAGE MAE: {avg_mae:.2f} MPa")
print(f"  Per-fold: {[f'{m:.2f}' for m in fold_maes]}")
print(f"{'='*60}")

# 7. Save results.json.gz
mb.to_file("results.json.gz")
print(f"\n✅ results.json.gz saved! File size: {os.path.getsize('results.json.gz')} bytes")
print("Download this file and include it in your Matbench PR.")


# ══════════════════════════════════════════════════════════════════════
# CELL 6: Download the file (Colab-specific)
# ══════════════════════════════════════════════════════════════════════
# from google.colab import files
# files.download("results.json.gz")
