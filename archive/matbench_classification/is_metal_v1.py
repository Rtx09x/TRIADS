"""
+=============================================================+
|  TRIADS on matbench_expt_is_metal — Classification          |
|  Metal vs Non-metal (ROCAUC) — 4,921 samples                |
|                                                             |
|  Composition features (354d, same as expt_gap V2)           |
|  75K model | BCEWithLogitsLoss | 5-Seed Ensemble            |
|  Seeds: [42, 123, 456, 789, 1024]                           |
|  Target: Kaggle P100                                        |
+=============================================================+
"""

import os, copy, json, time, logging, warnings, urllib.request, shutil
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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
log = logging.getLogger("TRIADS-is_metal")

BATCH_SIZE = 64
SEEDS = [42, 123, 456, 789, 1024]

# 75K config
MODEL_CFG = dict(
    d_attn=32, nhead=4, d_hidden=64, ff_dim=96,
    dropout=0.15, max_steps=16,
)


# ======================================================================
# FAST TENSOR DATALOADER
# ======================================================================

class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=64, shuffle=False):
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
# FEATURIZER — Composition features (354d, same as expt_gap V2)
# ======================================================================

_ORBITAL_ENERGIES = {
    'H':  {'1s': -13.6}, 'He': {'1s': -24.6},
    'Li': {'2s': -5.4}, 'Be': {'2s': -9.3},
    'B':  {'2s': -14.0, '2p': -8.3}, 'C':  {'2s': -19.4, '2p': -11.3},
    'N':  {'2s': -25.6, '2p': -14.5}, 'O':  {'2s': -32.4, '2p': -13.6},
    'F':  {'2s': -40.2, '2p': -17.4}, 'Ne': {'2s': -48.5, '2p': -21.6},
    'Na': {'3s': -5.1}, 'Mg': {'3s': -7.6},
    'Al': {'3s': -11.3, '3p': -6.0}, 'Si': {'3s': -15.0, '3p': -8.2},
    'P':  {'3s': -18.7, '3p': -10.5}, 'S':  {'3s': -22.7, '3p': -10.4},
    'Cl': {'3s': -25.3, '3p': -13.0}, 'Ar': {'3s': -29.2, '3p': -15.8},
    'K':  {'4s': -4.3}, 'Ca': {'4s': -6.1},
    'Sc': {'4s': -6.6, '3d': -8.0}, 'Ti': {'4s': -6.8, '3d': -8.5},
    'V':  {'4s': -6.7, '3d': -8.3}, 'Cr': {'4s': -6.8, '3d': -8.7},
    'Mn': {'4s': -7.4, '3d': -9.5}, 'Fe': {'4s': -7.9, '3d': -10.0},
    'Co': {'4s': -7.9, '3d': -10.0}, 'Ni': {'4s': -7.6, '3d': -10.0},
    'Cu': {'4s': -7.7, '3d': -11.7}, 'Zn': {'4s': -9.4, '3d': -17.3},
    'Ga': {'4s': -12.6, '4p': -6.0}, 'Ge': {'4s': -15.6, '4p': -7.9},
    'As': {'4s': -18.6, '4p': -9.8}, 'Se': {'4s': -21.1, '4p': -9.8},
    'Br': {'4s': -24.0, '4p': -11.8}, 'Kr': {'4s': -27.5, '4p': -14.0},
    'Rb': {'5s': -4.2}, 'Sr': {'5s': -5.7},
    'Y':  {'5s': -6.5, '4d': -7.4}, 'Zr': {'5s': -6.8, '4d': -8.3},
    'Nb': {'5s': -6.9, '4d': -8.5}, 'Mo': {'5s': -7.1, '4d': -8.9},
    'Ru': {'5s': -7.4, '4d': -8.7}, 'Rh': {'5s': -7.5, '4d': -8.8},
    'Pd': {'4d': -8.3}, 'Ag': {'5s': -7.6, '4d': -12.3},
    'Cd': {'5s': -9.0, '4d': -16.7}, 'In': {'5s': -12.0, '5p': -5.8},
    'Sn': {'5s': -14.6, '5p': -7.3}, 'Sb': {'5s': -16.5, '5p': -8.6},
    'Te': {'5s': -19.0, '5p': -9.0}, 'I':  {'5s': -21.1, '5p': -10.5},
    'Xe': {'5s': -23.4, '5p': -12.1}, 'Cs': {'6s': -3.9}, 'Ba': {'6s': -5.2},
    'La': {'6s': -5.6, '5d': -7.5},
    'Ce': {'6s': -5.5, '5d': -7.3, '4f': -7.0},
    'Hf': {'6s': -7.0, '5d': -8.1}, 'Ta': {'6s': -7.9, '5d': -9.6},
    'W':  {'6s': -8.0, '5d': -9.8}, 'Re': {'6s': -7.9, '5d': -9.2},
    'Os': {'6s': -8.4, '5d': -10.0}, 'Ir': {'6s': -9.1, '5d': -10.7},
    'Pt': {'6s': -9.0, '5d': -10.5}, 'Au': {'6s': -9.2, '5d': -12.8},
    'Pb': {'6s': -15.0, '6p': -7.4}, 'Bi': {'6s': -16.7, '6p': -7.3},
}


def _compute_homo_lumo_gap(comp):
    elements = comp.get_el_amt_dict()
    highest_occ = []
    all_energies = []
    for el, frac in elements.items():
        if el not in _ORBITAL_ENERGIES:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        orbs = _ORBITAL_ENERGIES[el]
        highest_occ.append((max(orbs.values()), frac))
        all_energies.extend(orbs.values())
    if not highest_occ:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    homo = sum(e * f for e, f in highest_occ) / sum(f for _, f in highest_occ)
    above = [e for e in all_energies if e > homo]
    lumo = min(above) if above else homo + 1.0
    return np.array([homo, lumo, lumo - homo], dtype=np.float32)


class MetallicityFeaturizer:
    """
    Composition featurizer for metal/non-metal classification.
    Magpie (132) + Extras (22) + Mat2Vec (200) = 354d

    HOMO/LUMO gap is KEPT here — it's directly relevant to metallicity
    (metals have zero band gap, non-metals have nonzero gap).
    """
    GCS = "https://storage.googleapis.com/mat2vec/"
    FILES = ["pretrained_embeddings",
             "pretrained_embeddings.wv.vectors.npy",
             "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache="mat2vec_cache"):
        from matminer.featurizers.composition import (
            Stoichiometry, ValenceOrbital, IonProperty, BandCenter
        )
        from matminer.featurizers.composition.element import TMetalFraction

        self.ep_magpie = ElementProperty.from_preset("magpie")
        self.n_mg = len(self.ep_magpie.feature_labels())

        self.extra_featurizers = [
            ("Stoichiometry",  Stoichiometry()),
            ("ValenceOrbital", ValenceOrbital()),
            ("IonProperty",    IonProperty()),
            ("BandCenter",     BandCenter()),     # relevant: band center → metallicity
            ("TMetalFraction", TMetalFraction()),  # directly relevant
        ]

        self._extra_sizes = {}
        for name, ftzr in self.extra_featurizers:
            try: self._extra_sizes[name] = len(ftzr.feature_labels())
            except: self._extra_sizes[name] = None

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

    def _featurize_extra(self, comp):
        parts = []
        for name, ftzr in self.extra_featurizers:
            try:
                vals = np.array(ftzr.featurize(comp), np.float32)
                parts.append(np.nan_to_num(vals, nan=0.0))
                if self._extra_sizes.get(name) is None:
                    self._extra_sizes[name] = len(vals)
            except:
                sz = self._extra_sizes.get(name, 0) or 1
                parts.append(np.zeros(sz, np.float32))
        # HOMO/LUMO gap — directly relevant to metallicity
        parts.append(_compute_homo_lumo_gap(comp))
        return np.concatenate(parts)

    def featurize_all(self, comps):
        out = []
        test_ex = self._featurize_extra(comps[0])
        self.n_extra = len(test_ex)
        total = self.n_mg + self.n_extra + 200
        log.info(f"Features: {self.n_mg} Magpie + "
                 f"{self.n_extra} Extra + 200 Mat2Vec = {total}d")
        for c in tqdm(comps, desc="  Featurizing", leave=False):
            try: mg = np.array(self.ep_magpie.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)
            ex = self._featurize_extra(c)
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
# MODEL — DeepHybridTRM (identical architecture, 1 logit output)
# ======================================================================

class DeepHybridTRM(nn.Module):
    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=32, nhead=4, d_hidden=64, ff_dim=96,
                 dropout=0.15, max_steps=16, **kw):
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
        self.head = nn.Linear(d_hidden, 1)  # outputs raw logit
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

    def forward(self, x, deep_supervision=False):
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        step_preds = []
        for s in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            step_preds.append(self.head(y).squeeze(1))  # raw logits
        return step_preds if deep_supervision else step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# LOSS (BCE) + UTILS
# ======================================================================

def deep_supervision_loss_bce(step_preds, targets):
    """BCE deep supervision: weighted BCE at each reasoning step."""
    preds = torch.stack(step_preds)  # (steps, B)
    n = preds.shape[0]
    w = torch.arange(1, n + 1, device=preds.device, dtype=preds.dtype)
    w = w / w.sum()
    per_step = torch.stack([
        F.binary_cross_entropy_with_logits(preds[i], targets, reduction='mean')
        for i in range(n)
    ])
    return (w * per_step).sum()


def strat_split_cls(targets, val_size=0.15, seed=42):
    """Stratified split for binary classification."""
    tr, vl = [], []
    rng = np.random.RandomState(seed)
    for cls in [0, 1]:
        m = np.where(targets == cls)[0]
        n = max(1, int(len(m) * val_size))
        c = rng.choice(m, n, replace=False)
        vl.extend(c.tolist()); tr.extend(np.setdiff1d(m, c).tolist())
    return np.array(tr), np.array(vl)


@torch.inference_mode()
def predict_proba(model, dl):
    """Returns probabilities (sigmoid of logits)."""
    model.eval()
    preds = []
    for bx, _ in dl:
        logits = model(bx)
        preds.append(torch.sigmoid(logits).cpu())
    return torch.cat(preds)


# ======================================================================
# TRAINING — BCE loss, ROCAUC metric
# ======================================================================

def train_fold(model, tr_dl, vl_dl, device,
               epochs=300, swa_start=200, fold=1, seed=42):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=swa_start, eta_min=1e-4)
    swa_m = AveragedModel(model)
    swa_s = SWALR(opt, swa_lr=5e-4)
    swa_on = False
    best_v, best_w = float('-inf'), None  # maximize ROCAUC

    pbar = tqdm(range(epochs), desc=f"  [75K|s{seed}] F{fold}/5",
                leave=False, ncols=120)
    for ep in pbar:
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for bx, by in tr_dl:
            sp = model(bx, deep_supervision=True)
            loss = deep_supervision_loss_bce(sp, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation ROCAUC
        model.eval()
        val_probs, val_targets = [], []
        with torch.inference_mode():
            for bx, by in vl_dl:
                logits = model(bx)
                val_probs.append(torch.sigmoid(logits).cpu())
                val_targets.append(by.cpu())

        vp = torch.cat(val_probs).numpy()
        vt = torch.cat(val_targets).numpy()
        try:
            val_auc = roc_auc_score(vt, vp)
        except:
            val_auc = 0.5

        if ep < swa_start:
            sch.step()
            if val_auc > best_v:  # maximize
                best_v = val_auc
                best_w = copy.deepcopy(model.state_dict())
        else:
            if not swa_on: swa_on = True
            swa_m.update_parameters(model); swa_s.step()

        if ep % 10 == 0 or ep == epochs - 1:
            pbar.set_postfix(Best=f'{best_v:.4f}', Ph='SWA' if swa_on else 'COS',
                            Loss=f'{epoch_loss/n_batches:.4f}', AUC=f'{val_auc:.4f}')

    if swa_on:
        update_bn(tr_dl, swa_m, device=device)
        model.load_state_dict(swa_m.module.state_dict())
    else:
        model.load_state_dict(best_w)
    return best_v, model


# ======================================================================
# MAIN — 5-SEED ENSEMBLE
# ======================================================================

def run_benchmark():
    t0 = time.time()

    print(f"""
  +==========================================================+
  |  TRIADS — matbench_expt_is_metal (5-Seed Ensemble)       |
  |  Metal vs Non-metal Classification (ROCAUC)              |
  |  75K model | BCEWithLogitsLoss | dropout=0.15            |
  |  Seeds: {SEEDS}                       |
  +==========================================================+
    """)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gm = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({gm:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ── LOAD DATASET ──────────────────────────────────────────────────
    print("\n  Loading matbench_expt_is_metal...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_expt_is_metal")

    # Target: boolean → float (0.0 or 1.0)
    targets_all = np.array(df['is_metal'].astype(float).tolist(), np.float32)

    # Composition
    if 'composition' in df.columns:
        comps_all = [Composition(c) for c in df['composition'].tolist()]
    elif 'structure' in df.columns:
        comps_all = [s.composition for s in df['structure'].tolist()]
    else:
        comps_all = [Composition(str(f)) for f in df['formula'].tolist()]

    n_pos = int(targets_all.sum())
    n_neg = len(targets_all) - n_pos
    print(f"  Dataset: {len(comps_all)} samples ({n_pos} metals, {n_neg} non-metals)")
    print(f"  Class balance: {n_pos/len(targets_all)*100:.1f}% metal")

    # ── FEATURIZE (once) ─────────────────────────────────────────────
    t_feat = time.time()
    feat = MetallicityFeaturizer()
    X_all = feat.featurize_all(comps_all)
    n_extra = feat.n_extra
    print(f"  Features: {X_all.shape} (n_extra={n_extra})")
    print(f"  Featurization: {time.time()-t_feat:.1f}s")

    # ── FOLDS ────────────────────────────────────────────────────────
    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    for fi, (tv, te) in enumerate(folds):
        assert len(set(tv) & set(te)) == 0
    print("  5 folds verified: zero leakage\n")

    # ── MODEL INFO ───────────────────────────────────────────────────
    model_kw = dict(n_props=22, stat_dim=6, n_extra=n_extra,
                    mat2vec_dim=200, **MODEL_CFG)
    test_model = DeepHybridTRM(**model_kw)
    n_params = test_model.count_parameters()
    del test_model
    print(f"  Model: {n_params:,} params")

    # ── TRAIN ALL SEEDS ──────────────────────────────────────────────
    model_dir = 'is_metal_models'
    os.makedirs(model_dir, exist_ok=True)

    all_seed_aucs = {}
    all_fold_probs = {}
    all_fold_targets = {}

    for seed in SEEDS:
        print(f"\n  {'─'*3} Seed {seed} {'─'*40}")
        t_seed = time.time()
        seed_aucs = {}

        for fi, (tv_i, te_i) in enumerate(folds):
            tri, vli = strat_split_cls(targets_all[tv_i], 0.15, seed + fi)
            feat.fit_scaler(X_all[tv_i][tri])

            tr_x = torch.tensor(feat.transform(X_all[tv_i][tri]), dtype=torch.float32).to(device)
            tr_y = torch.tensor(targets_all[tv_i][tri], dtype=torch.float32).to(device)
            vl_x = torch.tensor(feat.transform(X_all[tv_i][vli]), dtype=torch.float32).to(device)
            vl_y = torch.tensor(targets_all[tv_i][vli], dtype=torch.float32).to(device)
            te_x = torch.tensor(feat.transform(X_all[te_i]), dtype=torch.float32).to(device)
            te_y = torch.tensor(targets_all[te_i], dtype=torch.float32).to(device)

            tr_dl = FastTensorDataLoader(tr_x, tr_y, batch_size=BATCH_SIZE, shuffle=True)
            vl_dl = FastTensorDataLoader(vl_x, vl_y, batch_size=BATCH_SIZE, shuffle=False)
            te_dl = FastTensorDataLoader(te_x, te_y, batch_size=BATCH_SIZE, shuffle=False)

            torch.manual_seed(seed + fi)
            np.random.seed(seed + fi)
            if device.type == 'cuda': torch.cuda.manual_seed(seed + fi)

            model = DeepHybridTRM(**model_kw).to(device)
            bv, model = train_fold(model, tr_dl, vl_dl, device,
                                    epochs=300, swa_start=200,
                                    fold=fi+1, seed=seed)

            probs = predict_proba(model, te_dl)
            auc = roc_auc_score(te_y.cpu().numpy(), probs.numpy())
            seed_aucs[fi] = auc

            if fi not in all_fold_probs:
                all_fold_probs[fi] = {}
                all_fold_targets[fi] = te_y.cpu()
            all_fold_probs[fi][seed] = probs

            torch.save({
                'model_state': model.state_dict(),
                'test_auc': auc, 'fold': fi+1, 'seed': seed,
                'n_extra': n_extra,
            }, f'{model_dir}/is_metal_75K_s{seed}_f{fi+1}.pt')

            del model, tr_x, tr_y, vl_x, vl_y, te_x, te_y
            if device.type == 'cuda': torch.cuda.empty_cache()

        avg_s = np.mean(list(seed_aucs.values()))
        all_seed_aucs[seed] = seed_aucs
        dt = time.time() - t_seed
        print(f"\n  Seed {seed}: avg={avg_s:.4f} | "
              f"{[f'{seed_aucs[i]:.4f}' for i in range(5)]} ({dt:.0f}s)")

    # ── ENSEMBLE ─────────────────────────────────────────────────────
    ens_aucs = {}
    for fi in range(5):
        probs_stack = torch.stack([all_fold_probs[fi][s] for s in SEEDS])
        ens_prob = probs_stack.mean(dim=0)
        ens_aucs[fi] = roc_auc_score(
            all_fold_targets[fi].numpy(), ens_prob.numpy())

    single_avgs = [np.mean(list(all_seed_aucs[s].values())) for s in SEEDS]
    single_mean = np.mean(single_avgs)
    single_std = np.std(single_avgs)
    ens_mean = np.mean(list(ens_aucs.values()))
    ens_std = np.std(list(ens_aucs.values()))

    # ── RESULTS ──────────────────────────────────────────────────────
    tt = time.time() - t0

    print(f"""
{'='*72}
  FINAL RESULTS — TRIADS on matbench_expt_is_metal (ROCAUC)
{'='*72}

  Per-seed results:""")

    for seed in SEEDS:
        sm = all_seed_aucs[seed]
        avg_s = np.mean(list(sm.values()))
        print(f"    Seed {seed:>4}: {avg_s:.4f} | "
              f"{[f'{sm[i]:.4f}' for i in range(5)]}")

    print(f"""
    Single-seed avg: {single_mean:.4f} ± {single_std:.4f}
    5-Seed Ensemble: {ens_mean:.4f} ± {ens_std:.4f}
    Per-fold ens:    {[f'{ens_aucs[i]:.4f}' for i in range(5)]}

  {'Model':<40} {'ROCAUC':>10}
  {'─'*53}
  {'AMMExpress v2020':<40} {'0.9209':>10}
  {'TRIADS (75K, 5-seed ens)':<40} {f'{ens_mean:.4f}':>10} ← US
  {'─'*53}

  Total time: {tt/60:.1f} min
  Saved: {model_dir}/
""")

    summary = {
        'version': 'is_metal-V1-ensemble',
        'dataset': 'matbench_expt_is_metal',
        'task': 'classification',
        'metric': 'ROCAUC',
        'samples': len(comps_all),
        'class_balance': f'{n_pos} metals / {n_neg} non-metals',
        'model_config': MODEL_CFG,
        'params': n_params,
        'seeds': SEEDS,
        'per_seed': {str(s): {str(k): round(v, 4) for k, v in m.items()}
                     for s, m in all_seed_aucs.items()},
        'single_seed_avg': round(single_mean, 4),
        'single_seed_std': round(single_std, 4),
        'ensemble_aucs': {str(k): round(v, 4) for k, v in ens_aucs.items()},
        'ensemble_avg': round(ens_mean, 4),
        'ensemble_std': round(ens_std, 4),
        'total_time_min': round(tt/60, 1),
    }
    with open('is_metal_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Saved: is_metal_summary.json")

    shutil.make_archive(model_dir, 'zip', '.', model_dir)
    print(f"  Saved: {model_dir}.zip")


if __name__ == '__main__':
    run_benchmark()
