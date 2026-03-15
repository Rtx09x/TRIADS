"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V5 — Quick Wins + Novel Architecture                     ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV           ║
║                                                                      ║
║  V5A: MLP-TRM + SWA + Recursion Step Ensemble (~67K params)          ║
║       Same MLP architecture but smarter training & inference          ║
║                                                                      ║
║  V5B: Feature-Group Dual-Reference TRM (NOVEL) (~28K params)         ║
║       22 Magpie property tokens × 6 stats each                       ║
║       Dual-reference: E0 (fixed) + Et (evolved) cross-attention      ║
║       z initialized from Mat2Vec embeddings                          ║
║       The user's novel recursive reasoning architecture              ║
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
log = logging.getLogger("TRM5")

SEED = 42

BASELINES = {
    'TPOT-Mat (best)':    79.9468,
    'AutoML-Mat':         82.3043,
    'MODNet v0.1.12':     87.7627,
    'RF-SCM/Magpie':     103.5125,
    'CrabNet':           107.3160,
    'MLP-Magpie-L V3':   130.3264,
    'MLP-Combined-S V4': 131.6265,
    'MLP-TRM V2 best':   184.5662,
    'Dummy':             229.7445,
}

ENSEMBLE_LAST_K = 4  # Average last 4 recursion steps at inference


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMBINED FEATURIZER (Magpie + Mat2Vec) — same as V4
# ══════════════════════════════════════════════════════════════════════════════

class CombinedFeaturizer:
    """Magpie (132) + Mat2Vec pooled (200) = 332-dim features."""

    GCS_BASE = "https://storage.googleapis.com/mat2vec/"
    M2V_FILES = ["pretrained_embeddings",
                 "pretrained_embeddings.wv.vectors.npy",
                 "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache_dir="mat2vec_cache"):
        self.ep = ElementProperty.from_preset("magpie")
        self.magpie_names = self.ep.feature_labels()
        self.n_magpie = len(self.magpie_names)
        self.scaler = None

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.m2v = self._download_and_load_m2v()
        self.m2v_embeddings = {w: self.m2v.wv[w] for w in self.m2v.wv.index_to_key}

        self.n_features = self.n_magpie + 200
        log.info(f"Combined featurizer: {self.n_magpie} Magpie + 200 Mat2Vec "
                 f"= {self.n_features} features")

    def _download_and_load_m2v(self):
        for fname in self.M2V_FILES:
            fpath = os.path.join(self.cache_dir, fname)
            if not os.path.exists(fpath):
                log.info(f"  Downloading {fname}...")
                urllib.request.urlretrieve(self.GCS_BASE + fname, fpath)
        return Word2Vec.load(os.path.join(self.cache_dir, "pretrained_embeddings"))

    def _mat2vec_pooled(self, comp):
        vec, total = np.zeros(200, dtype=np.float32), 0.0
        for sym, frac in comp.get_el_amt_dict().items():
            if sym in self.m2v_embeddings:
                vec += frac * self.m2v_embeddings[sym]
                total += frac
        return vec / max(total, 1e-8)

    def featurize_all(self, comps):
        features = []
        for comp in tqdm(comps, desc="  Combined featurization", leave=False):
            try:
                magpie = np.array(self.ep.featurize(comp), dtype=np.float32)
            except Exception:
                magpie = np.zeros(self.n_magpie, dtype=np.float32)
            magpie = np.nan_to_num(magpie, nan=0.0)
            m2v = self._mat2vec_pooled(comp)
            features.append(np.concatenate([magpie, m2v]))
        return np.array(features)

    def fit_scaler(self, features):
        self.scaler = StandardScaler().fit(features)

    def transform(self, features):
        if self.scaler is None:
            return features
        return np.nan_to_num(
            self.scaler.transform(features), nan=0.0
        ).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SteelsDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets  = torch.tensor(np.array(targets, dtype=np.float32))
    def __len__(self):        return len(self.targets)
    def __getitem__(self, i): return self.features[i], self.targets[i]


# ══════════════════════════════════════════════════════════════════════════════
# 3. V5A — MLP-TRM WITH RECURSION ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

class MLPTRM(nn.Module):
    """MLP-TRM with recursion step ensemble at inference.
    At inference, averages predictions from the last K recursion steps
    instead of only using step 16. Free diversity, no extra cost.
    """
    def __init__(self, input_dim=332, hidden_dim=64, ff_dim=100,
                 dropout=0.2, steps=16):
        super().__init__()
        self.steps = steps
        self.D     = hidden_dim
        self.arch_type = 'MLP-TRM'

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU())

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

    def forward(self, x, return_trajectory=False, ensemble_last_k=0):
        B  = x.size(0)
        xp = self.input_proj(x)
        z  = torch.zeros(B, self.D, device=x.device)
        y  = torch.zeros(B, self.D, device=x.device)
        traj = []
        for _ in range(self.steps):
            z = z + self.z_update(torch.cat([xp, y, z], -1))
            y = y + self.y_update(torch.cat([y, z],     -1))
            traj.append(self.head(y).squeeze(1))

        # Recursion step ensemble: average last K steps at inference
        if ensemble_last_k > 0 and not self.training:
            out = torch.mean(torch.stack(traj[-ensemble_last_k:]), dim=0)
        else:
            out = traj[-1]

        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 4. V5B — FEATURE-GROUP DUAL-REFERENCE TRM (NOVEL ARCHITECTURE)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureGroupTRM(nn.Module):
    """
    Novel dual-reference recursive Transformer over structured feature groups.

    Architecture:
      Input: 132 Magpie features → 22 property tokens × 6 stats each
             200 Mat2Vec features → initializes reasoning state z

      E0 = projected property tokens (FIXED — raw data reference)
      Et = evolving property tokens (updated by self-attention each step)
      z  = reasoning state (initialized from Mat2Vec, cross-attends to E0 and Et)
      y  = output state (updated from z each step)

    The dual-reference mechanism:
      - E0 anchors reasoning to what the raw data shows
      - Et captures what attention has discovered about property interactions
      - z queries both to combine raw evidence with evolved understanding
    """
    def __init__(self, n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_model=32, nhead=2, ff_dim=64, dropout=0.2, steps=16,
                 # these are ignored but allow uniform config interface
                 input_dim=None, hidden_dim=None):
        super().__init__()
        self.steps = steps
        self.n_props = n_props
        self.stat_dim = stat_dim
        self.d_model = d_model
        self.D = d_model  # for compatibility
        self.arch_type = 'FeatureGroup-Novel'

        # ── Token projection: 6-dim stats → d_model ──────────────────────
        self.token_proj = nn.Sequential(
            nn.Linear(stat_dim, d_model), nn.LayerNorm(d_model), nn.GELU())

        # ── z initialization from Mat2Vec ─────────────────────────────────
        self.z_init = nn.Sequential(
            nn.Linear(mat2vec_dim, d_model), nn.LayerNorm(d_model), nn.GELU())

        # ── Et evolution: self-attention over property tokens ─────────────
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model))
        self.sa_ff_norm = nn.LayerNorm(d_model)

        # ── Cross-attention: z → E0 (fixed reference) ────────────────────
        self.cross_e0 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ce0_norm = nn.LayerNorm(d_model)

        # ── Cross-attention: z → Et (evolved reference) ──────────────────
        self.cross_et = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.cet_norm = nn.LayerNorm(d_model)

        # ── z reasoning update ────────────────────────────────────────────
        self.z_ff = nn.Sequential(
            nn.Linear(d_model * 3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.LayerNorm(d_model))

        # ── y output update ───────────────────────────────────────────────
        self.y_ff = nn.Sequential(
            nn.Linear(d_model * 2, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.LayerNorm(d_model))

        self.head = nn.Linear(d_model, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_trajectory=False, ensemble_last_k=0):
        B = x.size(0)

        # ── Split input: Magpie [132] + Mat2Vec [200] ─────────────────────
        magpie  = x[:, :self.n_props * self.stat_dim]   # [B, 132]
        mat2vec = x[:, self.n_props * self.stat_dim:]    # [B, 200]

        # ── Create structured property tokens ─────────────────────────────
        tokens = magpie.view(B, self.n_props, self.stat_dim)  # [B, 22, 6]
        E0 = self.token_proj(tokens)    # [B, 22, d_model] — FIXED reference
        Et = E0.clone()                 # [B, 22, d_model] — will evolve

        # ── Initialize z from Mat2Vec (chemical knowledge seed) ───────────
        z = self.z_init(mat2vec).unsqueeze(1)  # [B, 1, d_model]
        y = torch.zeros(B, self.d_model, device=x.device)

        traj = []
        for _ in range(self.steps):
            # 1. EVOLVE Et: self-attention discovers property interactions
            et_attn = self.self_attn(Et, Et, Et)[0]
            Et = self.sa_norm(Et + et_attn)
            et_ff = self.sa_ff(Et)
            Et = self.sa_ff_norm(Et + et_ff)

            # 2. z QUERIES E0: "what does the raw data say?"
            z_e0 = self.cross_e0(z, E0, E0)[0]  # [B, 1, d_model]
            z_e0 = self.ce0_norm(z + z_e0)

            # 3. z QUERIES Et: "what has reasoning discovered?"
            z_et = self.cross_et(z, Et, Et)[0]  # [B, 1, d_model]
            z_et = self.cet_norm(z + z_et)

            # 4. UPDATE z: combine raw evidence + evolved understanding
            z_cat = torch.cat([
                z.squeeze(1), z_e0.squeeze(1), z_et.squeeze(1)
            ], dim=-1)  # [B, d_model*3]
            z = z + self.z_ff(z_cat).unsqueeze(1)

            # 5. UPDATE y: accumulate prediction signal
            y = y + self.y_ff(torch.cat([y, z.squeeze(1)], dim=-1))

            traj.append(self.head(y).squeeze(1))

        # Recursion step ensemble at inference
        if ensemble_last_k > 0 and not self.training:
            out = torch.mean(torch.stack(traj[-ensemble_last_k:]), dim=0)
        else:
            out = traj[-1]

        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 5. STRATIFIED VAL SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def stratified_val_split(targets, val_size=0.15, seed=42):
    bins   = np.percentile(targets, [25, 50, 75])
    labels = np.digitize(targets, bins)
    train_idx, val_idx = [], []
    rng = np.random.RandomState(seed)
    for b in range(4):
        bin_mask = np.where(labels == b)[0]
        if len(bin_mask) == 0:
            continue
        n_val  = max(1, int(len(bin_mask) * val_size))
        chosen = rng.choice(bin_mask, n_val, replace=False)
        val_idx.extend(chosen.tolist())
        train_idx.extend(np.setdiff1d(bin_mask, chosen).tolist())
    return np.array(train_idx), np.array(val_idx)


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP WITH SWA
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(model, train_loader, val_loader, device,
               epochs=300, swa_start=200, fold_idx=1, config_name=""):
    """Train with cosine LR for swa_start epochs, then SWA for remaining.

    Phase 1 (epochs 0–swa_start): Normal cosine annealing + early stopping backup
    Phase 2 (epochs swa_start–end): SWA weight averaging at constant LR
    Final model: SWA-averaged weights (smoother, better generalization)
    """
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=swa_start, eta_min=1e-4)

    # SWA setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-4)
    swa_started = False

    best_val  = float('inf')
    best_wts  = copy.deepcopy(model.state_dict())
    history   = {'train': [], 'val': []}

    pbar = tqdm(range(epochs), desc=f"  [{config_name}] F{fold_idx}/5",
                leave=False, ncols=120)

    for epoch in pbar:
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            preds = model(bx)
            loss  = F.l1_loss(preds, by)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(by)
        tr_loss /= len(train_loader.dataset)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                preds  = model(bx, ensemble_last_k=ENSEMBLE_LAST_K)
                vl_loss += F.l1_loss(preds, by).item() * len(by)
        vl_loss /= len(val_loader.dataset)

        history['train'].append(tr_loss)
        history['val'].append(vl_loss)

        # ── LR scheduling + SWA ───────────────────────────────────────
        if epoch < swa_start:
            scheduler.step()
            if vl_loss < best_val:
                best_val = vl_loss
                best_wts = copy.deepcopy(model.state_dict())
        else:
            if not swa_started:
                log.info(f"  SWA started at epoch {epoch+1}")
                swa_started = True
            swa_model.update_parameters(model)
            swa_scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        phase = 'SWA' if epoch >= swa_start else 'COS'
        pbar.set_postfix({
            'Tr': f'{tr_loss:.1f}', 'Val': f'{vl_loss:.1f}',
            'Best': f'{best_val:.1f}', 'Ph': phase,
            'LR': f'{lr:.2e}'
        })

    # ── Finalize SWA model ────────────────────────────────────────────
    if swa_started:
        update_bn(train_loader, swa_model, device=device)
        # Copy SWA-averaged weights back to model
        model.load_state_dict(swa_model.module.state_dict())
        log.info(f"  SWA finalized (averaged {epochs - swa_start} checkpoints)")
    else:
        model.load_state_dict(best_wts)

    return best_val, model, history


# ══════════════════════════════════════════════════════════════════════════════
# 7. RECURSION TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

def eval_trajectory(model, loader, device):
    model.eval()
    step_preds, targets = None, []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            _, traj = model(bx, return_trajectory=True)
            if step_preds is None:
                step_preds = [[] for _ in range(len(traj))]
            for i, sp in enumerate(traj):
                step_preds[i].extend(sp.cpu().numpy().tolist())
            targets.extend(by.cpu().numpy().tolist())
    t = np.array(targets)
    return [float(np.mean(np.abs(np.array(p) - t))) for p in step_preds]


def eval_ensemble(model, loader, device, ensemble_k=4):
    """Evaluate with recursion step ensemble."""
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx, ensemble_last_k=ensemble_k)
            total_loss += F.l1_loss(preds, by).item() * len(by)
            n += len(by)
    return total_loss / n


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    t0 = time.time()
    print("\n" + "═"*72)
    print("  TRM-MatSci V5 │ Quick Wins + Novel Architecture │ matbench_steels")
    print("  V5A: MLP-TRM + SWA + Recursion Ensemble")
    print("  V5B: Feature-Group Dual-Reference TRM (NOVEL)")
    print("═"*72 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        log.info("Device: CPU")

    # ── Load dataset ──────────────────────────────────────────────────────
    log.info("Loading matbench_steels...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_steels")
    comps_raw   = df['composition'].tolist()
    targets_all = np.array(df['yield strength'].tolist(), dtype=np.float32)
    comps_all   = [Composition(c) for c in comps_raw]
    log.info(f"Dataset: {len(df)} samples")

    # ── Precompute features ───────────────────────────────────────────────
    log.info("Computing combined Magpie + Mat2Vec features...")
    feat = CombinedFeaturizer()
    all_features = feat.featurize_all(comps_all)
    INPUT_DIM = all_features.shape[1]
    log.info(f"Feature matrix: {all_features.shape} ({INPUT_DIM} features)")

    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    official_folds = list(kfold.split(comps_all))

    os.makedirs('trm_models_v5', exist_ok=True)

    # ── Model configs ─────────────────────────────────────────────────────
    CONFIGS = [
        # V5A: MLP with SWA + recursion ensemble
        ('V5A-MLP-SWA', MLPTRM,
         dict(input_dim=INPUT_DIM, hidden_dim=64, ff_dim=100,
              dropout=0.2, steps=16)),

        # V5B: Novel feature-group dual-reference Transformer
        ('V5B-FeatGroup-Novel', FeatureGroupTRM,
         dict(n_props=22, stat_dim=6, mat2vec_dim=200,
              d_model=32, nhead=2, ff_dim=64, dropout=0.2, steps=16)),
    ]

    all_results, all_histories, all_trajectories = {}, {}, {}

    for cfg_idx, (cfg_name, ModelClass, cfg_kwargs) in enumerate(CONFIGS):
        t_cfg = time.time()
        print(f"\n{'═'*72}")
        print(f"  [{cfg_idx+1:02d}/{len(CONFIGS):02d}]  {cfg_name}")
        _tmp = ModelClass(**cfg_kwargs)
        n_params = _tmp.count_parameters()
        print(f"  arch={_tmp.arch_type}  params={n_params:,}  "
              f"steps=16  SWA@200  ensemble_k={ENSEMBLE_LAST_K}")
        print(f"{'─'*72}")
        log.info(f"Params: {n_params:,}")
        del _tmp

        fold_maes, fold_maes_ensemble = [], []
        fold_histories, fold_trajectories, fold_details = [], [], []

        for fold, (tv_idx, test_idx) in enumerate(official_folds):
            tv_features = all_features[tv_idx]
            tv_targets  = targets_all[tv_idx]
            te_features = all_features[test_idx]
            te_targets  = targets_all[test_idx]

            torch.manual_seed(SEED + fold)
            np.random.seed(SEED + fold)

            tr_idx, vl_idx = stratified_val_split(
                tv_targets, val_size=0.15, seed=SEED + fold)

            train_features = tv_features[tr_idx]
            train_targets  = tv_targets[tr_idx]
            val_features   = tv_features[vl_idx]
            val_targets    = tv_targets[vl_idx]

            feat.fit_scaler(train_features)
            train_scaled = feat.transform(train_features)
            val_scaled   = feat.transform(val_features)
            test_scaled  = feat.transform(te_features)

            train_ds = SteelsDataset(train_scaled, train_targets)
            val_ds   = SteelsDataset(val_scaled,   val_targets)
            test_ds  = SteelsDataset(test_scaled,  te_targets)

            dl_kw = dict(batch_size=32,
                         pin_memory=(device.type == 'cuda'),
                         num_workers=0)
            train_dl = DataLoader(train_ds, shuffle=True,  **dl_kw)
            val_dl   = DataLoader(val_ds,   shuffle=False, **dl_kw)
            test_dl  = DataLoader(test_ds,  shuffle=False, **dl_kw)

            model = ModelClass(**cfg_kwargs).to(device)
            best_val, model, history = train_fold(
                model, train_dl, val_dl, device,
                epochs=300, swa_start=200,
                fold_idx=fold+1, config_name=cfg_name)

            # ── Test: standard (last step only) ───────────────────────
            model.eval()
            te_loss = 0.0
            with torch.no_grad():
                for bx, by in test_dl:
                    bx, by = bx.to(device), by.to(device)
                    preds   = model(bx)  # no ensemble
                    te_loss += F.l1_loss(preds, by).item() * len(by)
            te_mae_std = te_loss / len(test_dl.dataset)

            # ── Test: with recursion step ensemble ────────────────────
            te_mae_ens = eval_ensemble(model, test_dl, device,
                                       ensemble_k=ENSEMBLE_LAST_K)

            log.info(f"  Fold {fold+1}/5 → Test: {te_mae_std:.4f}  "
                     f"Ensemble: {te_mae_ens:.4f}  Val: {best_val:.4f}")

            fold_maes.append(te_mae_std)
            fold_maes_ensemble.append(te_mae_ens)
            fold_histories.append(history)
            fold_details.append({
                'val_mae': best_val,
                'test_mae': te_mae_std,
                'test_mae_ensemble': te_mae_ens,
            })
            fold_trajectories.append(
                eval_trajectory(model, test_dl, device))

            torch.save({
                'config_name': cfg_name, 'fold': fold+1,
                'kwargs': cfg_kwargs,
                'test_mae': te_mae_std, 'test_mae_ensemble': te_mae_ens,
                'val_mae': best_val,
                'model_state': model.state_dict(), 'history': history,
            }, f'trm_models_v5/{cfg_name}_fold{fold+1}.pt')

        avg_mae = float(np.mean(fold_maes))
        std_mae = float(np.std(fold_maes))
        avg_ens = float(np.mean(fold_maes_ensemble))
        std_ens = float(np.std(fold_maes_ensemble))
        t_cfg   = time.time() - t_cfg

        tag = ("  ← BEATS TPOT 🏆"   if avg_ens < 79.95  else
               "  ← Beats AutoML ✓"   if avg_ens < 82.30  else
               "  ← Beats MODNet ✓"   if avg_ens < 87.76  else
               "  ← Beats RF-SCM ✓"   if avg_ens < 103.51 else
               "  ← Beats CrabNet ✓"  if avg_ens < 107.32 else
               "  ← Beats V4 best ✓"  if avg_ens < 130.33 else
               "  ← Beats V2 MLP ✓"   if avg_ens < 184.57 else "")

        print(f"\n  ✓ {cfg_name}:")
        print(f"    Standard:  {avg_mae:.4f} ± {std_mae:.4f} MPa")
        print(f"    Ensemble:  {avg_ens:.4f} ± {std_ens:.4f} MPa{tag}")
        print(f"    (took {t_cfg/60:.1f} min)")

        all_results[cfg_name] = {
            'avg_mae': avg_mae, 'std_mae': std_mae,
            'avg_mae_ensemble': avg_ens, 'std_mae_ensemble': std_ens,
            'fold_maes': fold_maes,
            'fold_maes_ensemble': fold_maes_ensemble,
            'fold_details': fold_details,
            'params': n_params,
        }
        all_histories[cfg_name]    = fold_histories
        all_trajectories[cfg_name] = np.mean(
            fold_trajectories, axis=0).tolist()

    # ── Final leaderboard ─────────────────────────────────────────────────
    t_total = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V5 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<26} {'Params':>9} {'Standard':>10} {'Ensemble':>10}")
    print(f"  {'─'*58}")
    for name, r in sorted(all_results.items(),
                           key=lambda x: x[1]['avg_mae_ensemble']):
        print(f"  {name:<26} {r['params']:>9,} "
              f"{r['avg_mae']:>10.4f} {r['avg_mae_ensemble']:>10.4f}")
    print(f"  {'─'*58}")
    for bname, bval in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bname:<26} {'baseline':>9} {bval:>10.4f}")
    print(f"\n  Total time: {t_total/60:.1f} minutes\n")

    generate_plots(all_results, all_trajectories, all_histories)
    save_summary(all_results, all_trajectories, t_total)
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 9. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    'V5A-MLP-SWA':          '#1565C0',
    'V5B-FeatGroup-Novel':  '#E65100',
}

PREV_RESULTS = {
    'Combined-S (V4)':   {'mae': 131.6265, 'std': 14.8313},
    'Magpie-L (V3)':     {'mae': 130.3264, 'std': 12.9266},
    'MLP-V2-L':          {'mae': 184.5662, 'std': 11.2077},
}


def generate_plots(results, trajectories, histories):
    names  = list(results.keys())
    # Use ensemble MAE for the main chart
    maes   = [results[n]['avg_mae_ensemble'] for n in names]
    stds   = [results[n]['std_mae_ensemble'] for n in names]
    colors = [PALETTE.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30)

    # ── Panel 1: Main bar chart ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    all_names  = names + list(PREV_RESULTS.keys())
    all_maes   = maes + [PREV_RESULTS[n]['mae'] for n in PREV_RESULTS]
    all_stds   = stds + [PREV_RESULTS[n]['std'] for n in PREV_RESULTS]
    all_colors = colors + ['#90CAF9', '#BBDEFB', '#EF9A9A']

    bars = ax1.bar(all_names, all_maes, yerr=all_stds, capsize=5,
                   color=all_colors, alpha=0.88,
                   edgecolor='white', linewidth=1.2)
    for bv, col, ls, lbl in [
        (79.95,  '#2E7D32', '--', 'TPOT (79.95)'),
        (87.76,  '#F57F17', '--', 'MODNet (87.76)'),
        (103.51, '#795548', ':',  'RF-SCM/Magpie (103.51)'),
        (107.32, '#9E9E9E', ':',  'CrabNet (107.32)'),
    ]:
        ax1.axhline(bv, color=col, linestyle=ls, linewidth=1.8,
                    label=lbl, alpha=0.85)
    for bar, m, s in zip(bars, all_maes, all_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 2,
                 f'{m:.1f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylabel('MAE (MPa)', fontsize=11)
    ax1.set_title('TRM-MatSci V5 (Ensemble MAE) vs Previous │ matbench_steels',
                  fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(all_maes) * 1.25)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20,
             ha='right', fontsize=9)

    # ── Panel 2: Recursion convergence ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for n, traj in trajectories.items():
        ax2.plot(range(1, len(traj)+1), traj, label=n,
                 color=PALETTE.get(n, '#888'), linewidth=2.0, alpha=0.9)
    for bv, col, ls, lbl in [
        (107.32, '#795548', ':', 'CrabNet'),
        (87.76,  '#F57F17', '--', 'MODNet'),
        (79.95,  '#2E7D32', '--', 'TPOT'),
    ]:
        ax2.axhline(bv, color=col, linestyle=ls, linewidth=1.5,
                    label=lbl, alpha=0.8)
    # Shade ensemble region
    ax2.axvspan(16 - ENSEMBLE_LAST_K + 1, 16, alpha=0.15, color='#4CAF50',
                label=f'Ensemble region (steps {16-ENSEMBLE_LAST_K+1}-16)')
    ax2.set_xlabel('Recursion Step (1 → 16)', fontsize=10)
    ax2.set_ylabel('MAE (MPa)', fontsize=10)
    ax2.set_title('Recursion Convergence + Ensemble Region',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3)

    # ── Panel 3: Training curves ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for n in names:
        for fold_i, h in enumerate(histories[n]):
            lbl = f'{n} F{fold_i+1}' if fold_i == 0 else None
            ax3.plot(h['train'], alpha=0.25, linewidth=0.8, color='#1565C0')
            ax3.plot(h['val'], alpha=0.7, linewidth=1.0,
                     color=PALETTE.get(n, '#888'), label=lbl)
    ax3.axhline(107.32, color='#795548', linestyle=':', linewidth=1.2,
                label='CrabNet')
    ax3.axvline(200, color='#4CAF50', linestyle='--', linewidth=1.2,
                alpha=0.6, label='SWA start')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('MAE (MPa)', fontsize=10)
    ax3.set_title('Training Curves + SWA Phase',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(alpha=0.2)

    plt.suptitle(
        'TRM-MatSci V5 │ SWA + Novel Feature-Group TRM │ matbench_steels',
        fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v5.png', dpi=150, bbox_inches='tight')
    log.info("✓ Saved: trm_results_v5.png")
    plt.close(fig)

    # ── Detailed training curves ──────────────────────────────────────────
    n_models = len(names)
    fig2, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
    fig2.suptitle('V5 Training Curves — Train (blue) vs Val (colored)',
                  fontsize=13, fontweight='bold')
    if n_models == 1:
        axes = [axes]
    for ax, n in zip(axes, names):
        for fold_i, h in enumerate(histories[n]):
            ax.plot(h['train'], alpha=0.3, linewidth=1, color='#1565C0')
            ax.plot(h['val'], alpha=0.85, linewidth=1.2,
                    color=PALETTE.get(n, '#888'), label=f'F{fold_i+1}')
        ax.axhline(107.32, color='#795548', linestyle=':',
                   linewidth=1.2, label='CrabNet')
        ax.axvline(200, color='#4CAF50', linestyle='--',
                   linewidth=1.2, alpha=0.6, label='SWA')
        ax.set_title(n, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (MPa)')
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.2)
    plt.tight_layout()
    fig2.savefig('trm_curves_v5.png', dpi=130, bbox_inches='tight')
    plt.close(fig2)
    log.info("✓ Saved: trm_curves_v5.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(results, trajectories, total_s):
    summary = {
        'version': 'V5',
        'task': 'matbench_steels',
        'input_type': 'magpie_132_plus_mat2vec_200_combined_332',
        'innovations': {
            'V5A': 'SWA (epoch 200-300) + recursion step ensemble (avg steps 13-16)',
            'V5B': 'Feature-Group Dual-Reference TRM: 22 Magpie property tokens, '
                   'E0 (fixed) + Et (evolved) cross-attention, z seeded from Mat2Vec',
        },
        'recursion_steps': 16, 'epochs': 300, 'swa_start': 200,
        'ensemble_last_k': ENSEMBLE_LAST_K,
        'dropout': 0.2,
        'total_train_min': round(total_s / 60, 1),
        'baselines': BASELINES,
        'models': {}
    }
    for name, r in results.items():
        summary['models'][name] = {
            'params':              r['params'],
            'avg_mae':             round(r['avg_mae'], 4),
            'std_mae':             round(r['std_mae'], 4),
            'avg_mae_ensemble':    round(r['avg_mae_ensemble'], 4),
            'std_mae_ensemble':    round(r['std_mae_ensemble'], 4),
            'fold_maes':           [round(m, 4) for m in r['fold_maes']],
            'fold_maes_ensemble':  [round(m, 4) for m in r['fold_maes_ensemble']],
            'fold_details':        r['fold_details'],
            'convergence':         [round(v, 4) for v in trajectories[name]],
        }
    with open('trm_models_v5/summary_v5.json', 'w') as f:
        json.dump(summary, f, indent=2)

    rows = []
    for name, r in results.items():
        for fi, (fm, fe, fd) in enumerate(
                zip(r['fold_maes'], r['fold_maes_ensemble'],
                    r['fold_details'])):
            rows.append({
                'model': name, 'fold': fi+1,
                'test_mae': round(fm, 4),
                'test_mae_ensemble': round(fe, 4),
                'val_mae': round(fd['val_mae'], 4)
            })
    pd.DataFrame(rows).to_csv(
        'trm_models_v5/fold_results_v5.csv', index=False)
    log.info("✓ Saved: trm_models_v5/summary_v5.json + fold_results_v5.csv")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_benchmark()

    # Create zip for easy download
    shutil.make_archive("trm_v5_all", "zip", "trm_models_v5")
    log.info("✓ Created trm_v5_all.zip for download")
