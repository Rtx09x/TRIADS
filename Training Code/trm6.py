"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V6 — Scaled Novel Arch + Hybrid + Multi-Seed Ensemble    ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV           ║
║                                                                      ║
║  V6A: FeatureGroup-L  (d=48, nhead=2) — bigger novel TRM + SWA      ║
║  V6B: Hybrid-TRM (attention→pool→MLP-TRM) — best of both + SWA      ║
║  V6C: MLP-SWA ×3 seeds ensemble — average 3 models' predictions     ║
║                                                                      ║
║  All models use SWA (cosine LR 200ep → weight avg 100ep)             ║
║  Target: Beat Darwin (123.29 MPa), reach CrabNet (107.32 MPa)       ║
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
log = logging.getLogger("TRM6")

SEED = 42

BASELINES = {
    'TPOT-Mat (best)':    79.9468,
    'AutoML-Mat':         82.3043,
    'MODNet v0.1.12':     87.7627,
    'RF-SCM/Magpie':     103.5125,
    'CrabNet':           107.3160,
    'Darwin':            123.2932,
    'V5A MLP-SWA':       128.9836,
    'V3 Magpie-L':       130.3264,
    'MLP-TRM V2':        184.5662,
    'Dummy':             229.7445,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMBINED FEATURIZER (Magpie + Mat2Vec)
# ══════════════════════════════════════════════════════════════════════════════

class CombinedFeaturizer:
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
        self.m2v = self._load_m2v()
        self.m2v_embeddings = {w: self.m2v.wv[w] for w in self.m2v.wv.index_to_key}
        self.n_features = self.n_magpie + 200
        log.info(f"Combined: {self.n_magpie} Magpie + 200 Mat2Vec = {self.n_features}d")

    def _load_m2v(self):
        for f in self.M2V_FILES:
            fp = os.path.join(self.cache_dir, f)
            if not os.path.exists(fp):
                log.info(f"  Downloading {f}...")
                urllib.request.urlretrieve(self.GCS_BASE + f, fp)
        return Word2Vec.load(os.path.join(self.cache_dir, "pretrained_embeddings"))

    def _m2v_pooled(self, comp):
        vec, tot = np.zeros(200, dtype=np.float32), 0.0
        for s, fr in comp.get_el_amt_dict().items():
            if s in self.m2v_embeddings:
                vec += fr * self.m2v_embeddings[s]; tot += fr
        return vec / max(tot, 1e-8)

    def featurize_all(self, comps):
        feats = []
        for c in tqdm(comps, desc="  Featurizing", leave=False):
            try: mg = np.array(self.ep.featurize(c), dtype=np.float32)
            except: mg = np.zeros(self.n_magpie, dtype=np.float32)
            feats.append(np.concatenate([np.nan_to_num(mg, nan=0.0),
                                         self._m2v_pooled(c)]))
        return np.array(feats)

    def fit_scaler(self, X):
        self.scaler = StandardScaler().fit(X)

    def transform(self, X):
        if self.scaler is None: return X
        return np.nan_to_num(self.scaler.transform(X), nan=0.0).astype(np.float32)


class SteelsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y, dtype=np.float32))
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

class MLPTRM(nn.Module):
    """V6C: MLP-TRM for multi-seed ensemble."""
    def __init__(self, input_dim=332, hidden_dim=64, ff_dim=100,
                 dropout=0.2, steps=16, **kw):
        super().__init__()
        self.steps, self.D = steps, hidden_dim
        self.arch_type = 'MLP-TRM'
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU())
        self.z_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.y_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, ff_dim), nn.GELU(), nn.Dropout(dropout),
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
        xp = self.input_proj(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        traj = []
        for _ in range(self.steps):
            z = z + self.z_update(torch.cat([xp, y, z], -1))
            y = y + self.y_update(torch.cat([y, z], -1))
            if return_trajectory: traj.append(self.head(y).squeeze(1))
        out = self.head(y).squeeze(1)
        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FeatureGroupTRM(nn.Module):
    """V6A: Scaled-up dual-reference TRM over property tokens.
    Bigger than V5B (d=48 vs 32, ff=96 vs 64).
    """
    def __init__(self, n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_model=48, nhead=2, ff_dim=96, dropout=0.2, steps=16, **kw):
        super().__init__()
        self.steps, self.n_props, self.stat_dim = steps, n_props, stat_dim
        self.d_model, self.D = d_model, d_model
        self.arch_type = 'FeatGroup-L'

        self.token_proj = nn.Sequential(
            nn.Linear(stat_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.z_init = nn.Sequential(
            nn.Linear(mat2vec_dim, d_model), nn.LayerNorm(d_model), nn.GELU())

        # Et self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model))
        self.sa_ff_norm = nn.LayerNorm(d_model)

        # Cross-attention to E0 and Et
        self.cross_e0 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ce0_norm = nn.LayerNorm(d_model)
        self.cross_et = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.cet_norm = nn.LayerNorm(d_model)

        self.z_ff = nn.Sequential(
            nn.Linear(d_model * 3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.LayerNorm(d_model))
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

    def forward(self, x, return_trajectory=False):
        B = x.size(0)
        magpie = x[:, :self.n_props * self.stat_dim]
        mat2vec = x[:, self.n_props * self.stat_dim:]

        E0 = self.token_proj(magpie.view(B, self.n_props, self.stat_dim))
        Et = E0.clone()
        z = self.z_init(mat2vec).unsqueeze(1)
        y = torch.zeros(B, self.d_model, device=x.device)
        traj = []

        for _ in range(self.steps):
            Et = self.sa_norm(Et + self.self_attn(Et, Et, Et)[0])
            Et = self.sa_ff_norm(Et + self.sa_ff(Et))
            z_e0 = self.ce0_norm(z + self.cross_e0(z, E0, E0)[0])
            z_et = self.cet_norm(z + self.cross_et(z, Et, Et)[0])
            z_cat = torch.cat([z.squeeze(1), z_e0.squeeze(1), z_et.squeeze(1)], -1)
            z = z + self.z_ff(z_cat).unsqueeze(1)
            y = y + self.y_ff(torch.cat([y, z.squeeze(1)], -1))
            if return_trajectory: traj.append(self.head(y).squeeze(1))

        out = self.head(y).squeeze(1)
        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HybridTRM(nn.Module):
    """V6B: Attention as learned feature extractor → MLP-TRM for reasoning.

    Step 1: Self-attention over 22 property tokens (learns property interactions)
    Step 2: Cross-attention with Mat2Vec context (adds chemical knowledge)
    Step 3: Mean-pool attended tokens → single attention-enhanced vector
    Step 4: MLP-TRM recursive reasoning over this enriched representation
    """
    def __init__(self, n_props=22, stat_dim=6, mat2vec_dim=200,
                 d_attn=32, nhead=2, d_hidden=64, ff_dim=100,
                 dropout=0.2, steps=16, **kw):
        super().__init__()
        self.steps, self.D = steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.arch_type = 'Hybrid-TRM'

        # ── Attention feature extractor (runs ONCE, not per TRM step) ────
        self.token_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.mat2vec_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # Self-attention: property interactions
        self.self_attn = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa_norm = nn.LayerNorm(d_attn)
        self.sa_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn * 2, d_attn))
        self.sa_ff_norm = nn.LayerNorm(d_attn)

        # Cross-attention: property tokens attend to Mat2Vec context
        self.cross_attn = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_norm = nn.LayerNorm(d_attn)

        # Pool → project to MLP hidden dim
        self.pool_proj = nn.Sequential(
            nn.Linear(d_attn, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())

        # ── MLP-TRM recursive reasoning ──────────────────────────────────
        self.z_update = nn.Sequential(
            nn.Linear(d_hidden * 3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.y_update = nn.Sequential(
            nn.Linear(d_hidden * 2, ff_dim), nn.GELU(), nn.Dropout(dropout),
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
        magpie = x[:, :self.n_props * self.stat_dim]
        mat2vec = x[:, self.n_props * self.stat_dim:]

        # ── Attention feature extraction (one-shot, not recursive) ────────
        tokens = self.token_proj(
            magpie.view(B, self.n_props, self.stat_dim))    # [B, 22, d_attn]
        context = self.mat2vec_proj(mat2vec).unsqueeze(1)    # [B, 1, d_attn]

        # Self-attention: discover property interactions
        tokens = self.sa_norm(tokens + self.self_attn(tokens, tokens, tokens)[0])
        tokens = self.sa_ff_norm(tokens + self.sa_ff(tokens))

        # Cross-attention: enrich with chemical knowledge
        tokens = self.ca_norm(tokens + self.cross_attn(tokens, context, context)[0])

        # Mean pool + project
        xp = self.pool_proj(tokens.mean(dim=1))  # [B, d_hidden]

        # ── MLP-TRM recursive reasoning ──────────────────────────────────
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        traj = []
        for _ in range(self.steps):
            z = z + self.z_update(torch.cat([xp, y, z], -1))
            y = y + self.y_update(torch.cat([y, z], -1))
            if return_trajectory: traj.append(self.head(y).squeeze(1))

        out = self.head(y).squeeze(1)
        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 3. UTILS
# ══════════════════════════════════════════════════════════════════════════════

def stratified_val_split(targets, val_size=0.15, seed=42):
    bins = np.percentile(targets, [25, 50, 75])
    labels = np.digitize(targets, bins)
    tr, vl = [], []
    rng = np.random.RandomState(seed)
    for b in range(4):
        m = np.where(labels == b)[0]
        if len(m) == 0: continue
        n = max(1, int(len(m) * val_size))
        c = rng.choice(m, n, replace=False)
        vl.extend(c.tolist()); tr.extend(np.setdiff1d(m, c).tolist())
    return np.array(tr), np.array(vl)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING WITH SWA
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(model, train_dl, val_dl, device,
               epochs=300, swa_start=200, fold_idx=1, cfg_name=""):
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=swa_start, eta_min=1e-4)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=5e-4)
    swa_on = False

    best_val, best_wts = float('inf'), copy.deepcopy(model.state_dict())
    hist = {'train': [], 'val': []}

    pbar = tqdm(range(epochs), desc=f"  [{cfg_name}] F{fold_idx}/5",
                leave=False, ncols=120)
    for ep in pbar:
        model.train()
        tl = 0.0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            loss = F.l1_loss(model(bx), by)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item() * len(by)
        tl /= len(train_dl.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for bx, by in val_dl:
                bx, by = bx.to(device), by.to(device)
                vl += F.l1_loss(model(bx), by).item() * len(by)
        vl /= len(val_dl.dataset)
        hist['train'].append(tl); hist['val'].append(vl)

        if ep < swa_start:
            sched.step()
            if vl < best_val:
                best_val, best_wts = vl, copy.deepcopy(model.state_dict())
        else:
            if not swa_on:
                log.info(f"  SWA start ep {ep+1}")
                swa_on = True
            swa_model.update_parameters(model); swa_sched.step()

        pbar.set_postfix(Tr=f'{tl:.1f}', Val=f'{vl:.1f}',
                        Best=f'{best_val:.1f}',
                        Ph='SWA' if swa_on else 'COS')

    if swa_on:
        update_bn(train_dl, swa_model, device=device)
        model.load_state_dict(swa_model.module.state_dict())
        log.info(f"  SWA done ({epochs - swa_start} snapshots)")
    else:
        model.load_state_dict(best_wts)
    return best_val, model, hist


def eval_trajectory(model, dl, device):
    model.eval()
    sp, tgts = None, []
    with torch.no_grad():
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            _, traj = model(bx, return_trajectory=True)
            if sp is None: sp = [[] for _ in range(len(traj))]
            for i, s in enumerate(traj):
                sp[i].extend(s.cpu().numpy().tolist())
            tgts.extend(by.cpu().numpy().tolist())
    t = np.array(tgts)
    return [float(np.mean(np.abs(np.array(p) - t))) for p in sp]


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    t0 = time.time()
    print("\n" + "═"*72)
    print("  TRM-MatSci V6 │ Scaled Novel + Hybrid + Ensemble │ matbench_steels")
    print("  V6A: FeatureGroup-L (d=48) + SWA")
    print("  V6B: Hybrid-TRM (attn→MLP) + SWA")
    print("  V6C: MLP-SWA ×3 seeds ensemble")
    print("═"*72 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        log.info("Device: CPU")

    log.info("Loading matbench_steels...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_steels")
    comps_raw   = df['composition'].tolist()
    targets_all = np.array(df['yield strength'].tolist(), dtype=np.float32)
    comps_all   = [Composition(c) for c in comps_raw]
    log.info(f"Dataset: {len(df)} samples")

    log.info("Computing features...")
    feat = CombinedFeaturizer()
    all_features = feat.featurize_all(comps_all)
    INPUT_DIM = all_features.shape[1]
    log.info(f"Features: {all_features.shape}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    os.makedirs('trm_models_v6', exist_ok=True)

    # ── Single-model configs ──────────────────────────────────────────────
    SINGLE_CONFIGS = [
        ('V6A-FeatGroup-L', FeatureGroupTRM,
         dict(n_props=22, stat_dim=6, mat2vec_dim=200,
              d_model=48, nhead=2, ff_dim=96, dropout=0.2, steps=16)),
        ('V6B-Hybrid-TRM', HybridTRM,
         dict(n_props=22, stat_dim=6, mat2vec_dim=200,
              d_attn=32, nhead=2, d_hidden=64, ff_dim=100,
              dropout=0.2, steps=16)),
    ]

    all_results, all_hist, all_traj = {}, {}, {}

    # ── Train single-model configs ────────────────────────────────────────
    for ci, (cname, MCls, ckw) in enumerate(SINGLE_CONFIGS):
        tc = time.time()
        _t = MCls(**ckw); np_ = _t.count_parameters()
        print(f"\n{'═'*72}")
        print(f"  [{ci+1}/3]  {cname}  ({_t.arch_type}, {np_:,} params)")
        print(f"{'─'*72}"); del _t

        fmaes, fhist, ftraj, fdet = [], [], [], []
        for fi, (tv_i, te_i) in enumerate(folds):
            torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
            tri, vli = stratified_val_split(targets_all[tv_i], 0.15, SEED+fi)

            feat.fit_scaler(all_features[tv_i][tri])
            tr_s = feat.transform(all_features[tv_i][tri])
            vl_s = feat.transform(all_features[tv_i][vli])
            te_s = feat.transform(all_features[te_i])

            dl_kw = dict(batch_size=32, pin_memory=(device.type=='cuda'), num_workers=0)
            tr_dl = DataLoader(SteelsDataset(tr_s, targets_all[tv_i][tri]), shuffle=True, **dl_kw)
            vl_dl = DataLoader(SteelsDataset(vl_s, targets_all[tv_i][vli]), shuffle=False, **dl_kw)
            te_dl = DataLoader(SteelsDataset(te_s, targets_all[te_i]), shuffle=False, **dl_kw)

            model = MCls(**ckw).to(device)
            bv, model, hist = train_fold(model, tr_dl, vl_dl, device,
                                         epochs=300, swa_start=200,
                                         fold_idx=fi+1, cfg_name=cname)
            model.eval()
            tel = 0.0
            with torch.no_grad():
                for bx, by in te_dl:
                    bx, by = bx.to(device), by.to(device)
                    tel += F.l1_loss(model(bx), by).item() * len(by)
            te_mae = tel / len(te_dl.dataset)
            log.info(f"  Fold {fi+1}/5 → Test: {te_mae:.4f}  Val: {bv:.4f}")

            fmaes.append(te_mae); fhist.append(hist)
            fdet.append({'val_mae': bv, 'test_mae': te_mae})
            ftraj.append(eval_trajectory(model, te_dl, device))

            torch.save({'config': cname, 'fold': fi+1, 'test_mae': te_mae,
                        'val_mae': bv, 'model_state': model.state_dict()},
                       f'trm_models_v6/{cname}_fold{fi+1}.pt')

        avg, std = float(np.mean(fmaes)), float(np.std(fmaes))
        tag = ("  ← Beats CrabNet ✓"  if avg < 107.32 else
               "  ← Beats Darwin ✓"   if avg < 123.29 else
               "  ← Beats V5A ✓"      if avg < 128.98 else
               "  ← Beats V2 ✓"       if avg < 184.57 else "")
        print(f"\n  ✓ {cname}: {avg:.4f} ± {std:.4f} MPa{tag}"
              f"  ({(time.time()-tc)/60:.1f} min)")

        all_results[cname] = {'avg_mae': avg, 'std_mae': std,
            'fold_maes': fmaes, 'fold_details': fdet, 'params': np_}
        all_hist[cname] = fhist
        all_traj[cname] = np.mean(ftraj, axis=0).tolist()

    # ── V6C: Multi-seed ensemble ──────────────────────────────────────────
    N_SEEDS = 3
    cname = 'V6C-MLP-Ensemble'
    ckw = dict(input_dim=INPUT_DIM, hidden_dim=64, ff_dim=100,
               dropout=0.2, steps=16)
    _t = MLPTRM(**ckw); np_c = _t.count_parameters(); del _t

    tc = time.time()
    print(f"\n{'═'*72}")
    print(f"  [3/3]  {cname}  (MLP-TRM ×{N_SEEDS} seeds, {np_c:,} params each)")
    print(f"{'─'*72}")

    ens_fmaes, ens_fhist, ens_ftraj, ens_fdet = [], [], [], []
    for fi, (tv_i, te_i) in enumerate(folds):
        tri, vli = stratified_val_split(targets_all[tv_i], 0.15, SEED+fi)
        feat.fit_scaler(all_features[tv_i][tri])
        tr_s = feat.transform(all_features[tv_i][tri])
        vl_s = feat.transform(all_features[tv_i][vli])
        te_s = feat.transform(all_features[te_i])

        dl_kw = dict(batch_size=32, pin_memory=(device.type=='cuda'), num_workers=0)
        tr_dl = DataLoader(SteelsDataset(tr_s, targets_all[tv_i][tri]), shuffle=True, **dl_kw)
        vl_dl = DataLoader(SteelsDataset(vl_s, targets_all[tv_i][vli]), shuffle=False, **dl_kw)
        te_dl = DataLoader(SteelsDataset(te_s, targets_all[te_i]), shuffle=False, **dl_kw)

        # Train N_SEEDS models
        seed_models, seed_vals = [], []
        for si in range(N_SEEDS):
            torch.manual_seed(SEED + fi * 100 + si)
            np.random.seed(SEED + fi * 100 + si)
            model = MLPTRM(**ckw).to(device)
            bv, model, hist = train_fold(
                model, tr_dl, vl_dl, device, epochs=300, swa_start=200,
                fold_idx=fi+1, cfg_name=f"{cname}-s{si}")
            seed_models.append(model)
            seed_vals.append(bv)
            if si == 0: ens_fhist.append(hist)

        # Ensemble predictions: average across seeds
        all_preds, all_targets = [], []
        for model in seed_models:
            model.eval()
            preds_i = []
            with torch.no_grad():
                for bx, by in te_dl:
                    bx, by = bx.to(device), by.to(device)
                    preds_i.append(model(bx).cpu())
                    if len(all_targets) == 0:
                        pass  # targets collected below
            all_preds.append(torch.cat(preds_i))

        # Collect targets
        tgts = []
        for _, by in te_dl: tgts.append(by)
        tgts = torch.cat(tgts)

        # Average predictions
        ens_pred = torch.stack(all_preds).mean(dim=0)
        te_mae = F.l1_loss(ens_pred, tgts).item()

        # Individual MAEs for reference
        indiv = [F.l1_loss(p, tgts).item() for p in all_preds]
        avg_val = np.mean(seed_vals)
        log.info(f"  Fold {fi+1}/5 → Ensemble: {te_mae:.4f}  "
                 f"Individual: {[f'{m:.1f}' for m in indiv]}  Val-avg: {avg_val:.1f}")

        ens_fmaes.append(te_mae)
        ens_fdet.append({'val_mae': avg_val, 'test_mae': te_mae,
                         'individual_maes': indiv})
        ens_ftraj.append(eval_trajectory(seed_models[0], te_dl, device))

    avg_e, std_e = float(np.mean(ens_fmaes)), float(np.std(ens_fmaes))
    tag = ("  ← Beats CrabNet ✓"  if avg_e < 107.32 else
           "  ← Beats Darwin ✓"   if avg_e < 123.29 else
           "  ← Beats V5A ✓"      if avg_e < 128.98 else
           "  ← Beats V2 ✓"       if avg_e < 184.57 else "")
    print(f"\n  ✓ {cname} ({N_SEEDS} seeds): {avg_e:.4f} ± {std_e:.4f} MPa{tag}"
          f"  ({(time.time()-tc)/60:.1f} min)")

    all_results[cname] = {'avg_mae': avg_e, 'std_mae': std_e,
        'fold_maes': ens_fmaes, 'fold_details': ens_fdet,
        'params': np_c, 'n_seeds': N_SEEDS}
    all_hist[cname] = ens_fhist
    all_traj[cname] = np.mean(ens_ftraj, axis=0).tolist()

    # ── Final leaderboard ─────────────────────────────────────────────────
    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V6 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<26} {'Params':>9} {'MAE(MPa)':>10} {'±Std':>8}")
    print(f"  {'─'*56}")
    for n, r in sorted(all_results.items(), key=lambda x: x[1]['avg_mae']):
        seeds_tag = f" ×{r.get('n_seeds','')}" if 'n_seeds' in r else ""
        print(f"  {n:<26} {r['params']:>9,}{seeds_tag} "
              f"{r['avg_mae']:>10.4f} {r['std_mae']:>8.4f}")
    print(f"  {'─'*56}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<26} {'baseline':>9} {bv:>10.4f}")
    print(f"\n  Total time: {tt/60:.1f} minutes\n")

    generate_plots(all_results, all_traj, all_hist)
    save_summary(all_results, all_traj, tt)
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    'V6A-FeatGroup-L':  '#E65100',
    'V6B-Hybrid-TRM':   '#6A1B9A',
    'V6C-MLP-Ensemble': '#1565C0',
}

PREV = {
    'V5A MLP-SWA':   {'mae': 128.9836, 'std': 17.4191},
    'V3 Magpie-L':   {'mae': 130.3264, 'std': 12.9266},
}

def generate_plots(results, trajs, hists):
    names = list(results.keys())
    maes = [results[n]['avg_mae'] for n in names]
    stds = [results[n]['std_mae'] for n in names]
    cols = [PALETTE.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, :])
    an = names + list(PREV.keys())
    am = maes + [PREV[n]['mae'] for n in PREV]
    ast = stds + [PREV[n]['std'] for n in PREV]
    ac = cols + ['#90CAF9', '#BBDEFB']

    bars = ax1.bar(an, am, yerr=ast, capsize=5, color=ac, alpha=0.88,
                   edgecolor='white', linewidth=1.2)
    for bv, c, ls, lb in [
        (79.95, '#2E7D32', '--', 'TPOT (79.95)'),
        (87.76, '#F57F17', '--', 'MODNet (87.76)'),
        (103.51, '#795548', ':', 'RF-SCM (103.51)'),
        (107.32, '#9E9E9E', ':', 'CrabNet (107.32)'),
        (123.29, '#FF5722', ':', 'Darwin (123.29)'),
    ]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    for bar, m, s in zip(bars, am, ast):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+2,
                 f'{m:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_ylabel('MAE (MPa)'); ax1.set_title(
        'TRM-MatSci V6 vs Previous │ matbench_steels', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(am)*1.25); ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha='right', fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    for n, tr in trajs.items():
        ax2.plot(range(1, len(tr)+1), tr, label=n,
                 color=PALETTE.get(n, '#888'), linewidth=2.0, alpha=0.9)
    for bv, c, ls, lb in [(107.32, '#795548', ':', 'CrabNet'),
                            (87.76, '#F57F17', '--', 'MODNet')]:
        ax2.axhline(bv, color=c, linestyle=ls, linewidth=1.5, label=lb, alpha=0.8)
    ax2.set_xlabel('Recursion Step'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_title('Recursion Convergence', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    for n in names:
        for fi, h in enumerate(hists[n]):
            lb = f'{n}' if fi == 0 else None
            ax3.plot(h['train'], alpha=0.2, linewidth=0.8, color=PALETTE.get(n, '#888'))
            ax3.plot(h['val'], alpha=0.7, linewidth=1.0,
                     color=PALETTE.get(n, '#888'), label=lb)
    ax3.axhline(107.32, color='#795548', linestyle=':', linewidth=1.2, label='CrabNet')
    ax3.axvline(200, color='#4CAF50', linestyle='--', linewidth=1.2, alpha=0.6, label='SWA')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE (MPa)')
    ax3.set_title('Training Curves', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right'); ax3.grid(alpha=0.2)

    plt.suptitle('TRM-MatSci V6 │ matbench_steels', fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v6.png', dpi=150, bbox_inches='tight')
    log.info("✓ Saved: trm_results_v6.png"); plt.close(fig)

    nm = len(names)
    fig2, axes = plt.subplots(1, nm, figsize=(8*nm, 6))
    fig2.suptitle('V6 Training Curves', fontsize=13, fontweight='bold')
    if nm == 1: axes = [axes]
    for ax, n in zip(axes, names):
        for fi, h in enumerate(hists[n]):
            ax.plot(h['train'], alpha=0.3, linewidth=1, color='#1565C0')
            ax.plot(h['val'], alpha=0.85, linewidth=1.2,
                    color=PALETTE.get(n, '#888'), label=f'F{fi+1}')
        ax.axhline(107.32, color='#795548', linestyle=':', linewidth=1.2, label='CrabNet')
        ax.axvline(200, color='#4CAF50', linestyle='--', linewidth=1.2, alpha=0.6, label='SWA')
        ax.set_title(n, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (MPa)')
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.2)
    plt.tight_layout()
    fig2.savefig('trm_curves_v6.png', dpi=130, bbox_inches='tight')
    plt.close(fig2); log.info("✓ Saved: trm_curves_v6.png")


def save_summary(results, trajs, total_s):
    s = {
        'version': 'V6', 'task': 'matbench_steels',
        'innovations': {
            'V6A': 'FeatureGroup-L: d=48 dual-reference TRM + SWA',
            'V6B': 'Hybrid-TRM: attention feature extractor → MLP-TRM + SWA',
            'V6C': 'Multi-seed ensemble: 3 MLP-SWA models averaged',
        },
        'total_train_min': round(total_s/60, 1),
        'baselines': BASELINES, 'models': {}
    }
    for n, r in results.items():
        s['models'][n] = {
            'params': r['params'], 'avg_mae': round(r['avg_mae'], 4),
            'std_mae': round(r['std_mae'], 4),
            'fold_maes': [round(m, 4) for m in r['fold_maes']],
            'convergence': [round(v, 4) for v in trajs[n]],
        }
    with open('trm_models_v6/summary_v6.json', 'w') as f:
        json.dump(s, f, indent=2)

    rows = []
    for n, r in results.items():
        for fi, (fm, fd) in enumerate(zip(r['fold_maes'], r['fold_details'])):
            rows.append({'model': n, 'fold': fi+1,
                         'test_mae': round(fm, 4), 'val_mae': round(fd['val_mae'], 4)})
    pd.DataFrame(rows).to_csv('trm_models_v6/fold_results_v6.csv', index=False)
    log.info("✓ Saved: summary_v6.json + fold_results_v6.csv")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v6_all", "zip", "trm_models_v6")
    log.info("✓ Created trm_v6_all.zip")
