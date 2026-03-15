"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRIADS on matbench_expt_gap — Experimental Band Gap Prediction     ║
║  Dataset: matbench_expt_gap │ 4604 samples │ 5-Fold Nested CV       ║
║                                                                      ║
║  EG-A  Same-size V13A architecture (d_attn=64, d_hidden=96)         ║
║        2-Layer SA + Standard Deep Supervision, 20 steps              ║
║        Expected ~224K params                                         ║
║                                                                      ║
║  EG-B  Smaller architecture (d_attn=32, d_hidden=64)                ║
║        2-Layer SA + Standard Deep Supervision, 16 steps              ║
║        Expected ~50-60K params                                       ║
║                                                                      ║
║  Both: Expanded features (Magpie + Mat2Vec + Extra descriptors)      ║
║  Single seed (42) for initial testing                                ║
║  Official Matbench splits: KFold(5, shuffle=True, rs=18012019)       ║
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
log = logging.getLogger("TRIADS-EG")

# ── Single seed for initial run ──────────────────────────────────────
SEEDS = [42]
N_SEEDS = len(SEEDS)

# ── Baselines for matbench_expt_gap ──────────────────────────────────
BASELINES = {
    'Darwin':              0.2865,
    'Ax/SAASBO CrabNet':   0.3310,
    'MODNet v0.1.12':      0.3327,
    'AMMExpress v2020':    0.4161,
    'CrabNet':             0.4427,
    'RF-SCM/Magpie':       0.5205,
    'Dummy':               1.0280,
}


# ══════════════════════════════════════════════════════════════════════
# 1. FEATURIZER + DATASET
# ══════════════════════════════════════════════════════════════════════

class ExpandedFeaturizer:
    """Magpie (22 props × 6 stats) + Extra matminer descriptors + Mat2Vec (200d).

    Extra descriptors: ElementFraction, Stoichiometry, ValenceOrbital,
    IonProperty, BandCenter — all concatenated as a flat vector between
    the Magpie block and Mat2Vec.

    NOTE: n_extra will be LARGER for expt_gap than steels because
    ElementFraction creates one feature per unique element in the dataset.
    Steels had ~15 elements, expt_gap has ~60+ elements.
    The model architecture handles this automatically via the n_extra parameter.
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


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be pre-loaded into GPU memory.
    This eliminates the CPU->GPU transfer overhead entirely.
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate number of batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len, device=self.tensors[0].device)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        
        end = min(self.i + self.batch_size, self.dataset_len)
        batch = [t[self.i:end] for t in self.tensors]
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

    @property
    def dataset(self):
        return self

# ══════════════════════════════════════════════════════════════════════
# 2. MODEL — DeepHybridTRM (V13A architecture)
# ══════════════════════════════════════════════════════════════════════

class DeepHybridTRM(nn.Module):
    """2-Layer SA Hybrid-TRM with Standard Deep Supervision.

    Architecture:
      - TWO self-attention layers (SA1 → FF1 → SA2 → FF2 → CA)
      - Each SA layer has its own residual + LayerNorm + FF block
      - Cross-attention (CA) to Mat2Vec context after SA stack
      - Recursive MLP reasoning loop (shared weights)
      - Deep supervision at every recursion step

    Parameters are fully dynamic — n_extra auto-detected from featurizer.
    Same architecture for steels and expt_gap, just different data.
    """
    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, max_steps=20, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.n_extra = n_extra

        # ── Attention feature extractor (2-Layer SA) ──────────────────
        self.tok_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # Self-Attention Layer 1
        self.sa1 = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa1_n = nn.LayerNorm(d_attn)
        self.sa1_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa1_fn = nn.LayerNorm(d_attn)

        # Self-Attention Layer 2 (higher-order property interactions)
        self.sa2 = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa2_n = nn.LayerNorm(d_attn)
        self.sa2_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa2_fn = nn.LayerNorm(d_attn)

        # Cross-Attention to Mat2Vec context (after SA stack)
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

        # SA Layer 1: learn pairwise property interactions
        tok = self.sa1_n(tok + self.sa1(tok, tok, tok)[0])
        tok = self.sa1_fn(tok + self.sa1_ff(tok))

        # SA Layer 2: learn higher-order property interactions
        tok = self.sa2_n(tok + self.sa2(tok, tok, tok)[0])
        tok = self.sa2_fn(tok + self.sa2_ff(tok))

        # Cross-Attention to Mat2Vec chemistry context
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


# ══════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTION — Deep Supervision (L1)
# ══════════════════════════════════════════════════════════════════════

def deep_supervision_loss(step_preds, targets):
    """Linear-weighted L1 loss across all recursion steps.
    Step t receives weight (t+1), forcing calibrated predictions
    throughout the entire trajectory.
    """
    n = len(step_preds)
    weights = [(i + 1) for i in range(n)]
    total_w = sum(weights)
    loss = 0.0
    for pred, w in zip(step_preds, weights):
        loss += (w / total_w) * F.l1_loss(pred, targets)
    return loss


# ══════════════════════════════════════════════════════════════════════
# 4. UTILS + TRAINING (GPU-optimized)
# ══════════════════════════════════════════════════════════════════════

def strat_split(targets, val_size=0.15, seed=42):
    """Stratified train/val split within an outer fold.
    Bins targets into quartiles and samples proportionally.
    NO data from the test fold is ever seen — the targets array
    only contains train+val indices from the outer KFold split.
    """
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
    """Training with standard deep supervision + SWA.

    GPU optimizations:
      - non_blocking=True on .to(device) for async CPU→GPU transfer
      - Mixed precision disabled (too small model to benefit, adds overhead)
      - Gradient clipping at 1.0
    """
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
            # Tensors are already on GPU!
            step_preds = model(bx, deep_supervision=True)
            loss = deep_supervision_loss(step_preds, by)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += F.l1_loss(step_preds[-1], by).item() * len(by)
        tl /= tr_dl.dataset_len

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                pred = model(bx)
                vl += F.l1_loss(pred, by).item() * len(by)
        vl /= vl_dl.dataset_len
        hist['train'].append(tl); hist['val'].append(vl)

        if ep < swa_start:
            sch.step()
            if vl < best_v: best_v, best_w = vl, copy.deepcopy(model.state_dict())
        else:
            if not swa_on: swa_on = True
            swa_m.update_parameters(model); swa_s.step()

        pbar.set_postfix(Tr=f'{tl:.4f}', Val=f'{vl:.4f}',
                        Best=f'{best_v:.4f}', Ph='SWA' if swa_on else 'COS')

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
            preds.append(model(bx).cpu())
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
    print("  TRIADS on matbench_expt_gap │ Experimental Band Gap Prediction")
    print("  EG-A: Same-size V13A (d_attn=64, d_hidden=96, 20 steps)")
    print("  EG-B: Smaller (d_attn=32, d_hidden=64, 16 steps)")
    print(f"  Seeds: {SEEDS} │ Single-seed initial run")
    print("═"*72 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ── GPU optimization: Full VRAM Loading ───────────────────────────
    # Since the dataset is small (~10MB), we load it ALL directly into 
    # the GPU memory. This drops CPU usage to 0% and maximizes GPU,
    # solving the Kaggle CPU bottleneck completely.
    if device.type == 'cuda':
        log.info("DataLoader: FastTensorDataLoader (Full VRAM loading)")
    else:
        log.info("DataLoader: FastTensorDataLoader (CPU fallback)")

    # ══════════════════════════════════════════════════════════════════
    # LOAD DATA — matbench_expt_gap
    # ══════════════════════════════════════════════════════════════════
    log.info("Loading matbench_expt_gap...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_expt_gap")

    # Column names for matbench_expt_gap:
    #   'composition' → chemical formula string
    #   'gap expt'    → experimental band gap in eV
    comps_raw = df['composition'].tolist()
    targets_all = np.array(df['gap expt'].tolist(), np.float32)
    comps_all = [Composition(c) for c in comps_raw]

    log.info(f"Dataset: {len(comps_all)} samples")
    log.info(f"Target stats: mean={targets_all.mean():.3f} eV, "
             f"std={targets_all.std():.3f} eV, "
             f"min={targets_all.min():.3f}, max={targets_all.max():.3f}")

    # ── FEATURIZE ─────────────────────────────────────────────────────
    log.info("Computing EXPANDED features...")
    feat = ExpandedFeaturizer()
    X_all = feat.featurize_all(comps_all)
    n_extra = feat.n_extra
    log.info(f"Features: {X_all.shape} (n_extra={n_extra})")

    # ── OFFICIAL MATBENCH SPLITS ──────────────────────────────────────
    # This is the EXACT same KFold config used by Matbench.
    # random_state=18012019 is the official Matbench seed.
    # Using sklearn KFold directly ensures identical splits to the
    # official benchmark — NO data leakage possible because:
    #   1. Outer KFold splits are never modified
    #   2. Inner strat_split only touches train+val indices
    #   3. Test fold indices are NEVER used during training/val
    #   4. Scaler is fit ONLY on the training subset (not val, not test)
    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))

    # Verify no leakage
    for fi, (tv_i, te_i) in enumerate(folds):
        overlap = set(tv_i) & set(te_i)
        assert len(overlap) == 0, f"DATA LEAK in fold {fi+1}! {len(overlap)} shared indices"
    log.info("✓ All 5 folds verified: no train-test overlap")

    os.makedirs('expt_gap_models_v2', exist_ok=True)

    # ── DataLoader kwargs ─────────────────────────────────────────────
    # Batch size 64. No more CPU workers needed.
    dl_kw = dict(batch_size=64)

    # ── CONFIGS ───────────────────────────────────────────────────────
    # Both use DeepHybridTRM (V13A architecture) with standard DS.
    # n_extra is auto-detected from featurizer — will be different from
    # steels due to wider element coverage in expt_gap.

    configs = {
        'EG-A-SameSize': {
            'model_cls': DeepHybridTRM,
            'model_kw': dict(
                n_props=22, stat_dim=6, n_extra=n_extra,
                mat2vec_dim=200, d_attn=64, nhead=4,
                d_hidden=96, ff_dim=150, dropout=0.15,
                max_steps=20,
            ),
            'epochs': 300,
            'swa_start': 200,
        },
        'EG-B-Smaller': {
            'model_cls': DeepHybridTRM,
            'model_kw': dict(
                n_props=22, stat_dim=6, n_extra=n_extra,
                mat2vec_dim=200, d_attn=32, nhead=4,
                d_hidden=64, ff_dim=96, dropout=0.15,
                max_steps=16,
            ),
            'epochs': 300,
            'swa_start': 200,
        },
    }

    # Print param counts
    print(f"\n  {'Config':<24} {'Params':>10} {'Steps':>8}  {'Dropout':>8}")
    print(f"  {'─'*56}")
    for cname, cfg in configs.items():
        _m = cfg['model_cls'](**cfg['model_kw'])
        np_ = _m.count_parameters(); del _m
        cfg['n_params'] = np_
        steps = cfg['model_kw']['max_steps']
        drop = cfg['model_kw']['dropout']
        print(f"  {cname:<24} {np_:>10,} {steps:>8}  {drop:>8.2f}")
    print()

    # ── TRAIN + EVALUATE ──────────────────────────────────────────────
    all_results = {}
    all_hists = {}

    for cname, cfg in configs.items():
        print(f"\n{'═'*72}")
        print(f"  {cname}")
        print(f"{'═'*72}")

        seed = SEEDS[0]
        fold_maes = []
        fold_hists = []
        fold_preds = {}

        for fi, (tv_i, te_i) in enumerate(folds):
            print(f"\n  ── [{cname} seed={seed}] Fold {fi+1}/5 {'─'*30}")

            # ── DATA SPLIT (no leakage) ───────────────────────────────
            # tv_i = train+val indices, te_i = test indices (held out)
            # strat_split further divides tv_i into train and val
            # Scaler is fit ONLY on train subset
            tri, vli = strat_split(targets_all[tv_i], 0.15, seed+fi)

            # Scaler fitted ONLY on training data (no val, no test)
            feat.fit_scaler(X_all[tv_i][tri])
            
            # Transform and move DIRECTLY TO GPU
            tr_x = torch.tensor(feat.transform(X_all[tv_i][tri]), dtype=torch.float32).to(device)
            tr_y = torch.tensor(targets_all[tv_i][tri], dtype=torch.float32).to(device)
            
            vl_x = torch.tensor(feat.transform(X_all[tv_i][vli]), dtype=torch.float32).to(device)
            vl_y = torch.tensor(targets_all[tv_i][vli], dtype=torch.float32).to(device)
            
            te_x = torch.tensor(feat.transform(X_all[te_i]), dtype=torch.float32).to(device)
            te_y = torch.tensor(targets_all[te_i], dtype=torch.float32).to(device)

            tr_dl = FastTensorDataLoader(tr_x, tr_y, shuffle=True, **dl_kw)
            vl_dl = FastTensorDataLoader(vl_x, vl_y, shuffle=False, **dl_kw)
            te_dl = FastTensorDataLoader(te_x, te_y, shuffle=False, **dl_kw)
            
            te_tgt = get_targets(te_dl).cpu()  # Needed for evaluation later

            log.info(f"  Fold {fi+1}: train={len(tr_x)}, val={len(vl_x)}, "
                     f"test={len(te_x)}")

            # ── Seed everything ───────────────────────────────────────
            torch.manual_seed(seed + fi)
            np.random.seed(seed + fi)
            if device.type == 'cuda':
                torch.cuda.manual_seed(seed + fi)

            model = cfg['model_cls'](**cfg['model_kw']).to(device)
            bv, model, hist = train_fold(
                model, tr_dl, vl_dl, device,
                epochs=cfg['epochs'], swa_start=cfg['swa_start'],
                fold=fi+1, name=cname,
            )

            fold_hists.append(hist)

            # Predict on test fold
            pred = predict(model, te_dl, device)
            mae = F.l1_loss(pred, te_tgt).item()
            log.info(f"  Fold {fi+1}: test MAE = {mae:.4f} eV  (val best = {bv:.4f})")

            fold_preds[fi] = pred
            fold_maes.append(mae)

            torch.save({
                'model_state': model.state_dict(),
                'test_mae': mae, 'config': cname, 'seed': seed,
                'fold': fi+1, 'n_extra': n_extra,
            }, f'expt_gap_models_v2/{cname}_seed{seed}_fold{fi+1}.pt')

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_mae = float(np.mean(fold_maes))
        std_mae = float(np.std(fold_maes))

        all_results[cname] = {
            'avg': avg_mae, 'std': std_mae, 'folds': fold_maes,
            'params': cfg['n_params'],
        }
        all_hists[cname] = fold_hists

        print(f"\n  ═══ {cname} ═══")
        print(f"      5-Fold Avg MAE: {avg_mae:.4f} ± {std_mae:.4f} eV")
        print(f"      Per-fold: {[f'{m:.4f}' for m in fold_maes]}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════
    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_expt_gap (5-Fold Avg MAE, eV)")
    print(f"{'═'*72}")
    print(f"  {'Model':<28} {'Params':>10} {'MAE(eV)':>10} {'±Std':>8}  Notes")
    print(f"  {'─'*72}")

    for n, r in sorted(all_results.items(), key=lambda x: x[1]['avg']):
        tag = ("  ← BEATS Darwin 🏆" if r['avg'] < 0.2865 else
               "  ← BEATS CrabNet ✓"  if r['avg'] < 0.3310 else
               "  ← BEATS MODNet ✓"   if r['avg'] < 0.3327 else
               "  ← BEATS AMMExpress ✓" if r['avg'] < 0.4161 else "")
        print(f"  {n:<28} {r['params']:>9,} "
              f"{r['avg']:>10.4f} {r['std']:>8.4f}{tag}")

    print(f"  {'─'*72}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<28} {'baseline':>10} {bv:>10.4f}")

    print(f"\n  Total time: {tt/60:.1f} minutes")

    # Per-fold breakdown
    print(f"\n{'═'*72}")
    print(f"  PER-FOLD BREAKDOWN (MAE in eV)")
    print(f"{'═'*72}")
    cnames = list(all_results.keys())
    header = f"  {'Fold':<6}"
    for cn in cnames:
        header += f" {cn:>20}"
    print(header)
    print(f"  {'─'*50}")
    for fi in range(5):
        row = f"  {fi+1:<6}"
        for cn in cnames:
            row += f" {all_results[cn]['folds'][fi]:>20.4f}"
        print(row)
    print()

    generate_plots(all_results, all_hists)
    save_summary(all_results, all_hists, tt, n_extra)
    return all_results


# ══════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════

PAL = {'EG-A-SameSize': '#1565C0', 'EG-B-Smaller': '#E65100'}

def generate_plots(all_results, all_hists):
    names = list(all_results.keys())
    avgs = [all_results[n]['avg'] for n in names]
    stds = [all_results[n]['std'] for n in names]
    cols = [PAL.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ── Plot 1: Bar chart vs baselines ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(names))
    bars = ax1.bar(x_pos, avgs, 0.5, yerr=stds, capsize=6,
                   color=cols, alpha=0.88, edgecolor='white', linewidth=1.5)

    baseline_lines = [
        (0.2865, '#F44336', '--', 'Darwin (0.2865)'),
        (0.3310, '#FF9800', '-.', 'Ax CrabNet (0.331)'),
        (0.3327, '#F57F17', ':', 'MODNet (0.3327)'),
        (0.4161, '#9E9E9E', ':', 'AMMExpress (0.4161)'),
    ]
    for bv, c, ls, lb in baseline_lines:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)

    for bar, m, s in zip(bars, avgs, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.005,
                 f'{m:.4f}', ha='center', fontsize=11, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_ylabel('MAE (eV)'); ax1.set_ylim(0, max(avgs)*1.6)
    ax1.set_title('TRIADS on matbench_expt_gap vs Baselines',
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # ── Plot 2: Per-fold grouped bars ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(1, 6)
    w = 0.35
    for i, (n, col) in enumerate(zip(names, cols)):
        fold_vals = all_results[n]['folds']
        ax2.bar(x + (i - 0.5) * w, fold_vals, w, color=col, alpha=0.8,
                label=n, edgecolor='white')
    ax2.axhline(0.2865, color='#F44336', ls='--', lw=1.5, label='Darwin (0.2865)')
    ax2.axhline(0.3327, color='#F57F17', ls=':', lw=1.5, label='MODNet (0.3327)')
    ax2.set_xlabel('Fold'); ax2.set_ylabel('MAE (eV)')
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
    ax3.axhline(0.2865, color='#F44336', ls='--', lw=1.2, label='Darwin (0.2865)')
    ax3.axvline(200, color='#4CAF50', ls='--', lw=1.2, alpha=0.6, label='SWA start')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE (eV)')
    ax3.set_title('Training Curves (all folds)', fontweight='bold')
    ax3.legend(fontsize=6, ncol=2); ax3.grid(alpha=0.2)

    # ── Plot 4: Model comparison summary ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    # Scatter: params vs MAE
    for n, col in zip(names, cols):
        r = all_results[n]
        ax4.scatter(r['params'], r['avg'], s=200, c=col, alpha=0.9,
                   label=f"{n} ({r['avg']:.4f} eV)", zorder=5,
                   edgecolors='white', linewidth=2)
        ax4.errorbar(r['params'], r['avg'], yerr=r['std'],
                    color=col, capsize=8, capthick=2, linewidth=0, elinewidth=2)

    for bname, bval in BASELINES.items():
        if bval < 0.5:  # only show competitive baselines
            ax4.axhline(bval, color='#999', ls=':', lw=1, alpha=0.5)
            ax4.text(ax4.get_xlim()[1]*0.95, bval, f'  {bname}',
                    fontsize=6, ha='right', va='bottom', alpha=0.6)

    ax4.set_xlabel('Parameters')
    ax4.set_ylabel('MAE (eV)')
    ax4.set_title('Parameters vs Performance', fontweight='bold')
    ax4.legend(fontsize=8); ax4.grid(alpha=0.2)

    fig.suptitle('TRIADS │ matbench_expt_gap │ Experimental Band Gap Prediction',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('expt_gap_results_v2.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: expt_gap_results_v2.png")


def save_summary(all_results, all_hists, total_s, n_extra):
    s = {
        'version': 'EG-V2',
        'task': 'matbench_expt_gap',
        'dataset_size': 4604,
        'target': 'gap expt (eV)',
        'strategy': 'TRIADS V13A architecture, standard deep supervision',
        'seeds': SEEDS,
        'n_seeds': N_SEEDS,
        'n_extra_features': n_extra,
        'total_min': round(total_s/60, 1),
        'kfold': {
            'n_splits': 5, 'shuffle': True, 'random_state': 18012019,
            'note': 'Official Matbench split protocol'
        },
        'models': {},
        'baselines': BASELINES,
    }
    for n, r in all_results.items():
        s['models'][n] = {
            'avg_mae': round(r['avg'], 4),
            'std_mae': round(r['std'], 4),
            'folds': [round(x, 4) for x in r['folds']],
            'params': r['params'],
        }
        if n in all_hists:
            s['models'][n]['final_train_mae'] = [
                round(h['train'][-1], 4) for h in all_hists[n]
            ]
            s['models'][n]['final_val_mae'] = [
                round(h['val'][-1], 4) for h in all_hists[n]
            ]

    with open('expt_gap_summary_v2.json', 'w') as f:
        json.dump(s, f, indent=2)
    log.info("✓ Saved: expt_gap_summary_v2.json")


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    run_benchmark()
