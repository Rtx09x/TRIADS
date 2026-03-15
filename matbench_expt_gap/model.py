"""
+=============================================================+
|  TRIADS V3 on matbench_expt_gap                             |
|  2x T4 GPU Parallel Training (auto-fallback to 1 GPU)      |
|  4 Models: Steps(16,20) x Dropout(0.15,0.20)               |
|  Proven arch: d_attn=64, d_hidden=96 | batch_size=64       |
|  FastTensorDataLoader | Clean output                        |
+=============================================================+
"""

import os, copy, json, time, logging, warnings, urllib.request
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

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
log = logging.getLogger("TRIADS-V3")

SEEDS = [42]
BATCH_SIZE = 64

BASELINES = {
    'Darwin':              0.2865,
    'Ax/SAASBO CrabNet':   0.3310,
    'MODNet v0.1.12':      0.3327,
    'AMMExpress v2020':    0.4161,
    'CrabNet':             0.4427,
    'RF-SCM/Magpie':       0.5205,
    'Dummy':               1.0280,
}
V1_BEST = {'EG-A (V1)': 0.3510, 'EG-B (V1)': 0.3616}

# Use ALL available CPU cores for PyTorch operations
torch.set_num_threads(4)      # 4 vCPUs on Kaggle
torch.set_num_interop_threads(2)  # 2 physical cores


# ======================================================================
# FAST TENSOR DATALOADER
# ======================================================================

class FastTensorDataLoader:
    """Zero-CPU DataLoader. Entire dataset in GPU VRAM."""
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
# FEATURIZER
# ======================================================================

class ExpandedFeaturizer:
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
            ElementFraction(), Stoichiometry(), ValenceOrbital(),
            IonProperty(), BandCenter(),
        ])
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

    def featurize_all(self, comps):
        out = []
        for c in tqdm(comps, desc="  Featurizing", leave=False):
            try: mg = np.array(self.ep_magpie.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)
            try: ex = np.array(self.extra_feats.featurize(c), np.float32)
            except: ex = np.zeros(self.n_extra or 200, np.float32)
            if self.n_extra is None:
                self.n_extra = len(ex)
                log.info(f"Features: {self.n_mg} Magpie + {self.n_extra} Extra + 200 Mat2Vec")
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
# MODEL — DeepHybridTRM (V13A proven architecture)
# ======================================================================

class DeepHybridTRM(nn.Module):
    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, max_steps=20, **kw):
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
            step_preds.append(self.head(y).squeeze(1))
        return step_preds if deep_supervision else step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# LOSS + UTILS
# ======================================================================

def deep_supervision_loss(step_preds, targets):
    n = len(step_preds)
    weights = [(i+1) for i in range(n)]
    tw = sum(weights)
    return sum((w/tw) * F.l1_loss(p, targets) for p, w in zip(step_preds, weights))


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


def predict(model, dl):
    model.eval(); preds = []
    with torch.no_grad():
        for bx, _ in dl:
            preds.append(model(bx).cpu())
    return torch.cat(preds)


# ======================================================================
# TRAINING — clean, simple, V1-style
# ======================================================================

def train_fold(model, tr_dl, vl_dl, device,
               epochs=300, swa_start=200, fold=1, name="", gpu_tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=swa_start, eta_min=1e-4)
    swa_m = AveragedModel(model)
    swa_s = SWALR(opt, swa_lr=5e-4)
    swa_on = False
    best_v, best_w = float('inf'), copy.deepcopy(model.state_dict())
    hist = {'train': [], 'val': []}
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    pbar = tqdm(range(epochs), desc=f"  {gpu_tag}[{name}] F{fold}/5",
                leave=False, ncols=120)
    for ep in pbar:
        model.train(); tl = 0.0
        for bx, by in tr_dl:
            with torch.amp.autocast('cuda', enabled=use_amp):
                sp = model(bx, deep_supervision=True)
                loss = deep_supervision_loss(sp, by)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tl += F.l1_loss(sp[-1], by).item() * len(by)
        tl /= tr_dl.dataset_len

        model.eval(); vl = 0.0
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                for bx, by in vl_dl:
                    vl += F.l1_loss(model(bx), by).item() * len(by)
        vl /= vl_dl.dataset_len
        hist['train'].append(tl); hist['val'].append(vl)

        if ep < swa_start:
            sch.step()
            if vl < best_v:
                best_v = vl
                best_w = copy.deepcopy(model.state_dict())
        else:
            if not swa_on: swa_on = True
            swa_m.update_parameters(model); swa_s.step()

        pbar.set_postfix(Best=f'{best_v:.4f}', Ph='SWA' if swa_on else 'COS',
                        Tr=f'{tl:.4f}', Val=f'{vl:.4f}')

    if swa_on:
        update_bn(tr_dl, swa_m, device=device)
        model.load_state_dict(swa_m.module.state_dict())
    else:
        model.load_state_dict(best_w)
    return best_v, model, hist


# ======================================================================
# GPU WORKER — trains assigned models on one GPU
# ======================================================================

def gpu_worker(gpu_id, config_list, X_all, targets_all, folds, n_extra,
               result_file):
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    tag = f"[GPU{gpu_id}] "

    print(f"\n  {tag}Started on {torch.cuda.get_device_name(gpu_id)}")
    print(f"  {tag}Models: {[c[0] for c in config_list]}")

    feat = ExpandedFeaturizer()
    results = {}

    for ci, (cname, model_kw) in enumerate(config_list):
        print(f"\n  {tag}{'='*50}")
        print(f"  {tag}[{ci+1}/{len(config_list)}] {cname}")
        print(f"  {tag}{'='*50}")

        seed = SEEDS[0]
        fold_maes = []

        for fi, (tv_i, te_i) in enumerate(folds):
            print(f"\n  {tag}-- [{cname}] Fold {fi+1}/5 " + "-"*20)

            tri, vli = strat_split(targets_all[tv_i], 0.15, seed + fi)
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
            torch.cuda.manual_seed(seed + fi)

            model = DeepHybridTRM(**model_kw).to(device)
            if fi == 0:
                print(f"  {tag}Params: {model.count_parameters():,}")

            bv, model, hist = train_fold(
                model, tr_dl, vl_dl, device,
                epochs=300, swa_start=200, fold=fi+1, name=cname, gpu_tag=tag)

            pred = predict(model, te_dl)
            mae = F.l1_loss(pred, te_y.cpu()).item()
            print(f"  {tag}Fold {fi+1} TEST: {mae:.4f} eV (val best: {bv:.4f})")

            fold_maes.append(mae)
            os.makedirs('expt_gap_models_v3', exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'test_mae': mae, 'config': cname, 'seed': seed,
                'fold': fi+1, 'n_extra': n_extra,
            }, f'expt_gap_models_v3/{cname}_s{seed}_f{fi+1}.pt')

            del model, tr_x, tr_y, vl_x, vl_y, te_x, te_y
            torch.cuda.empty_cache()

        avg = float(np.mean(fold_maes))
        std = float(np.std(fold_maes))
        results[cname] = {'avg': avg, 'std': std, 'folds': fold_maes}

        print(f"\n  {tag}=== {cname} ===")
        print(f"  {tag}    5-Fold Avg MAE: {avg:.4f} +/- {std:.4f} eV")
        print(f"  {tag}    Per-fold: {[f'{m:.4f}' for m in fold_maes]}")

    with open(result_file, 'w') as f:
        json.dump(results, f)
    print(f"\n  {tag}DONE. Saved to {result_file}")


# ======================================================================
# MAIN
# ======================================================================

def run_benchmark():
    t0 = time.time()

    print(f"""
  +==========================================================+
  |  TRIADS V3 -- P100 | FastTensorDataLoader                |
  |  4 Models: Steps(16,20) x Dropout(0.15,0.20)            |
  |  d_attn=64, d_hidden=96 (proven V1 arch)                |
  |  batch_size={BATCH_SIZE} | All CPU cores active                  |
  +==========================================================+
    """)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        try: gm = torch.cuda.get_device_properties(0).total_memory / 1e9
        except: gm = 0
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({gm:.1f} GB)")
        print(f"  CPU threads: {torch.get_num_threads()} | Interop: {torch.get_num_interop_threads()}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ---- LOAD + FEATURIZE ----
    print("\n  Loading matbench_expt_gap...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_expt_gap")
    targets_all = np.array(df['gap expt'].tolist(), np.float32)
    comps_all = [Composition(c) for c in df['composition'].tolist()]
    print(f"  Dataset: {len(comps_all)} samples")

    feat = ExpandedFeaturizer()
    X_all = feat.featurize_all(comps_all)
    n_extra = feat.n_extra
    print(f"  Features: {X_all.shape}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    for fi, (tv, te) in enumerate(folds):
        assert len(set(tv) & set(te)) == 0
    print("  5 folds verified: zero leakage")

    # ---- CONFIGS ----
    base = dict(n_props=22, stat_dim=6, n_extra=n_extra, mat2vec_dim=200,
                d_attn=64, nhead=4, d_hidden=96, ff_dim=150)

    all_configs = [
        ('V3-S16-D15', {**base, 'max_steps': 16, 'dropout': 0.15}),
        ('V3-S16-D20', {**base, 'max_steps': 16, 'dropout': 0.20}),
        ('V3-S20-D15', {**base, 'max_steps': 20, 'dropout': 0.15}),
        ('V3-S20-D20', {**base, 'max_steps': 20, 'dropout': 0.20}),
    ]

    print(f"\n  {'Config':<16} {'Params':>10} {'Steps':>6} {'Drop':>6}")
    for cn, kw in all_configs:
        m = DeepHybridTRM(**kw); print(f"  {cn:<16} {m.count_parameters():>10,} {kw['max_steps']:>6} {kw['dropout']:>6.2f}"); del m

    # ---- TRAIN ----
    all_results = {}

    for ci, (cname, model_kw) in enumerate(all_configs):
        print(f"\n  {'='*60}")
        print(f"  [{ci+1}/4] {cname}")
        print(f"  {'='*60}")

        seed = SEEDS[0]
        fold_maes = []

        for fi, (tv_i, te_i) in enumerate(folds):
            print(f"\n  -- [{cname}] Fold {fi+1}/5 " + "-"*30)
            tri, vli = strat_split(targets_all[tv_i], 0.15, seed + fi)
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

            torch.manual_seed(seed + fi); np.random.seed(seed + fi)
            if device.type == 'cuda': torch.cuda.manual_seed(seed + fi)

            model = DeepHybridTRM(**model_kw).to(device)
            if fi == 0: print(f"  Params: {model.count_parameters():,}")

            bv, model, hist = train_fold(model, tr_dl, vl_dl, device,
                epochs=300, swa_start=200, fold=fi+1, name=cname)

            pred = predict(model, te_dl)
            mae = F.l1_loss(pred, te_y.cpu()).item()
            print(f"  Fold {fi+1} TEST: {mae:.4f} eV (val: {bv:.4f})")
            fold_maes.append(mae)

            os.makedirs('expt_gap_models_v3', exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'test_mae': mae, 'config': cname, 'seed': seed,
                'fold': fi+1, 'n_extra': n_extra,
            }, f'expt_gap_models_v3/{cname}_s{seed}_f{fi+1}.pt')

            del model, tr_x, tr_y, vl_x, vl_y, te_x, te_y
            if device.type == 'cuda': torch.cuda.empty_cache()

        avg = float(np.mean(fold_maes))
        std = float(np.std(fold_maes))
        all_results[cname] = {'avg': avg, 'std': std, 'folds': fold_maes}
        print(f"\n  === {cname}: {avg:.4f} +/- {std:.4f} eV ===")

    # ======== FINAL RESULTS ========
    tt = time.time() - t0
    print(f"\n{'='*72}")
    print(f"  FINAL LEADERBOARD -- TRIADS V3 (5-Fold Avg MAE, eV)")
    print(f"{'='*72}")
    print(f"  {'Model':<20} {'MAE':>10} {'Std':>8}  Notes")
    print(f"  {'-'*60}")

    for n, r in sorted(all_results.items(), key=lambda x: x[1]['avg']):
        tag = (" <-- DARWIN BEATEN!" if r['avg'] < 0.2865 else
               " <-- Top 3!"        if r['avg'] < 0.3327 else
               " <-- Beats V1!"     if r['avg'] < 0.3510 else
               " <-- Beats AMMExp"  if r['avg'] < 0.4161 else "")
        print(f"  {n:<20} {r['avg']:>10.4f} {r['std']:>8.4f}{tag}")

    print(f"  {'-'*60}")
    for vn, vm in sorted(V1_BEST.items(), key=lambda x: x[1]):
        print(f"  {vn:<20} {vm:>10.4f}  (V1)")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<20} {bv:>10.4f}")

    # Per-fold
    names = sorted(all_results.keys())
    print(f"\n  PER-FOLD:")
    hdr = f"  {'Fold':<6}"; [hdr := hdr + f"  {cn:>14}" for cn in names]
    print(hdr)
    for fi in range(5):
        row = f"  F{fi+1:<5}"; [row := row + f"  {all_results[cn]['folds'][fi]:>14.4f}" for cn in names]
        print(row)

    print(f"\n  HP GRID:  {'D=0.15':>10} {'D=0.20':>10}")
    for s in [16, 20]:
        d15 = all_results.get(f'V3-S{s}-D15', {}).get('avg', 0)
        d20 = all_results.get(f'V3-S{s}-D20', {}).get('avg', 0)
        print(f"  S={s:>2}   {d15:>10.4f}  {d20:>10.4f}")

    print(f"\n  Total: {tt/60:.1f} min")

    s = {'version': 'EG-V3', 'batch_size': BATCH_SIZE,
         'total_min': round(tt/60, 1), 'models': all_results,
         'baselines': BASELINES, 'v1': V1_BEST}
    with open('expt_gap_summary_v3.json', 'w') as f:
        json.dump(s, f, indent=2)
    print("  Saved: expt_gap_summary_v3.json")


if __name__ == '__main__':
    run_benchmark()
