"""
+=============================================================+
|  TRIADS V4 on matbench_jdft2d — 5-Seed Ensemble            |
|  Exfoliation Energy (meV/atom) — 636 samples                |
|                                                             |
|  Structural + Composition features (~361d)                  |
|  75K model (d_attn=32, d_hidden=64) | dropout=0.20          |
|  Seeds: [42, 123, 456, 789, 1024]                           |
|  Target: Kaggle P100 | ~30 min                              |
+=============================================================+
"""

import os, copy, json, time, logging, warnings, urllib.request, shutil
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.composition import ElementProperty
from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
log = logging.getLogger("TRIADS-jdft2d")

BATCH_SIZE = 64
SEEDS = [42, 123, 456, 789, 1024]

# 75K config — best for 636 samples
MODEL_CFG = dict(
    d_attn=32, nhead=4, d_hidden=64, ff_dim=96,
    dropout=0.20, max_steps=16,
)

V1_BEST = {'V1 (100K, comp-only)': 45.8045}
V2_BEST = {'V2 (44K, comp-only)': 46.5889}
V3_BEST = {'V3 (75K, +struct, single)': 37.0033}


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
# FEATURIZER — Composition + Structural (~361d)
# ======================================================================

def _extract_structural_features(structure):
    feats = []
    try:
        lat = structure.lattice
        feats.extend([lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma])
        feats.append(structure.volume / max(len(structure), 1))
        feats.append(structure.density)
        feats.append(float(len(structure)))
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            feats.append(float(sga.get_space_group_number()))
        except:
            feats.append(0.0)
        try:
            total_vol = sum(
                (4/3) * np.pi * site.specie.atomic_radius**3
                for site in structure if hasattr(site.specie, 'atomic_radius')
                and site.specie.atomic_radius is not None
            )
            feats.append(total_vol / structure.volume if structure.volume > 0 else 0.0)
        except:
            feats.append(0.0)
    except:
        feats = [0.0] * 11
    return np.array(feats, dtype=np.float32)


class ExfoliationFeaturizer:
    GCS = "https://storage.googleapis.com/mat2vec/"
    FILES = ["pretrained_embeddings",
             "pretrained_embeddings.wv.vectors.npy",
             "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache="mat2vec_cache"):
        from matminer.featurizers.composition import (
            Stoichiometry, ValenceOrbital, IonProperty
        )
        from matminer.featurizers.composition.element import TMetalFraction

        self.ep_magpie = ElementProperty.from_preset("magpie")
        self.n_mg = len(self.ep_magpie.feature_labels())

        self.extra_featurizers = [
            ("Stoichiometry",  Stoichiometry()),
            ("ValenceOrbital", ValenceOrbital()),
            ("IonProperty",    IonProperty()),
            ("TMetalFraction", TMetalFraction()),
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

    def _featurize_extra(self, comp, structure=None):
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
        if structure is not None:
            parts.append(_extract_structural_features(structure))
        else:
            parts.append(np.zeros(11, np.float32))
        return np.concatenate(parts)

    def featurize_all(self, comps, structures=None):
        out = []
        test_struct = structures[0] if structures else None
        test_ex = self._featurize_extra(comps[0], test_struct)
        self.n_extra = len(test_ex)
        total = self.n_mg + self.n_extra + 200
        comp_extras = sum(self._extra_sizes.get(n, 0) or 0
                         for n, _ in self.extra_featurizers)
        log.info(f"Features: {self.n_mg} Magpie + {comp_extras} CompExtra + "
                 f"11 Structural + 200 Mat2Vec = {total}d")
        for i, c in enumerate(tqdm(comps, desc="  Featurizing", leave=False)):
            struct = structures[i] if structures else None
            try: mg = np.array(self.ep_magpie.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)
            ex = self._featurize_extra(c, struct)
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
# MODEL
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
    preds = torch.stack(step_preds)
    n = preds.shape[0]
    w = torch.arange(1, n + 1, device=preds.device, dtype=preds.dtype)
    w = w / w.sum()
    per_step = (preds - targets.unsqueeze(0)).abs().mean(dim=1)
    return (w * per_step).sum()


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


@torch.inference_mode()
def predict(model, dl):
    model.eval()
    preds = []
    for bx, _ in dl:
        preds.append(model(bx).cpu())
    return torch.cat(preds)


# ======================================================================
# TRAINING
# ======================================================================

def train_fold(model, tr_dl, vl_dl, device,
               epochs=300, swa_start=200, fold=1, seed=42):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=swa_start, eta_min=1e-4)
    swa_m = AveragedModel(model)
    swa_s = SWALR(opt, swa_lr=5e-4)
    swa_on = False
    best_v, best_w = float('inf'), None

    pbar = tqdm(range(epochs), desc=f"  [75K|s{seed}] F{fold}/5",
                leave=False, ncols=120)
    for ep in pbar:
        model.train()
        epoch_loss = torch.tensor(0.0, device=device)
        n_samples = 0

        for bx, by in tr_dl:
            sp = model(bx, deep_supervision=True)
            loss = deep_supervision_loss(sp, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                epoch_loss += (sp[-1] - by).abs().sum()
            n_samples += len(by)

        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_n = 0
        with torch.inference_mode():
            for bx, by in vl_dl:
                val_loss += (model(bx) - by).abs().sum()
                val_n += len(by)

        tl = epoch_loss.item() / n_samples
        vl = val_loss.item() / val_n

        if ep < swa_start:
            sch.step()
            if vl < best_v:
                best_v = vl
                best_w = copy.deepcopy(model.state_dict())
        else:
            if not swa_on: swa_on = True
            swa_m.update_parameters(model); swa_s.step()

        if ep % 10 == 0 or ep == epochs - 1:
            pbar.set_postfix(Best=f'{best_v:.2f}', Ph='SWA' if swa_on else 'COS',
                            Tr=f'{tl:.2f}', Val=f'{vl:.2f}')

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
  |  TRIADS V4 — matbench_jdft2d (5-Seed Ensemble)          |
  |  Structural + Composition features (~361d)               |
  |  75K model | dropout=0.20                                |
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
    print("\n  Loading matbench_jdft2d...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_jdft2d")
    targets_all = np.array(df['exfoliation_en'].tolist(), np.float32)
    structures_all = df['structure'].tolist()
    comps_all = [s.composition for s in structures_all]
    print(f"  Dataset: {len(comps_all)} samples")

    # ── FEATURIZE (once) ─────────────────────────────────────────────
    t_feat = time.time()
    feat = ExfoliationFeaturizer()
    X_all = feat.featurize_all(comps_all, structures_all)
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
    print(f"  Config: d_attn={MODEL_CFG['d_attn']}, d_hidden={MODEL_CFG['d_hidden']}, "
          f"ff_dim={MODEL_CFG['ff_dim']}, dropout={MODEL_CFG['dropout']}\n")

    # ── TRAIN ALL SEEDS ──────────────────────────────────────────────
    model_dir = 'jdft2d_models_v4'
    os.makedirs(model_dir, exist_ok=True)

    # Store predictions and MAEs per seed
    all_seed_maes = {}        # {seed: {fold: mae}}
    all_fold_preds = {}       # {fold: {seed: predictions}}
    all_fold_targets = {}     # {fold: targets}

    for seed in SEEDS:
        print(f"\n  {'─'*3} Seed {seed} {'─'*40}")
        t_seed = time.time()
        seed_maes = {}

        for fi, (tv_i, te_i) in enumerate(folds):
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
            if device.type == 'cuda': torch.cuda.manual_seed(seed + fi)

            model = DeepHybridTRM(**model_kw).to(device)
            bv, model = train_fold(model, tr_dl, vl_dl, device,
                                    epochs=300, swa_start=200,
                                    fold=fi+1, seed=seed)

            pred = predict(model, te_dl)
            mae = F.l1_loss(pred, te_y.cpu()).item()
            seed_maes[fi] = mae

            # Store for ensemble
            if fi not in all_fold_preds:
                all_fold_preds[fi] = {}
                all_fold_targets[fi] = te_y.cpu()
            all_fold_preds[fi][seed] = pred

            torch.save({
                'model_state': model.state_dict(),
                'test_mae': mae, 'fold': fi+1, 'seed': seed,
                'n_extra': n_extra,
            }, f'{model_dir}/jdft2d_75K_s{seed}_f{fi+1}.pt')

            del model, tr_x, tr_y, vl_x, vl_y, te_x, te_y
            if device.type == 'cuda': torch.cuda.empty_cache()

        avg_s = np.mean(list(seed_maes.values()))
        all_seed_maes[seed] = seed_maes
        dt = time.time() - t_seed
        print(f"\n  Seed {seed}: avg={avg_s:.4f} | "
              f"{[f'{seed_maes[i]:.4f}' for i in range(5)]} ({dt:.0f}s)")

    # ── ENSEMBLE ─────────────────────────────────────────────────────
    ens_maes = {}
    for fi in range(5):
        preds_stack = torch.stack([all_fold_preds[fi][s] for s in SEEDS])
        ens_pred = preds_stack.mean(dim=0)
        ens_maes[fi] = F.l1_loss(ens_pred, all_fold_targets[fi]).item()

    single_avgs = [np.mean(list(all_seed_maes[s].values())) for s in SEEDS]
    single_mean = np.mean(single_avgs)
    single_std = np.std(single_avgs)
    ens_mean = np.mean(list(ens_maes.values()))
    ens_std = np.std(list(ens_maes.values()))
    ens_drop = (1 - ens_mean / single_mean) * 100

    # ── RESULTS ──────────────────────────────────────────────────────
    tt = time.time() - t0

    print(f"""
{'='*72}
  FINAL RESULTS — TRIADS V4 on matbench_jdft2d
{'='*72}

  Per-seed results:""")

    for seed in SEEDS:
        sm = all_seed_maes[seed]
        avg_s = np.mean(list(sm.values()))
        print(f"    Seed {seed:>4}: {avg_s:.4f} | "
              f"{[f'{sm[i]:.4f}' for i in range(5)]}")

    print(f"""
    Single-seed avg: {single_mean:.4f} ± {single_std:.4f}
    5-Seed Ensemble: {ens_mean:.4f} ± {ens_std:.4f} (↓{ens_drop:.1f}% from single)
    Per-fold ens:    {[f'{ens_maes[i]:.4f}' for i in range(5)]}

  {'Model':<40} {'MAE(meV/atom)':>15}
  {'─'*58}
  {'MODNet v0.1.12':<40} {'33.1918':>15}
  {'TRIADS V3 (75K, +struct, single)':<40} {'37.0033':>15}
  {'TRIADS V4 (75K, +struct, 5-seed ens)':<40} {f'{ens_mean:.4f}':>15} ← NEW
  {'TRIADS V1 (100K, comp-only)':<40} {'45.8045':>15}
  {'─'*58}

  Total time: {tt/60:.1f} min
  Saved: {model_dir}/
""")

    # ── SAVE ─────────────────────────────────────────────────────────
    summary = {
        'version': 'jdft2d-V4-ensemble',
        'dataset': 'matbench_jdft2d',
        'samples': len(comps_all),
        'target_unit': 'meV/atom',
        'model_config': MODEL_CFG,
        'params': n_params,
        'seeds': SEEDS,
        'per_seed': {str(s): {str(k): round(v, 4) for k, v in m.items()}
                     for s, m in all_seed_maes.items()},
        'single_seed_avg': round(single_mean, 4),
        'single_seed_std': round(single_std, 4),
        'ensemble_maes': {str(k): round(v, 4) for k, v in ens_maes.items()},
        'ensemble_avg': round(ens_mean, 4),
        'ensemble_std': round(ens_std, 4),
        'ensemble_improvement': f'{ens_drop:.1f}%',
        'total_time_min': round(tt/60, 1),
    }
    with open('jdft2d_summary_v4.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Saved: jdft2d_summary_v4.json")

    # Zip models
    shutil.make_archive(model_dir, 'zip', '.', model_dir)
    print(f"  Saved: {model_dir}.zip (download this!)")


if __name__ == '__main__':
    run_benchmark()
