"""
╔══════════════════════════════════════════════════════════════════════╗
║       TRM-MatSci v2 │ Tiny Recursive Model for Materials Science     ║
║       Dataset  : MatBench Steels (Yield Strength Prediction)         ║
║       Benchmark: matbench_v0.1 │ 5-Fold Nested CV                    ║
║       Models   : 12 total — 6 MLP-TRM + 6 Transformer-TRM            ║
║       Hardware : Kaggle P100 / T4 / Google Colab                     ║
╚══════════════════════════════════════════════════════════════════════╝

ARCHITECTURE OVERVIEW
─────────────────────
Both families share the same dual-state recursive TRM structure:

    z = latent reasoning state   (what the model is "thinking")
    y = current answer state     (best prediction so far)

    Every recursive step (shared weights):
        z_new = UPDATE_Z( concat(x_proj, y, z) )
        y_new = UPDATE_Y( concat(y, z_new)     )

    MLP-TRM   → UPDATE_Z and UPDATE_Y are 2-layer MLPs
    Trans-TRM → UPDATE_Z and UPDATE_Y use self-attention + FFN

    Dropout fires at EVERY step inside both update functions.
    Gradient clipping prevents exploding gradients from deep unrolling.

12 CONFIGURATIONS
─────────────────
    MLP-TRM-10K-h64   │ MLP-TRM-10K-h128
    MLP-TRM-50K-h64   │ MLP-TRM-50K-h128
    MLP-TRM-100K-h64  │ MLP-TRM-100K-h128

    Trans-TRM-10K-h64  │ Trans-TRM-10K-h128
    Trans-TRM-50K-h64  │ Trans-TRM-50K-h128
    Trans-TRM-100K-h64 │ Trans-TRM-100K-h128

NOTE ON PARAMETER TARGETS
──────────────────────────
The input_proj layer (Mat2Vec 200-dim → hidden) costs ~23K params on its own.
So "10K target" means the core network adds ~10K on top of that — total ~33K.
This is fine and expected. We don't stress exact targets; we just ensure the
core reasoning network itself stays in the right ballpark.

OUTPUTS
───────
    trm_results_v2.png  — 4-panel benchmark + scaling + convergence + head-to-head
    trm_models/         — all 12 × 5 = 60 fold models saved as .pt files
                          + summary.json + README.md (HuggingFace ready)

INSTALL (run once on fresh Kaggle / Colab):
    !pip install matminer pymatgen torch tqdm matplotlib gensim scikit-learn -q
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, copy, json, math, time, logging, warnings, urllib.request
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

from pymatgen.core import Composition
from sklearn.model_selection import train_test_split, KFold
from gensim.models import Word2Vec

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SEED  — must be set before anything else
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s │ %(levelname)s │ %(message)s',
    datefmt = '%H:%M:%S'
)
log = logging.getLogger("TRM-MatSci")

# ─────────────────────────────────────────────────────────────────────────────
# LEADERBOARD BASELINES
# ─────────────────────────────────────────────────────────────────────────────
BASELINES = {
    'TPOT-Mat (best)':   79.9468,
    'AutoML-Mat':        82.3043,
    'MODNet v0.1.12':    87.7627,
    'RF-Regex Steels':   90.5896,
    'MODNet v0.1.10':    96.2139,
    'RF-SCM/Magpie':    103.5125,
    'CrabNet':          107.3160,
    'Dummy baseline':   229.7445,
}

# ─────────────────────────────────────────────────────────────────────────────
# VALID STEEL ELEMENTS  (anything outside this is rejected)
# ─────────────────────────────────────────────────────────────────────────────
VALID_STEEL_ELEMENTS = {
    'Fe','C','Mn','Si','Cr','Ni','Mo','V','W','Co','Cu','Ti','Nb',
    'Al','N','S','P','B','Zr','Ta','Re','Hf','Se','Te','Ca','Mg','Sn'
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MAT2VEC FEATURIZER
# ══════════════════════════════════════════════════════════════════════════════

class Mat2VecFeaturizer:
    """
    Downloads the REAL pre-trained Mat2Vec Word2Vec model from the official 
    Google Cloud Storage bucket (materialsintelligence/mat2vec repo).
    Produces true 200-dim element embeddings. No fallbacks.

    Files downloaded (cached locally):
        pretrained_embeddings                          (gensim model file)
        pretrained_embeddings.wv.vectors.npy            (word vectors)
        pretrained_embeddings.trainables.syn1neg.npy    (output embeddings)
    """

    GCS_BASE = "https://storage.googleapis.com/mat2vec/"
    FILES = [
        "pretrained_embeddings",
        "pretrained_embeddings.wv.vectors.npy",
        "pretrained_embeddings.trainables.syn1neg.npy",
    ]

    def __init__(self, cache_dir="mat2vec_cache"):
        self.cache_dir = cache_dir
        self.w2v_model = self._download_and_load()
        self.embedding_dim = 200

        # Build element lookup: symbol -> 200-dim vector
        self.embeddings = {}
        from pymatgen.core import Element
        for el in Element:
            sym = el.symbol
            if sym in self.w2v_model.wv:
                self.embeddings[sym] = np.array(self.w2v_model.wv[sym], dtype=np.float32)

        log.info(f"\u2713 Mat2Vec loaded: {len(self.embeddings)} elements, dim={self.embedding_dim}")
        if len(self.embeddings) < 20:
            raise RuntimeError(
                f"Mat2Vec only found {len(self.embeddings)} elements. "
                f"Download may have failed. Delete '{cache_dir}/' and retry."
            )

    def _download_and_load(self) -> Word2Vec:
        os.makedirs(self.cache_dir, exist_ok=True)
        for fname in self.FILES:
            fpath = os.path.join(self.cache_dir, fname)
            if not os.path.exists(fpath):
                url = self.GCS_BASE + fname
                log.info(f"Downloading {fname} from GCS...")
                urllib.request.urlretrieve(url, fpath)
                log.info(f"  -> Saved to {fpath}")
        model_path = os.path.join(self.cache_dir, "pretrained_embeddings")
        log.info("Loading Word2Vec model...")
        return Word2Vec.load(model_path)

    def validate_steel(self, comp: Composition) -> bool:
        """True only if composition looks like a real steel."""
        syms = {str(el) for el in comp.elements}
        if 'Fe' not in syms:                          return False
        if comp.get_atomic_fraction('Fe') < 0.50:     return False
        if not syms.issubset(VALID_STEEL_ELEMENTS):   return False
        return True

    def featurize(self, comp: Composition) -> np.ndarray:
        """Fraction-weighted sum of element embeddings -> 1D vector (200-dim)."""
        vec   = np.zeros(self.embedding_dim, dtype=np.float32)
        total = 0.0
        for sym, frac in comp.get_el_amt_dict().items():
            if sym in self.embeddings:
                vec   += frac * self.embeddings[sym]
                total += frac
        if total > 1e-8:
            vec /= total
        return vec


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SteelsDataset(Dataset):
    def __init__(self, inputs_series, outputs_series,
                 featurizer: Mat2VecFeaturizer, validate: bool = True):
        self.featurizer = featurizer
        self.data: list = []
        skipped = 0
        for formula, target in zip(inputs_series.tolist(), outputs_series.tolist()):
            try:
                comp = Composition(formula)
                if validate and not featurizer.validate_steel(comp):
                    skipped += 1
                    continue
                vec = featurizer.featurize(comp)
                self.data.append((
                    torch.tensor(vec,           dtype=torch.float32),
                    torch.tensor(float(target), dtype=torch.float32)
                ))
            except Exception as ex:
                log.debug(f"Skipped '{formula}': {ex}")
                skipped += 1
        if skipped:
            log.warning(f"  Dataset: {len(self.data)} kept, {skipped} skipped")

    def __len__(self):              return len(self.data)
    def __getitem__(self, idx):     return self.data[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 3A.  MLP-TRM
# ══════════════════════════════════════════════════════════════════════════════

class MLPTinyRecursiveModel(nn.Module):
    """
    Dual-state TRM with MLP core.

    UPDATE_Z and UPDATE_Y are both 2-layer MLPs with LayerNorm + GELU + Dropout.
    Dropout fires at every recursive step → strong regularisation for tiny datasets.
    """

    def __init__(self, input_dim=200, hidden_dim=64, ff_dim=128,
                 dropout=0.1, recursion_steps=16):
        super().__init__()
        self.recursion_steps = recursion_steps
        self.hidden_dim      = hidden_dim
        self.arch_type       = 'MLP'

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # z update: receives concat(x_proj, y, z) → hidden*3 in
        self.z_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # y update: receives concat(y, z_new) → hidden*2 in
        self.y_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.output_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_trajectory=False):
        B      = x.size(0)
        x_proj = self.input_proj(x)
        z      = torch.zeros(B, self.hidden_dim, device=x.device)
        y      = torch.zeros(B, self.hidden_dim, device=x.device)
        traj   = []

        for _ in range(self.recursion_steps):
            z = z + self.z_update(torch.cat([x_proj, y, z], dim=-1))
            y = y + self.y_update(torch.cat([y, z],           dim=-1))
            if return_trajectory:
                traj.append(self.output_head(y).squeeze(1))

        out = self.output_head(y).squeeze(1)
        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 3B.  TRANSFORMER-TRM
# ══════════════════════════════════════════════════════════════════════════════

class TransformerTinyRecursiveModel(nn.Module):
    """
    Dual-state TRM with Transformer core.

    UPDATE_Z uses a single self-attention layer (multi-head) over a 3-token
    sequence [x_proj, y, z] — so attention can directly model relationships
    between the input, current answer and current reasoning state.

    UPDATE_Y uses a cross-attention layer where y attends to z.

    Dropout fires at every recursive step inside both update functions.

    Why this is interesting vs MLP-TRM:
    - MLP mixes all three states through learned linear combinations
    - Transformer explicitly lets each state *attend* to the others
    - For small datasets, MLP often wins; for richer inputs Transformer may win
    - Comparing both is one of the paper's main contributions
    """

    def __init__(self, input_dim=200, hidden_dim=64, ff_dim=128,
                 nhead=4, dropout=0.1, recursion_steps=16):
        super().__init__()
        self.recursion_steps = recursion_steps
        self.hidden_dim      = hidden_dim
        self.arch_type       = 'Transformer'

        # nhead must divide hidden_dim evenly
        while hidden_dim % nhead != 0 and nhead > 1:
            nhead -= 1
        self.nhead = nhead

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # ── z update via self-attention over [x_proj, y, z] (3 tokens) ──
        self.z_attn = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = nhead,
            dropout     = dropout,
            batch_first = True
        )
        self.z_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.z_norm1 = nn.LayerNorm(hidden_dim)
        self.z_norm2 = nn.LayerNorm(hidden_dim)

        # ── y update via cross-attention (y attends to z) ─────────────
        self.y_cross_attn = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = nhead,
            dropout     = dropout,
            batch_first = True
        )
        self.y_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.y_norm1 = nn.LayerNorm(hidden_dim)
        self.y_norm2 = nn.LayerNorm(hidden_dim)

        self.output_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_trajectory=False):
        B      = x.size(0)
        x_proj = self.input_proj(x)
        z      = torch.zeros(B, self.hidden_dim, device=x.device)
        y      = torch.zeros(B, self.hidden_dim, device=x.device)
        traj   = []

        for _ in range(self.recursion_steps):

            # ── z update ────────────────────────────────────────────
            # Stack 3 tokens: [x_proj, y, z] → shape (B, 3, hidden)
            seq    = torch.stack([x_proj, y, z], dim=1)
            attn_z, _ = self.z_attn(seq, seq, seq)
            # z token is at index 2 — take it and apply residual + FFN
            z_res  = self.z_norm1(z + attn_z[:, 2, :])
            z      = self.z_norm2(z_res + self.z_ffn(z_res))

            # ── y update ────────────────────────────────────────────
            # y (query) cross-attends to z (key/value) → (B, 1, hidden)
            y_q    = y.unsqueeze(1)
            z_kv   = z.unsqueeze(1)
            ca_y, _ = self.y_cross_attn(y_q, z_kv, z_kv)
            y_res  = self.y_norm1(y + ca_y.squeeze(1))
            y      = self.y_norm2(y_res + self.y_ffn(y_res))

            if return_trajectory:
                traj.append(self.output_head(y).squeeze(1))

        out = self.output_head(y).squeeze(1)
        return (out, traj) if return_trajectory else out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL BUILDER  (searches ff_dim to approach a target param count)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(arch: str, target_params: int, hidden_dim: int,
                input_dim: int = 200, recursion_steps: int = 16,
                dropout: float = 0.1) -> tuple:
    """
    Searches ff_dim to produce a model with actual params ≈ target_params.
    NOTE: The input_proj layer (~23K) is included in the count.
          For a 10K target the core will be small but input_proj dominates — this is fine.

    Returns:
        model         — instantiated model
        ff_dim        — feedforward dim used
        actual_params — true param count
    """
    assert arch in ('MLP', 'Transformer'), f"Unknown arch: {arch}"

    best_model  = None
    best_ff     = 16
    best_diff   = float('inf')

    nhead = 4 if hidden_dim % 4 == 0 else 2 if hidden_dim % 2 == 0 else 1

    for ff in range(16, 32_768, 8):
        if arch == 'MLP':
            m = MLPTinyRecursiveModel(input_dim, hidden_dim, ff, dropout, recursion_steps)
        else:
            m = TransformerTinyRecursiveModel(input_dim, hidden_dim, ff, nhead, dropout, recursion_steps)

        p    = m.count_parameters()
        diff = abs(p - target_params)

        if diff < best_diff:
            best_diff  = diff
            best_model = m
            best_ff    = ff

        if p > target_params:
            break

    return best_model, best_ff, best_model.count_parameters()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(model, train_loader, val_loader, device,
               epochs=200, fold_idx=1, config_name="") -> tuple:
    """
    Train for one fold. Saves best checkpoint internally, restores at end.

    Returns:
        best_val_mae  — best val MAE seen during training
        final_model   — model with best weights loaded
        history       — {'train': [...], 'val': [...]} per epoch
    """
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    best_val_mae = float('inf')
    best_weights = copy.deepcopy(model.state_dict())  # init with starting weights
    history      = {'train': [], 'val': []}

    pbar = tqdm(range(epochs),
                desc=f"  [{config_name}] F{fold_idx}/5",
                leave=False, ncols=115)

    for epoch in pbar:

        # ── train ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss  = F.l1_loss(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        # ── validate ─────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += F.l1_loss(model(bx), by).item() * len(bx)
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val_mae:
            best_val_mae = val_loss
            best_weights = copy.deepcopy(model.state_dict())

        pbar.set_postfix({
            'Tr':   f'{train_loss:.1f}',
            'Val':  f'{val_loss:.1f}',
            'Best': f'{best_val_mae:.1f}',
            'LR':   f'{scheduler.get_last_lr()[0]:.2e}'
        })

    # Restore best weights
    model.load_state_dict(best_weights)
    return best_val_mae, model, history


# ══════════════════════════════════════════════════════════════════════════════
# 6.  RECURSION STEP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_recursion_trajectory(model, dataloader, device) -> list:
    """Returns list of MAE per recursion step — unique to TRM family."""
    model.eval()
    step_preds = None
    all_targets = []

    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            _, traj = model(bx, return_trajectory=True)

            if step_preds is None:
                step_preds = [[] for _ in range(len(traj))]
            for i, sp in enumerate(traj):
                step_preds[i].extend(sp.cpu().numpy().tolist())
            all_targets.extend(by.numpy().tolist())

    targets = np.array(all_targets)
    return [
        float(np.mean(np.abs(np.array(p) - targets)))
        for p in step_preds
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark():

    t_total_start = time.time()

    print("\n" + "═" * 72)
    print("   TRM-MatSci v2 │ 12-Model Ablation │ MLP + Transformer")
    print("   Dataset: matbench_steels │ 312 samples │ 5-Fold Nested CV")
    print("═" * 72 + "\n")

    # ── device ─────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {name}  ({mem:.1f} GB VRAM)")
    else:
        log.info("Device: CPU")

    # ── dataset (uses matminer, NOT matbench) ──────────────────────────
    log.info("Loading matbench_steels dataset via matminer...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_steels")
    # The dataset has columns: 'composition' (str) and 'yield strength' (float)
    compositions = df['composition']
    targets = df['yield strength']
    log.info(f"Dataset loaded: {len(df)} samples")

    # ── Reproduce EXACT matbench 5-fold splits ─────────────────────────
    # matbench internally uses: KFold(n_splits=5, shuffle=True, random_state=18012019)
    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(df))
    log.info(f"Reproduced {len(folds)} matbench-identical folds\n")

    # ── featurizer ─────────────────────────────────────────────────────
    log.info("Initialising Mat2Vec featurizer...")
    featurizer = Mat2VecFeaturizer()
    input_dim  = featurizer.embedding_dim

    # ── output directory ───────────────────────────────────────────────
    os.makedirs('trm_models', exist_ok=True)

    # ── 12 configurations ──────────────────────────────────────────────
    # (arch, target_params, hidden_dim, config_name)
    configs = [
        # ── MLP family ──────────────────────────────────────────────
        ('MLP',         10_000,  64,  'MLP-TRM-10K-h64'),
        ('MLP',         10_000, 128,  'MLP-TRM-10K-h128'),
        ('MLP',         50_000,  64,  'MLP-TRM-50K-h64'),
        ('MLP',         50_000, 128,  'MLP-TRM-50K-h128'),
        ('MLP',        100_000,  64,  'MLP-TRM-100K-h64'),
        ('MLP',        100_000, 128,  'MLP-TRM-100K-h128'),
        # ── Transformer family ──────────────────────────────────────
        ('Transformer',  10_000,  64,  'Trans-TRM-10K-h64'),
        ('Transformer',  10_000, 128,  'Trans-TRM-10K-h128'),
        ('Transformer',  50_000,  64,  'Trans-TRM-50K-h64'),
        ('Transformer',  50_000, 128,  'Trans-TRM-50K-h128'),
        ('Transformer', 100_000,  64,  'Trans-TRM-100K-h64'),
        ('Transformer', 100_000, 128,  'Trans-TRM-100K-h128'),
    ]

    all_results      = {}
    all_trajectories = {}

    # ── loop over configs ──────────────────────────────────────────────
    for cfg_idx, (arch, target_p, hidden_dim, config_name) in enumerate(configs):

        t_cfg_start = time.time()

        print(f"\n{'═'*72}")
        print(f"  [{cfg_idx+1:02d}/12]  {config_name}")
        print(f"  arch={arch}  target={target_p:,}  hidden={hidden_dim}  steps=16")
        print(f"{'─'*72}")

        _, ff_dim, actual_p = build_model(arch, target_p, hidden_dim, input_dim)
        log.info(f"Actual params: {actual_p:,}  (ff_dim={ff_dim})")

        fold_maes         = []
        fold_trajectories = []

        for fold_idx, (train_val_indices, test_indices) in enumerate(folds):
            torch.manual_seed(SEED + fold_idx)

            # ── data ──────────────────────────────────────────────────
            # 1. Exact matbench splits
            train_val_inputs  = compositions.iloc[train_val_indices]
            train_val_outputs = targets.iloc[train_val_indices]
            test_inputs       = compositions.iloc[test_indices]
            test_outputs      = targets.iloc[test_indices]

            # 2. Prevent Data Leakage: Split train further into train (85%) and val (15%) for early-stopping
            X_train, X_val, y_train, y_val = train_test_split(
                train_val_inputs, train_val_outputs, test_size=0.15, random_state=SEED+fold_idx
            )

            train_ds = SteelsDataset(X_train, y_train, featurizer, validate=True)
            val_ds   = SteelsDataset(X_val,   y_val,   featurizer, validate=False)
            test_ds  = SteelsDataset(test_inputs, test_outputs, featurizer, validate=False)

            train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
            val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False)
            test_dl  = DataLoader(test_ds,  batch_size=16, shuffle=False)

            # ── fresh model ───────────────────────────────────────────
            nhead = 4 if hidden_dim % 4 == 0 else 2
            if arch == 'MLP':
                model = MLPTinyRecursiveModel(
                    input_dim, hidden_dim, ff_dim, 0.1, 16).to(device)
            else:
                model = TransformerTinyRecursiveModel(
                    input_dim, hidden_dim, ff_dim, nhead, 0.1, 16).to(device)

            # ── train ─────────────────────────────────────────────────
            best_val_mae, trained_model, history = train_fold(
                model, train_dl, val_dl, device,
                epochs=200, fold_idx=fold_idx+1, config_name=config_name)

            # ── blind test inference (No Data Leakage) ───────────────
            trained_model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for bx, by in test_dl:
                    bx, by = bx.to(device), by.to(device)
                    test_loss += F.l1_loss(trained_model(bx), by).item() * len(bx)
            test_mae = test_loss / len(test_dl.dataset)

            log.info(f"  Fold {fold_idx+1}/5 -> Test MAE: {test_mae:.4f} MPa (Val was {best_val_mae:.4f})")
            fold_maes.append(test_mae)

            # ── recursion trajectory ──────────────────────────────────
            traj = evaluate_recursion_trajectory(trained_model, test_dl, device)
            fold_trajectories.append(traj)

            # ── SAVE THIS FOLD'S MODEL ────────────────────────────────
            # We save every fold of every model — all 60 checkpoints
            save_path = f'trm_models/{config_name}_fold{fold_idx+1}.pt'
            torch.save({
                'config_name':     config_name,
                'arch':            arch,
                'fold':            fold_idx + 1,
                'input_dim':       input_dim,
                'hidden_dim':      hidden_dim,
                'ff_dim':          ff_dim,
                'recursion_steps': 16,
                'actual_params':   actual_p,
                'test_mae':        test_mae,
                'val_mae':         best_val_mae,
                'model_state':     trained_model.state_dict(),
            }, save_path)

        # ── aggregate ─────────────────────────────────────────────────
        avg_mae = float(np.mean(fold_maes))
        std_mae = float(np.std(fold_maes))
        t_cfg   = time.time() - t_cfg_start

        tag = ""
        if avg_mae < BASELINES['TPOT-Mat (best)']:   tag = "  ← BEATS TPOT 🏆"
        elif avg_mae < BASELINES['AutoML-Mat']:       tag = "  ← Beats AutoML ✓"
        elif avg_mae < BASELINES['MODNet v0.1.12']:   tag = "  ← Beats MODNet ✓"

        print(f"\n  ✓ {config_name}: {avg_mae:.4f} ± {std_mae:.4f} MPa{tag}  "
              f"(took {t_cfg/60:.1f} min)")

        all_results[config_name] = {
            'arch':          arch,
            'actual_params': actual_p,
            'hidden_dim':    hidden_dim,
            'ff_dim':        ff_dim,
            'avg_mae':       avg_mae,
            'std_mae':       std_mae,
            'fold_maes':     fold_maes,
        }
        all_trajectories[config_name] = np.mean(fold_trajectories, axis=0).tolist()

    # ── final summary table ────────────────────────────────────────────
    t_total = time.time() - t_total_start
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<26} {'Arch':<14} {'Params':>9} {'MAE(MPa)':>10} {'±Std':>8}  Status")
    print(f"  {'─'*70}")

    # sort by avg_mae ascending
    sorted_res = sorted(all_results.items(), key=lambda x: x[1]['avg_mae'])

    for name, r in sorted_res:
        status = "BEATS TPOT 🏆" if r['avg_mae'] < 79.95 else \
                 "Beats AutoML ✓" if r['avg_mae'] < 82.30 else \
                 "Beats MODNet ✓" if r['avg_mae'] < 87.76 else \
                 "Close ≈"        if r['avg_mae'] < 105   else ""
        print(f"  {name:<26} {r['arch']:<14} {r['actual_params']:>9,} "
              f"{r['avg_mae']:>10.4f} {r['std_mae']:>8.4f}  {status}")

    print(f"  {'─'*70}")
    print(f"  {'TPOT-Mat (best)':<26} {'baseline':<14} {'—':>9} {79.9468:>10.4f}")
    print(f"  {'AutoML-Mat':<26} {'baseline':<14} {'—':>9} {82.3043:>10.4f}")
    print(f"  {'MODNet v0.1.12':<26} {'baseline':<14} {'—':>9} {87.7627:>10.4f}")
    print(f"  {'RF-SCM/Magpie':<26} {'baseline':<14} {'—':>9} {103.5125:>10.4f}")
    print(f"\n  Total training time: {t_total/60:.1f} minutes\n")

    generate_plots(all_results, all_trajectories)
    save_summary(all_results, all_trajectories, input_dim, t_total)

    log.info(f"All done! Files: trm_results_v2.png | trm_models/")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PLOTS  (4 panels)
# ══════════════════════════════════════════════════════════════════════════════

PALETTE_MLP   = ['#1565C0','#42A5F5','#0D47A1','#1976D2','#1E88E5','#64B5F6']
PALETTE_TRANS = ['#B71C1C','#EF5350','#C62828','#E53935','#F44336','#EF9A9A']


def generate_plots(all_results: dict, all_trajectories: dict):

    names   = list(all_results.keys())
    maes    = [all_results[n]['avg_mae']       for n in names]
    stds    = [all_results[n]['std_mae']       for n in names]
    params  = [all_results[n]['actual_params'] for n in names]
    arches  = [all_results[n]['arch']          for n in names]
    colors  = [PALETTE_MLP[i % 6] if a == 'MLP' else PALETTE_TRANS[i % 6]
               for i, a in enumerate(arches)]

    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.32)

    # ── Panel 1: Main bar chart (all 12 models) ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    short_names = [n.replace('MLP-TRM-', 'MLP-').replace('Trans-TRM-', 'Tr-')
                   for n in names]
    bars = ax1.bar(short_names, maes, yerr=stds, capsize=5,
                   color=colors, alpha=0.88, edgecolor='white', linewidth=1.2)

    for bname, bval, col, ls in [
        ('TPOT (79.95)',    79.95,  '#2E7D32', '--'),
        ('AutoML (82.30)',  82.30,  '#558B2F', '--'),
        ('MODNet (87.76)',  87.76,  '#F57F17', '--'),
        ('RF-Magpie (103)', 103.51, '#795548', ':'),
    ]:
        ax1.axhline(bval, color=col, linestyle=ls, linewidth=1.8,
                    label=bname, alpha=0.8)

    for bar, mae, std in zip(bars, maes, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + std + 1,
                 f'{mae:.1f}', ha='center', va='bottom',
                 fontsize=7.5, fontweight='bold')

    # Legend patches for architecture
    from matplotlib.patches import Patch
    handles, labels = ax1.get_legend_handles_labels()
    handles += [Patch(color=PALETTE_MLP[0], label='MLP-TRM family'),
                Patch(color=PALETTE_TRANS[0], label='Transformer-TRM family')]
    ax1.legend(handles=handles, fontsize=8, loc='upper right', ncol=2)
    ax1.set_ylabel('Mean Absolute Error (MPa)', fontsize=11)
    ax1.set_title('TRM-MatSci — 12-Model Ablation │ matbench_steels 5-Fold CV',
                  fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(maes) * 1.30)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)

    # ── Panel 2: Params vs MAE scatter ────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for i, (name, p, m, s, a) in enumerate(zip(names, params, maes, stds, arches)):
        c = PALETTE_MLP[i % 6] if a == 'MLP' else PALETTE_TRANS[i % 6]
        ax2.scatter(p, m, s=80, color=c, zorder=5)
        ax2.errorbar(p, m, yerr=s, fmt='none', color=c, capsize=3, linewidth=1.2)
        ax2.annotate(name.split('-')[-2]+'-'+name.split('-')[-1],
                     (p, m), textcoords='offset points', xytext=(4, 3), fontsize=6)

    ax2.axhline(79.95, color='#2E7D32', linestyle='--', linewidth=1.2, alpha=0.7, label='TPOT')
    ax2.axhline(87.76, color='#F57F17', linestyle='--', linewidth=1.2, alpha=0.7, label='MODNet')
    ax2.set_xlabel('Actual Parameters', fontsize=10)
    ax2.set_ylabel('MAE (MPa)', fontsize=10)
    ax2.set_title('Scaling: Params vs MAE', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: MLP vs Transformer head-to-head ──────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    mlp_maes   = [(n, r['avg_mae']) for n, r in all_results.items() if r['arch'] == 'MLP']
    trans_maes = [(n, r['avg_mae']) for n, r in all_results.items() if r['arch'] == 'Transformer']

    mlp_labels   = [n.replace('MLP-TRM-',   '') for n, _ in mlp_maes]
    trans_labels = [n.replace('Trans-TRM-', '') for n, _ in trans_maes]
    mlp_vals     = [v for _, v in mlp_maes]
    trans_vals   = [v for _, v in trans_maes]

    x = np.arange(len(mlp_labels))
    w = 0.35
    ax3.bar(x - w/2, mlp_vals,   w, label='MLP-TRM',   color=PALETTE_MLP[0],   alpha=0.85)
    ax3.bar(x + w/2, trans_vals, w, label='Trans-TRM', color=PALETTE_TRANS[0], alpha=0.85)
    ax3.axhline(87.76, color='#F57F17', linestyle='--', linewidth=1.2, alpha=0.7, label='MODNet')
    ax3.set_xticks(x)
    ax3.set_xticklabels(mlp_labels, rotation=25, ha='right', fontsize=8)
    ax3.set_ylabel('MAE (MPa)', fontsize=10)
    ax3.set_title('MLP-TRM vs Transformer-TRM\nHead-to-Head Comparison', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # ── Panel 4: Convergence all 12 models overlay ────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    for i, (name, traj) in enumerate(all_trajectories.items()):
        a = all_results[name]['arch']
        c = PALETTE_MLP[i % 6] if a == 'MLP' else PALETTE_TRANS[i % 6]
        ls = '-' if a == 'MLP' else '--'
        short = name.replace('MLP-TRM-', 'MLP-').replace('Trans-TRM-', 'Tr-')
        ax4.plot(range(1, len(traj)+1), traj, label=short,
                 color=c, linewidth=1.8, linestyle=ls, alpha=0.9)

    ax4.axhline(79.95, color='#2E7D32', linestyle=':', linewidth=2.0, label='TPOT (79.95)')
    ax4.axhline(87.76, color='#F57F17', linestyle=':', linewidth=2.0, label='MODNet (87.76)')
    ax4.set_xlabel('Recursion Step (1 → 16)', fontsize=11)
    ax4.set_ylabel('MAE (MPa)', fontsize=11)
    ax4.set_title('Recursion Convergence — All 12 Models\n'
                  'Solid = MLP-TRM │ Dashed = Transformer-TRM',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, loc='upper right', ncol=3)
    ax4.grid(alpha=0.3)

    plt.suptitle(
        'TRM-MatSci v2 │ Full Ablation: 12 Models × 5 Folds │ matbench_steels',
        fontsize=14, fontweight='bold', y=1.01)

    plt.savefig('trm_results_v2.png', dpi=150, bbox_inches='tight')
    log.info("✓ Saved: trm_results_v2.png")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 9.  SAVE SUMMARY + HUGGINGFACE README
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(all_results: dict, all_trajectories: dict,
                 input_dim: int, total_time_s: float):

    summary = {
        'task':             'matbench_steels',
        'benchmark':        'matbench_v0.1',
        'input_type':       'mat2vec_weighted_sum',
        'input_dim':        input_dim,
        'recursion_steps':  16,
        'dataset_size':     312,
        'cv_strategy':      '5-fold nested cross-validation',
        'total_train_time_min': round(total_time_s / 60, 1),
        'baselines':        BASELINES,
        'models':           {}
    }

    for name, r in all_results.items():
        summary['models'][name] = {
            'arch':          r['arch'],
            'actual_params': r['actual_params'],
            'hidden_dim':    r['hidden_dim'],
            'ff_dim':        r['ff_dim'],
            'avg_mae_mpa':   round(r['avg_mae'], 4),
            'std_mae_mpa':   round(r['std_mae'], 4),
            'fold_maes':     [round(m, 4) for m in r['fold_maes']],
            'convergence_trajectory': [round(v, 4) for v in all_trajectories[name]],
            'beats_tpot':    r['avg_mae'] < 79.9468,
            'beats_modnet':  r['avg_mae'] < 87.7627,
        }

    with open('trm_models/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    readme = f"""# TRM-MatSci v2

Tiny Recursive Model (TRM) applied to materials science.
Predicts **steel yield strength** from chemical composition alone.

## Architectures
**MLP-TRM** and **Transformer-TRM** — both use the dual-state TRM mechanism:
- `z` = latent reasoning state
- `y` = current answer state  
- Both updated every recursive step with shared weights
- Dropout per step for regularisation on tiny datasets

## Benchmark
- **Dataset**: matbench_steels (312 samples)
- **Protocol**: matbench_v0.1 official 5-fold nested CV
- **Input**: Mat2Vec composition embedding ({input_dim}-dim weighted sum)
- **Output**: Yield strength (MPa)

## Results Summary

| Model | Arch | Params | MAE (MPa) | vs MODNet |
|-------|------|--------|-----------|-----------|
""" + "\n".join(
        f"| {n} | {r['arch']} | {r['actual_params']:,} | "
        f"{r['avg_mae']:.4f} ± {r['std_mae']:.4f} | "
        f"{'✓ better' if r['avg_mae'] < 87.76 else '✗'} |"
        for n, r in sorted(all_results.items(), key=lambda x: x[1]['avg_mae'])
    ) + f"""

**Baselines**: TPOT 79.95 │ AutoML 82.30 │ MODNet 87.76 │ RF-Magpie 103.51

## Loading a Model

```python
import torch
from trm_matsci_v2 import MLPTinyRecursiveModel, TransformerTinyRecursiveModel
from trm_matsci_v2 import Mat2VecFeaturizer
from pymatgen.core import Composition

featurizer = Mat2VecFeaturizer()

# Load any saved fold checkpoint
ckpt  = torch.load('trm_models/MLP-TRM-50K-h64_fold1.pt')
model = MLPTinyRecursiveModel(
    input_dim       = ckpt['input_dim'],
    hidden_dim      = ckpt['hidden_dim'],
    ff_dim          = ckpt['ff_dim'],
    recursion_steps = ckpt['recursion_steps']
)
model.load_state_dict(ckpt['model_state'])
model.eval()

comp = Composition("Fe0.8C0.02Mn0.01Cr0.05")
vec  = featurizer.featurize(comp)
x    = torch.tensor(vec).unsqueeze(0)
pred = model(x).item()
print(f"Predicted yield strength: {{pred:.1f}} MPa")
```
"""

    with open('trm_models/README.md', 'w') as f:
        f.write(readme)

    log.info("✓ Saved: trm_models/summary.json + trm_models/README.md")
    log.info(f"  → {12 * 5} model checkpoints saved in trm_models/")
    log.info("  → Upload trm_models/ folder to HuggingFace Hub")


# ══════════════════════════════════════════════════════════════════════════════
# 10. EXTERNAL TEST SET  (your curated steels from literature)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_custom_testset(model, csv_path: str,
                            featurizer: Mat2VecFeaturizer,
                            device: torch.device) -> pd.DataFrame:
    """
    Blind test on your own curated steels from external literature.

    CSV format (two columns, no extra spaces):
        composition,yield_strength
        Fe0.8C0.02Mn0.01Cr0.05,850.0
        Fe0.75Mn0.15Cr0.08Al0.02,920.0

    Returns DataFrame: composition | actual | predicted | abs_error
    """
    df   = pd.read_csv(csv_path)
    model.eval()
    rows = []

    with torch.no_grad():
        for _, row in df.iterrows():
            try:
                comp = Composition(row['composition'])
                vec  = featurizer.featurize(comp)
                x    = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
                pred = model(x).item()
                rows.append({
                    'composition': row['composition'],
                    'actual':      float(row['yield_strength']),
                    'predicted':   round(pred, 2),
                    'abs_error':   round(abs(pred - float(row['yield_strength'])), 2)
                })
            except Exception as ex:
                log.warning(f"Failed: '{row['composition']}' → {ex}")

    results = pd.DataFrame(rows)
    if len(results):
        mae = results['abs_error'].mean()
        log.info(f"External test MAE: {mae:.2f} MPa over {len(results)} samples")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_benchmark()