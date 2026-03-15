"""
+=============================================================+
|  TRIADS — Unified GNN-Attention-TRM for matbench_phonons    |
|  "The Phonon Simulator"                                     |
|                                                             |
|  Each reasoning step (x12) executes ALL THREE:              |
|    A) GNN Message Passing  (local spring-mass physics)      |
|    B) Self-Attention       (global chemical reasoning)      |
|    C) TRM State Update     (z/y memory + deep supervision)  |
|    D) Broadcast feedback   (TRM informs next GNN step)      |
|                                                             |
|  Tokens = 22 Magpie + 1 Extra + 1 Mat2Vec + 1 GNN = 25     |
|  ~184K params | d=80 | 12 reasoning steps                   |
+=============================================================+

DEPENDENCIES:
    pip install matminer pymatgen gensim tqdm scikit-learn torch

USAGE:
    python phonons_v3.py
    (Auto-builds dataset on first run if phonons_dataset.pt missing)
"""

import os, copy, json, time, warnings, urllib.request
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

D = 80                    # Hidden dimension
MAX_STEPS = 12            # Reasoning loop iterations
BATCH_SIZE = 32
EPOCHS = 350
LR = 1e-3
SEEDS = [42]
MATBENCH_FOLD_SEED = 18012019
N_GAUSSIAN = 40
MAX_NEIGHBORS = 12
CUTOFF = 8.0

BASELINES = {
    'MEGNet':            28.7636,
    'ALIGNN':            29.3378,
    'MODNet v0.1.12':    45.3924,
    'CrabNet':           47.0921,
    'TRIADS V1 (comp)':  71.8169,
    'Dummy':            323.7588,
}

# Atomic masses Z=0..102
_MASSES = [1e-8, 1.008, 4.003, 6.94, 9.012, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
    22.99, 24.31, 26.98, 28.09, 30.97, 32.06, 35.45, 39.95, 39.10, 40.08,
    44.96, 47.87, 50.94, 52.00, 54.94, 55.85, 58.93, 58.69, 63.55, 65.38,
    69.72, 72.63, 74.92, 78.97, 79.90, 83.80, 85.47, 87.62, 88.91, 91.22,
    92.91, 95.95, 98.0, 101.1, 102.9, 106.4, 107.9, 112.4, 114.8, 118.7,
    121.8, 127.6, 126.9, 131.3, 132.9, 137.3, 138.9, 140.1, 140.9, 144.2,
    145.0, 150.4, 152.0, 157.3, 158.9, 162.5, 164.9, 167.3, 168.9, 173.1,
    175.0, 178.5, 181.0, 183.8, 186.2, 190.2, 192.2, 195.1, 197.0, 200.6,
    204.4, 207.2, 209.0, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0,
    231.0, 238.0, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0,
    258.0, 259.0, 262.0]
INV_SQRT_MASS = torch.tensor([1.0 / (m ** 0.5) for m in _MASSES], dtype=torch.float32).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════
# 1. DATASET BUILDER (runs once)
# ═══════════════════════════════════════════════════════════════

def gaussian_expand(distances, n_bins=N_GAUSSIAN, d_min=0.0, d_max=CUTOFF):
    centers = torch.linspace(d_min, d_max, n_bins)
    gamma = 1.0 / ((d_max - d_min) / n_bins) ** 2
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)


def build_graph(structure):
    atom_z = torch.tensor([s.specie.Z for s in structure], dtype=torch.long)
    src, dst, dists, vecs = [], [], [], []
    try:
        all_nbrs = structure.get_all_neighbors(CUTOFF)
        for i, nbrs in enumerate(all_nbrs):
            nbrs = sorted(nbrs, key=lambda x: x.nn_distance)[:MAX_NEIGHBORS]
            for nbr in nbrs:
                src.append(i); dst.append(nbr.index)
                dists.append(nbr.nn_distance)
                vecs.append(nbr.coords - structure[i].coords)
    except: pass
    if not src:
        return atom_z, torch.zeros(2,1,dtype=torch.long), torch.zeros(1,N_GAUSSIAN), torch.zeros(1,3)
    ei = torch.tensor([src, dst], dtype=torch.long)
    d = torch.tensor(dists, dtype=torch.float32)
    v = torch.tensor(np.array(vecs), dtype=torch.float32)
    return atom_z, ei, gaussian_expand(d), v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def build_comp_features(comp, struct):
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, ValenceOrbital, IonProperty)
    from matminer.featurizers.composition.element import TMetalFraction
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    ep = ElementProperty.from_preset("magpie")
    mg = np.nan_to_num(np.array(ep.featurize(comp), np.float32), nan=0.0)
    extras = []
    for ft in [Stoichiometry(), ValenceOrbital(), IonProperty(), TMetalFraction()]:
        try: extras.append(np.nan_to_num(np.array(ft.featurize(comp), np.float32), nan=0.0))
        except: extras.append(np.zeros(1, np.float32))
    lat = struct.lattice
    try: sg = float(SpacegroupAnalyzer(struct, symprec=0.1).get_space_group_number())
    except: sg = 0.0
    s_feats = np.array([lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma,
                        struct.volume/max(len(struct),1), struct.density,
                        float(len(struct)), sg, 0.0], np.float32)
    return np.concatenate([mg] + extras + [s_feats])


def build_dataset():
    print("  Building phonons_dataset.pt ...")
    from matminer.datasets import load_dataset
    from gensim.models import Word2Vec
    df = load_dataset("matbench_phonons")
    targets = np.array(df['last phdos peak'].tolist(), np.float32)
    structs = df['structure'].tolist()
    cache = "mat2vec_cache"; os.makedirs(cache, exist_ok=True)
    for f in ["pretrained_embeddings", "pretrained_embeddings.wv.vectors.npy",
              "pretrained_embeddings.trainables.syn1neg.npy"]:
        p = os.path.join(cache, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve("https://storage.googleapis.com/mat2vec/" + f, p)
    m2v = Word2Vec.load(os.path.join(cache, "pretrained_embeddings"))
    emb = {w: m2v.wv[w] for w in m2v.wv.index_to_key}
    def m2v_pool(comp):
        v, t = np.zeros(200, np.float32), 1e-8
        for s, f in comp.get_el_amt_dict().items():
            if s in emb: v += f * emb[s]; t += f
        return v / t
    graphs, comp_feats = [], []
    for i, s in enumerate(tqdm(structs, desc="  Processing crystals")):
        az, ei, ef, ev = build_graph(s)
        graphs.append({'z': az, 'ei': ei, 'ef': ef, 'ev': ev, 'na': len(az)})
        cf = build_comp_features(s.composition, s)
        comp_feats.append(np.concatenate([cf, m2v_pool(s.composition)]))
    data = {
        'graphs': graphs,
        'comp': torch.tensor(np.array(comp_feats), dtype=torch.float32),
        'targets': torch.tensor(targets, dtype=torch.float32),
        'comp_dim': len(comp_feats[0]),
    }
    torch.save(data, "phonons_dataset.pt")
    print(f"  ✅ Saved phonons_dataset.pt ({os.path.getsize('phonons_dataset.pt')/1e6:.1f} MB)")
    return data


# ═══════════════════════════════════════════════════════════════
# 2. SCATTER OPS
# ═══════════════════════════════════════════════════════════════

def scatter_sum(src, index, dim_size):
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    return out

def scatter_mean(src, index, dim_size):
    s = scatter_sum(src, index, dim_size)
    c = scatter_sum(torch.ones_like(src[:, :1]), index, dim_size).clamp(min=1)
    return s / c


# ═══════════════════════════════════════════════════════════════
# 3. THE UNIFIED GNN-ATTENTION-TRM  (~184K params)
# ═══════════════════════════════════════════════════════════════

class UnifiedGNNAttentionTRM(nn.Module):
    """
    The complete unified reasoning loop. At EACH of 12 steps:

      A) GNN Message Passing   — local bond-level spring physics
      B) Self-Attention        — global chemical reasoning over all tokens
      C) TRM State Update      — memory z,y updated from pooled attention
      D) Broadcast feedback    — TRM state feeds back to atoms for next step

    Tokens (25 total):
      22 Magpie property tokens (EVOLVING through the loop)
      1  Extra composition token
      1  Mat2Vec chemical context token
      1  GNN geometric token (re-computed each step from atoms)

    All GNN, Attention, and TRM weights are SHARED across steps.
    12 steps = 12 layers of GNN + 12 layers of Attention + 12 TRM updates.
    Parameter cost = 1 layer of each.
    """

    def __init__(self, comp_dim, d=D, max_steps=MAX_STEPS):
        super().__init__()
        self.d = d
        self.max_steps = max_steps

        # ── Physics Buffers ──────────────────────────────────
        self.register_buffer('inv_sqrt_m', INV_SQRT_MASS)

        # ── Input Projections (outside loop) ─────────────────
        # Atoms
        self.atom_embed = nn.Embedding(103, d)
        self.mass_proj  = nn.Linear(1, d)
        # Edges
        self.edge_enc   = nn.Linear(N_GAUSSIAN, d)
        self.vec_enc    = nn.Linear(3, d)
        self.ang_enc    = nn.Linear(9, d)
        # Composition → Tokens
        self.tok_proj   = nn.Linear(6, d)       # Magpie stat (6d) → d per token
        self.extra_proj = nn.Linear(comp_dim - 132 - 200, d)  # extras → 1 token
        self.m2v_proj   = nn.Linear(200, d)     # Mat2Vec → 1 token

        # ── Shared GNN Layer (inside loop) ───────────────────
        self.gnn_msg  = nn.Sequential(nn.Linear(d*3, d), nn.SiLU(), nn.Linear(d, d))
        self.gnn_gate = nn.Sequential(nn.Linear(d*3, d), nn.Sigmoid())
        self.gnn_up   = nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.SiLU())

        # ── Shared Self-Attention (inside loop) ──────────────
        self.sa   = nn.MultiheadAttention(d, num_heads=4, dropout=0.1, batch_first=True)
        self.sa_n = nn.LayerNorm(d)
        self.sa_ff = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d*2, d))
        self.sa_fn = nn.LayerNorm(d)

        # ── Shared TRM State Update (inside loop) ────────────
        self.pool_proj = nn.Linear(d, d)
        self.z_up = nn.Sequential(nn.Linear(d*3, d), nn.SiLU(), nn.Linear(d, d))
        self.y_up = nn.Sequential(nn.Linear(d*2, d), nn.SiLU(), nn.Linear(d, d))

        # ── Feedback: TRM → Atoms (inside loop) ─────────────
        self.feedback = nn.Linear(d, d)

        # ── Output Head ──────────────────────────────────────
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, comp, g, deep_supervision=False):
        z_nums = g['z']
        ei     = g['ei']
        ef     = g['ef']
        ev     = g['ev']
        batch  = g['batch']
        B      = g['n_crystals']

        # ══════════════════════════════════════════════════════
        #  INITIALIZATION
        # ══════════════════════════════════════════════════════

        # A. Initialize Atoms with Physics
        atoms = self.atom_embed(z_nums) + self.mass_proj(self.inv_sqrt_m[z_nums])
        edges = self.edge_enc(ef) * torch.tanh(self.vec_enc(ev))
        # Implicit angular context
        v_outer = (ev.unsqueeze(-1) * ev.unsqueeze(-2)).view(-1, 9)
        atoms = atoms + self.ang_enc(scatter_sum(v_outer, ei[1], atoms.size(0)))

        # B. Initialize Composition Tokens
        # Magpie: 22 tokens of stat_dim=6 → projected to d
        magpie_raw = comp[:, :132].view(B, 22, 6)
        magpie_tokens = self.tok_proj(magpie_raw)         # (B, 22, d)
        # Extra composition: 1 token
        extra_token = self.extra_proj(comp[:, 132:-200]).unsqueeze(1)  # (B, 1, d)
        # Mat2Vec: 1 context token
        m2v_token = self.m2v_proj(comp[:, -200:]).unsqueeze(1)         # (B, 1, d)

        # C. Initialize TRM Memory
        z_state = torch.zeros(B, self.d, device=comp.device)
        y_state = torch.zeros(B, self.d, device=comp.device)

        step_preds = []

        # ══════════════════════════════════════════════════════
        #  THE UNIFIED REASONING LOOP (12 steps)
        # ══════════════════════════════════════════════════════
        for step in range(self.max_steps):

            # ── A. GNN: Local Bond Physics ───────────────────
            inp = torch.cat([atoms[ei[0]], atoms[ei[1]], edges], dim=-1)
            msg = self.gnn_msg(inp) * self.gnn_gate(inp)
            agg = scatter_sum(msg, ei[1], atoms.size(0))
            atoms = atoms + self.gnn_up(torch.cat([atoms, agg], dim=-1))

            # ── B. Pool Atoms → Geometric Token ──────────────
            gnn_token = scatter_mean(atoms, batch, B).unsqueeze(1)  # (B, 1, d)

            # ── C. Assemble Token Set (25 tokens) ────────────
            tokens = torch.cat([
                magpie_tokens,   # (B, 22, d) — EVOLVING
                extra_token,     # (B, 1, d)
                m2v_token,       # (B, 1, d)
                gnn_token,       # (B, 1, d)  — re-computed each step
            ], dim=1)           # (B, 25, d)

            # ── D. Self-Attention: TRIADS Global Reasoning ───
            tokens = self.sa_n(tokens + self.sa(tokens, tokens, tokens)[0])
            tokens = self.sa_fn(tokens + self.sa_ff(tokens))

            # ── E. Extract Evolved Tokens ────────────────────
            magpie_tokens = tokens[:, :22]
            extra_token   = tokens[:, 22:23]
            m2v_token     = tokens[:, 23:24]
            # gnn_token is re-computed next step, no need to extract

            # ── F. Pool Tokens → TRM State Update ────────────
            xp = self.pool_proj(tokens.mean(dim=1))  # (B, d)
            z_state = z_state + self.z_up(torch.cat([xp, y_state, z_state], dim=-1))
            y_state = y_state + self.y_up(torch.cat([y_state, z_state], dim=-1))

            # ── G. Broadcast TRM Feedback to Atoms ───────────
            fb = self.feedback(y_state)
            atoms = atoms + fb[batch]

            # ── H. Predict from Current State ────────────────
            step_preds.append(self.head(y_state).squeeze(-1))

        return step_preds if deep_supervision else step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# 4. LOSS + UTILS
# ═══════════════════════════════════════════════════════════════

def deep_supervision_loss(step_preds, targets):
    preds = torch.stack(step_preds)
    S = preds.shape[0]
    w = torch.arange(1, S + 1, device=preds.device, dtype=preds.dtype)
    w = w / w.sum()
    per_step = (preds - targets.unsqueeze(0)).abs().mean(dim=1)
    return (w * per_step).sum()


def strat_split(targets, val_frac=0.15, seed=42):
    bins = np.digitize(targets, np.percentile(targets, [25, 50, 75]))
    tr, vl = [], []
    rng = np.random.RandomState(seed)
    for b in range(4):
        m = np.where(bins == b)[0]
        if len(m) == 0: continue
        n = max(1, int(len(m) * val_frac))
        c = rng.choice(m, n, replace=False)
        vl.extend(c.tolist()); tr.extend(np.setdiff1d(m, c).tolist())
    return np.array(tr), np.array(vl)


# ═══════════════════════════════════════════════════════════════
# 5. GRAPH BATCHING + DATALOADER
# ═══════════════════════════════════════════════════════════════

def collate_batch(graphs, comp, targets, indices, device):
    zs, eis, efs, evs, bs = [], [], [], [], []
    offset = 0
    for k, i in enumerate(indices):
        g = graphs[i]
        na = g['na']
        zs.append(g['z']); eis.append(g['ei'] + offset)
        efs.append(g['ef']); evs.append(g['ev'])
        bs.append(torch.full((na,), k, dtype=torch.long))
        offset += na
    return (
        comp[indices].to(device),
        {'z': torch.cat(zs).to(device), 'ei': torch.cat(eis, dim=1).to(device),
         'ef': torch.cat(efs).to(device), 'ev': torch.cat(evs).to(device),
         'batch': torch.cat(bs).to(device), 'n_crystals': len(indices)},
        targets[indices].to(device)
    )


class ShuffledGraphLoader:
    def __init__(self, graphs, comp, targets, indices, bs, device, shuffle=False):
        self.g, self.c, self.t = graphs, comp, targets
        self.idx, self.bs, self.dev, self.shuf = np.array(indices), bs, device, shuffle

    def __iter__(self):
        idx = self.idx.copy()
        if self.shuf: np.random.shuffle(idx)
        self._b = [idx[i:i+self.bs] for i in range(0, len(idx), self.bs)]
        self._p = 0; return self

    def __next__(self):
        if self._p >= len(self._b): raise StopIteration
        b = self._b[self._p]; self._p += 1
        return collate_batch(self.g, self.c, self.t, b, self.dev)

    def __len__(self):
        return (len(self.idx) + self.bs - 1) // self.bs


# ═══════════════════════════════════════════════════════════════
# 6. TRAINING
# ═══════════════════════════════════════════════════════════════

def train_fold(model, tr_dl, vl_dl, device, fold, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    best_v, best_w = float('inf'), None

    pbar = tqdm(range(EPOCHS), desc=f"  [GNN+Attn+TRM|s{seed}] F{fold}/5",
                leave=False, ncols=120)
    for ep in pbar:
        model.train()
        total_err, total_n = 0.0, 0
        for comp_b, graph_b, target_b in tr_dl:
            sp = model(comp_b, graph_b, deep_supervision=True)
            loss = deep_supervision_loss(sp, target_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                total_err += (sp[-1] - target_b).abs().sum().item()
                total_n += len(target_b)

        model.eval()
        val_err, val_n = 0.0, 0
        with torch.inference_mode():
            for comp_b, graph_b, target_b in vl_dl:
                pred = model(comp_b, graph_b)
                val_err += (pred - target_b).abs().sum().item()
                val_n += len(target_b)

        tl = total_err / total_n
        vl = val_err / val_n
        sch.step()

        if vl < best_v:
            best_v = vl
            best_w = copy.deepcopy(model.state_dict())

        if ep % 20 == 0 or ep == EPOCHS - 1:
            pbar.set_postfix(Best=f'{best_v:.1f}', Tr=f'{tl:.1f}', Val=f'{vl:.1f}')

    model.load_state_dict(best_w)
    return best_v, model


# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  TRIADS V3 — Unified GNN-Attention-TRM                  ║
  ║  "The Phonon Simulator"                                 ║
  ║                                                         ║
  ║  Each step: GNN → Self-Attention → TRM → Broadcast      ║
  ║  25 tokens | d={D} | 12 steps | <200K params             ║
  ╚══════════════════════════════════════════════════════════╝
    """)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if os.path.exists("phonons_dataset.pt"):
        print("  Loading phonons_dataset.pt ...")
        data = torch.load("phonons_dataset.pt", weights_only=False)
    else:
        data = build_dataset()

    graphs = data['graphs']
    comp_all = data['comp']
    targets_all = data['targets']
    comp_dim = data['comp_dim']
    N = len(graphs)
    print(f"  Dataset: {N} samples | Comp dim: {comp_dim}")

    test_model = UnifiedGNNAttentionTRM(comp_dim=comp_dim)
    n_params = test_model.count_parameters()
    print(f"  Model: {n_params:,} parameters")
    assert n_params < 200_000, f"OVER BUDGET: {n_params:,}"
    print(f"  ✓ Under 200K\n")
    del test_model

    kfold = KFold(n_splits=5, shuffle=True, random_state=MATBENCH_FOLD_SEED)
    folds = list(kfold.split(range(N)))
    all_te = []
    for tv, te in folds:
        assert len(set(tv) & set(te)) == 0
        all_te.extend(te.tolist())
    assert len(set(all_te)) == N and len(all_te) == N
    print("  5 folds: zero leakage ✓\n")

    targets_np = targets_all.numpy()
    all_seed_maes = {}

    for seed in SEEDS:
        print(f"  {'─'*3} Seed {seed} {'─'*45}")
        t_seed = time.time()
        seed_maes = {}

        for fi, (tv_i, te_i) in enumerate(folds):
            tri, vli = strat_split(targets_np[tv_i], 0.15, seed + fi)
            sc = StandardScaler().fit(comp_all[tv_i[tri]].numpy())
            comp_scaled = torch.tensor(
                np.nan_to_num(sc.transform(comp_all.numpy()), nan=0.0).astype(np.float32))

            tr_dl = ShuffledGraphLoader(graphs, comp_scaled, targets_all, tv_i[tri],
                                        BATCH_SIZE, device, shuffle=True)
            vl_dl = ShuffledGraphLoader(graphs, comp_scaled, targets_all, tv_i[vli],
                                        BATCH_SIZE, device, shuffle=False)
            te_dl = ShuffledGraphLoader(graphs, comp_scaled, targets_all, te_i,
                                        BATCH_SIZE, device, shuffle=False)

            torch.manual_seed(seed + fi)
            np.random.seed(seed + fi)
            if device.type == 'cuda': torch.cuda.manual_seed(seed + fi)

            model = UnifiedGNNAttentionTRM(comp_dim=comp_dim).to(device)
            bv, model = train_fold(model, tr_dl, vl_dl, device, fi+1, seed)

            model.eval()
            te_err, te_n = 0.0, 0
            with torch.inference_mode():
                for comp_b, graph_b, target_b in te_dl:
                    pred = model(comp_b, graph_b)
                    te_err += (pred - target_b).abs().sum().item()
                    te_n += len(target_b)
            mae = te_err / te_n
            seed_maes[fi] = mae
            print(f"    Fold {fi+1}: MAE = {mae:.2f} cm⁻¹")

            del model
            if device.type == 'cuda': torch.cuda.empty_cache()

        avg = np.mean(list(seed_maes.values()))
        all_seed_maes[seed] = seed_maes
        print(f"\n  Seed {seed} avg: {avg:.2f} ({time.time()-t_seed:.0f}s)")

    final_avg = np.mean([np.mean(list(v.values())) for v in all_seed_maes.values()])
    tt = time.time() - t0

    print(f"""
{'='*60}
  FINAL — Unified GNN-Attention-TRM on matbench_phonons
{'='*60}

  {'Model':<40} {'MAE':>10}
  {'─'*52}""")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        mk = " ← BEATEN!" if final_avg < bv else ""
        print(f"  {bn:<40} {bv:>10.2f}{mk}")
    print(f"  {'TRIADS V3 GNN-Attn-TRM ('+str(n_params//1000)+'K)':<40} {final_avg:>10.2f} ← US")
    print(f"  {'─'*52}\n  Time: {tt/60:.1f} min")

    with open('phonons_v3_results.json', 'w') as f:
        json.dump({'model': 'Unified-GNN-Attn-TRM', 'params': n_params,
                   'final_avg': round(final_avg, 2),
                   'per_fold': {str(s): {str(k): round(v,2) for k,v in m.items()}
                                for s,m in all_seed_maes.items()},
                   'time_min': round(tt/60, 1)}, f, indent=2)
    print("  Saved: phonons_v3_results.json")


if __name__ == '__main__':
    main()
