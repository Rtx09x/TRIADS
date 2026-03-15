"""
+=============================================================+
|  TRIADS V4 — Angular GNN + Attention-TRM for phonons        |
|                                                             |
|  KEY CHANGE FROM V3.1:                                      |
|    The GNN Sensors now see BOND ANGLES, not just distances. |
|    Each GNN layer updates both nodes AND edges using        |
|    explicit i→j→k triplet angles (like ALIGNN).             |
|                                                             |
|  Stage 1: 3 Unshared AngularGNN Layers (d_gnn=40)          |
|    - 7 per-atom features (mass, EN, radius, val, grp, per,  |
|      + coordination number)                                 |
|    - Edge features updated from angular triplets            |
|    → N per-atom tokens                                      |
|                                                             |
|  Stage 2: 16 Shared TRM Cycles (d=64)                       |
|    ① Self-Attention [24 comp + N atom] + type_embed         |
|    ② Cross-Attention: comp queries atoms                    |
|    ③ TRM z,y update with Channel C struct metadata          |
|                                                             |
|  ~180K params | d_gnn=40, d=64 | 16 cycles                  |
+=============================================================+
"""

import os, copy, json, time, math, warnings, urllib.request
from collections import defaultdict
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

D           = 64      # TRM / token hidden dim
D_GNN       = 48      # GNN hidden dim
N_GNN       = 3       # Unshared Angular GNN layers
N_CYCLES    = 16      # TRM reasoning cycles (shared weights)
N_ANGLE_GAUSS = 8     # Gaussian bins for angle expansion
BATCH_SIZE  = 32
EPOCHS      = 350
LR          = 1e-3
SEEDS       = [42]
FOLD_SEED   = 18012019
N_GAUSS     = 40
MAX_NBR     = 12
CUTOFF      = 8.0

BASELINES = {
    'MEGNet': 28.76, 'ALIGNN': 29.34, 'MODNet': 45.39,
    'CrabNet': 47.09, 'TRIADS V3.1': 63.00, 'TRIADS V1': 71.82, 'Dummy': 323.76,
}

# ═══════════════════════════════════════════════════════════════
# PHYSICS LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════

def _build_phys_table():
    """Build (103, 6) table: [1/√m, EN, radius, valence_e, group, period]."""
    from pymatgen.core.periodic_table import Element
    rows = [[0.]*6]
    for z in range(1, 103):
        try:
            el = Element.from_Z(z)
            m = float(el.atomic_mass)
            en = float(el.X) if el.X is not None else 0.0
            ar = float(el.atomic_radius) if el.atomic_radius is not None else 1.5
            g = int(el.group) if el.group is not None else 0
            p = int(el.row) if el.row is not None else 0
            ve = g if g <= 2 else (g - 10 if g >= 13 else 2)
            rows.append([1./m**0.5, en, ar, float(ve), float(g), float(p)])
        except:
            rows.append([0., 0., 1.5, 0., 0., 0.])
    return torch.tensor(rows, dtype=torch.float32)

# ═══════════════════════════════════════════════════════════════
# GAUSSIAN EXPANSIONS
# ═══════════════════════════════════════════════════════════════

def gaussian_expand(d, n=N_GAUSS, lo=0., hi=CUTOFF):
    c = torch.linspace(lo, hi, n)
    g = 1. / ((hi - lo) / n) ** 2
    return torch.exp(-g * (d.unsqueeze(-1) - c) ** 2)

def gaussian_expand_angle(angles, n=N_ANGLE_GAUSS, lo=0., hi=math.pi):
    c = torch.linspace(lo, hi, n)
    g = 1. / ((hi - lo) / n) ** 2
    return torch.exp(-g * (angles.unsqueeze(-1) - c) ** 2)

# ═══════════════════════════════════════════════════════════════
# DATASET BUILDER
# ═══════════════════════════════════════════════════════════════

def build_graph(struct):
    az = torch.tensor([s.specie.Z for s in struct], dtype=torch.long)
    src, dst, ds, vs = [], [], [], []
    try:
        for i, nbrs in enumerate(struct.get_all_neighbors(CUTOFF)):
            for nbr in sorted(nbrs, key=lambda x: x.nn_distance)[:MAX_NBR]:
                src.append(i); dst.append(nbr.index)
                ds.append(nbr.nn_distance)
                vs.append(nbr.coords - struct[i].coords)
    except: pass
    if not src:
        return az, torch.zeros(2,1,dtype=torch.long), torch.zeros(1,N_GAUSS), torch.zeros(1,3)
    ei = torch.tensor([src, dst], dtype=torch.long)
    d = torch.tensor(ds, dtype=torch.float32)
    v = torch.tensor(np.array(vs), dtype=torch.float32)
    return az, ei, gaussian_expand(d), v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

def build_comp_features(comp, struct):
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, ValenceOrbital, IonProperty)
    from matminer.featurizers.composition.element import TMetalFraction
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    mg = np.nan_to_num(np.array(
        ElementProperty.from_preset("magpie").featurize(comp), np.float32), nan=0.)
    extras = []
    for ft in [Stoichiometry(), ValenceOrbital(), IonProperty(), TMetalFraction()]:
        try: extras.append(np.nan_to_num(np.array(ft.featurize(comp), np.float32), nan=0.))
        except: extras.append(np.zeros(1, np.float32))
    lat = struct.lattice
    try: sg = float(SpacegroupAnalyzer(struct, symprec=0.1).get_space_group_number())
    except: sg = 0.
    sf = np.array([lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma,
                   struct.volume/max(len(struct),1), struct.density,
                   float(len(struct)), sg, 0.], np.float32)
    return np.concatenate([mg] + extras + [sf])

def build_dataset():
    print("  Building phonons_dataset.pt ...")
    from matminer.datasets import load_dataset
    from gensim.models import Word2Vec
    df = load_dataset("matbench_phonons")
    tgt = np.array(df['last phdos peak'].tolist(), np.float32)
    structs = df['structure'].tolist()
    cache = "mat2vec_cache"; os.makedirs(cache, exist_ok=True)
    for f in ["pretrained_embeddings", "pretrained_embeddings.wv.vectors.npy",
              "pretrained_embeddings.trainables.syn1neg.npy"]:
        p = os.path.join(cache, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve("https://storage.googleapis.com/mat2vec/" + f, p)
    m2v = Word2Vec.load(os.path.join(cache, "pretrained_embeddings"))
    emb = {w: m2v.wv[w] for w in m2v.wv.index_to_key}
    def pool(c):
        v, t = np.zeros(200, np.float32), 1e-8
        for s, f in c.get_el_amt_dict().items():
            if s in emb: v += f*emb[s]; t += f
        return v / t
    graphs, cfs = [], []
    for s in tqdm(structs, desc="  Crystals"):
        az, ei, ef, ev = build_graph(s)
        graphs.append({'z': az, 'ei': ei, 'ef': ef, 'ev': ev, 'na': len(az)})
        cfs.append(np.concatenate([build_comp_features(s.composition, s), pool(s.composition)]))
    data = {'graphs': graphs, 'comp': torch.tensor(np.array(cfs), dtype=torch.float32),
            'targets': torch.tensor(tgt, dtype=torch.float32), 'comp_dim': len(cfs[0])}
    torch.save(data, "phonons_dataset.pt")
    print(f"  ✅ Saved ({os.path.getsize('phonons_dataset.pt')/1e6:.1f} MB)")
    return data

# ═══════════════════════════════════════════════════════════════
# ANGULAR FEATURE PRECOMPUTATION
#   Runs once at load time on existing phonons_dataset.pt.
#   Adds: triplet indices, Gaussian-expanded angles, coord nums.
# ═══════════════════════════════════════════════════════════════

def precompute_angular(graphs):
    """
    For each graph, compute:
      - triplets: (2, n_triplets) edge-index pairs [ij, kj] sharing dest j
      - angle_feat: (n_triplets, N_ANGLE_GAUSS) Gaussian-expanded angles
      - coord: (na,) coordination number per atom
    """
    for g in tqdm(graphs, desc="  Computing bond angles"):
        ei = g['ei']           # (2, ne)
        ev = g['ev']           # (ne, 3) — unit vectors
        na = g['na']
        ne = ei.shape[1]

        # ── Coordination Number ──────────────────────────────
        coord = torch.zeros(na, dtype=torch.float32)
        if ne > 0:
            coord.scatter_add_(0, ei[0], torch.ones(ne, dtype=torch.float32))
        g['coord'] = coord

        # ── Triplet Indices ──────────────────────────────────
        # Group edges by destination atom
        dst = ei[1].numpy()
        dest_to_edges = defaultdict(list)
        for e_idx in range(ne):
            dest_to_edges[int(dst[e_idx])].append(e_idx)

        trip_ij, trip_kj = [], []
        for j, edge_list in dest_to_edges.items():
            for idx_ij in edge_list:
                for idx_kj in edge_list:
                    if idx_ij != idx_kj:
                        trip_ij.append(idx_ij)
                        trip_kj.append(idx_kj)

        if len(trip_ij) == 0:
            g['triplets'] = torch.zeros(2, 0, dtype=torch.long)
            g['angle_feat'] = torch.zeros(0, N_ANGLE_GAUSS)
            continue

        triplets = torch.tensor([trip_ij, trip_kj], dtype=torch.long)

        # ── Angle Computation ────────────────────────────────
        # ev stores unit vectors from i→j.
        # Angle at j between bonds j-i and j-k:
        #   cos(θ) = dot(-v_ij, -v_kj) = dot(v_ij, v_kj)
        v_ij = ev[triplets[0]]  # (nt, 3)
        v_kj = ev[triplets[1]]  # (nt, 3)
        cos_theta = (v_ij * v_kj).sum(-1)
        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))

        g['triplets'] = triplets
        g['angle_feat'] = gaussian_expand_angle(theta)

# ═══════════════════════════════════════════════════════════════
# SCATTER OPS
# ═══════════════════════════════════════════════════════════════

def scatter_sum(src, idx, dim_size):
    out = src.new_zeros(dim_size, src.shape[-1])
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
    return out

def scatter_mean(src, idx, dim_size):
    s = scatter_sum(src, idx, dim_size)
    c = scatter_sum(torch.ones_like(src[:,:1]), idx, dim_size).clamp(min=1)
    return s / c

# ═══════════════════════════════════════════════════════════════
# ANGULAR GNN LAYER (Unshared)
#   The key difference from V3.1: edges are UPDATED using the
#   angles between neighboring bonds BEFORE node messages.
# ═══════════════════════════════════════════════════════════════

class AngularGNNLayer(nn.Module):
    """
    Two-phase message passing:
      Phase 1 (Edge Update): For each edge i→j, gather angular info
          from all other edges k→j sharing the same destination j.
          edge_ij += Σ_k  msg([edge_ij, edge_kj, angle(ij,kj)])
      Phase 2 (Node Update): Standard gated message passing using
          the angle-enriched edge features.

    This gives the GNN explicit knowledge of bond ANGLES, not just
    bond distances. Critical for phonon force constants.
    """

    def __init__(self, d, n_angle=N_ANGLE_GAUSS):
        super().__init__()
        # Phase 1: Edge update from angular triplets
        self.edge_msg  = nn.Sequential(nn.Linear(d*2 + n_angle, d), nn.SiLU(), nn.Linear(d, d))
        self.edge_gate = nn.Sequential(nn.Linear(d*2 + n_angle, d), nn.Sigmoid())
        self.edge_ln   = nn.LayerNorm(d)

        # Phase 2: Node update from angle-enriched edges
        self.msg  = nn.Sequential(nn.Linear(d*3, d), nn.SiLU(), nn.Linear(d, d))
        self.gate = nn.Sequential(nn.Linear(d*3, d), nn.Sigmoid())
        self.up   = nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.SiLU())

    def forward(self, x, ei, edge_h, triplets, angle_feat):
        # ── Phase 1: Edge Update (angular) ───────────────────
        if triplets.shape[1] > 0:
            ij_h = edge_h[triplets[0]]          # (nt, d)
            kj_h = edge_h[triplets[1]]          # (nt, d)
            ang_inp = torch.cat([ij_h, kj_h, angle_feat], -1)  # (nt, 2d+8)
            ang_msg = self.edge_msg(ang_inp) * self.edge_gate(ang_inp)
            # Scatter angular messages back to edge ij
            edge_update = edge_h.new_zeros(edge_h.size(0), edge_h.size(1))
            edge_update.scatter_add_(0, triplets[0].unsqueeze(-1).expand_as(ang_msg), ang_msg)
            edge_h = self.edge_ln(edge_h + edge_update)

        # ── Phase 2: Node Update (with angle-enriched edges) ─
        inp = torch.cat([x[ei[0]], x[ei[1]], edge_h], -1)
        msg = self.msg(inp) * self.gate(inp)
        agg = scatter_sum(msg, ei[1], x.size(0))
        x = x + self.up(torch.cat([x, agg], -1))

        return x, edge_h

# ═══════════════════════════════════════════════════════════════
# V4 MODEL  (~180K params)
# ═══════════════════════════════════════════════════════════════

class PhononV4(nn.Module):
    """
    Stage 1 — Angular Sensors (3 Unshared AngularGNN Layers):
        Edges learn from their angular relationships with neighboring bonds.
        Nodes receive messages from angle-enriched edges.
        Output: N per-atom tokens with deep angular physics.

    Stage 2 — The Brain (16 Shared TRM Cycles):
        Same as V3.1: SA + CA + z/y update with Channel C.
    """

    def __init__(self, comp_dim, d=D, d_gnn=D_GNN, n_gnn=N_GNN, n_cycles=N_CYCLES):
        super().__init__()
        self.d = d
        self.n_cycles = n_cycles
        self.n_extra = comp_dim - 343

        # ── Physics table ────────────────────────────────────
        self.register_buffer('phys_table', _build_phys_table())

        # ── GNN Input Projections ────────────────────────────
        self.atom_embed = nn.Embedding(103, d_gnn)
        self.phys_proj  = nn.Linear(6, d_gnn)
        self.coord_proj = nn.Linear(1, d_gnn)
        self.edge_enc   = nn.Linear(N_GAUSS, d_gnn)
        self.vec_enc    = nn.Linear(3, d_gnn)

        # ── 3 Unshared Angular GNN Layers ────────────────────
        self.gnn_layers = nn.ModuleList([AngularGNNLayer(d_gnn) for _ in range(n_gnn)])
        self.gnn_out = nn.Sequential(nn.Linear(d_gnn, d), nn.LayerNorm(d), nn.SiLU())

        # ── Composition Token Projections ────────────────────
        self.tok_proj   = nn.Linear(6, d)
        self.extra_proj = nn.Linear(self.n_extra, d)
        self.m2v_proj   = nn.Linear(200, d)

        # ── Channel C ────────────────────────────────────────
        self.struct_proj = nn.Linear(11, d)

        # ── Token Type Embeddings ────────────────────────────
        self.type_embed = nn.Embedding(2, d)

        # ── Shared Self-Attention ────────────────────────────
        self.sa   = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
        self.sa_n = nn.LayerNorm(d)
        self.sa_ff = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1),
                                   nn.Linear(d*2, d))
        self.sa_fn = nn.LayerNorm(d)

        # ── Shared Cross-Attention ───────────────────────────
        self.ca   = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
        self.ca_n = nn.LayerNorm(d)

        # ── Shared TRM State ─────────────────────────────────
        self.z_up = nn.Sequential(nn.Linear(d*4, d), nn.SiLU(), nn.Linear(d, d))
        self.y_up = nn.Sequential(nn.Linear(d*2, d), nn.SiLU(), nn.Linear(d, d))

        # ── Output ───────────────────────────────────────────
        self.head = nn.Sequential(nn.Linear(d, d//2), nn.SiLU(), nn.Linear(d//2, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, comp, g, deep_supervision=False):
        B   = g['n_crystals']
        ei  = g['ei']
        dev = comp.device

        # ══════════════════════════════════════════════════════
        #  STAGE 1: ANGULAR SENSORS (3 Unshared Layers)
        # ══════════════════════════════════════════════════════

        phys = self.phys_table[g['z']]
        atoms = (self.atom_embed(g['z'])
                 + self.phys_proj(phys)
                 + self.coord_proj(g['coord'].unsqueeze(-1)))

        edge_h = self.edge_enc(g['ef']) * torch.tanh(self.vec_enc(g['ev']))

        triplets   = g['triplets']
        angle_feat = g['angle_feat']

        for layer in self.gnn_layers:
            atoms, edge_h = layer(atoms, ei, edge_h, triplets, angle_feat)

        atoms = self.gnn_out(atoms)

        # ══════════════════════════════════════════════════════
        #  PAD ATOMS → (B, max_atoms, d) + mask
        # ══════════════════════════════════════════════════════
        n_atoms = g['n_atoms']
        ma = max(n_atoms)
        atom_tok = atoms.new_zeros(B, ma, self.d)
        atom_mask = torch.ones(B, ma, dtype=torch.bool, device=dev)
        off = 0
        for i, na in enumerate(n_atoms):
            atom_tok[i, :na] = atoms[off:off+na]
            atom_mask[i, :na] = False
            off += na

        # ══════════════════════════════════════════════════════
        #  COMPOSITION TOKENS (24 total)
        # ══════════════════════════════════════════════════════
        magpie  = comp[:, :132].view(B, 22, 6)
        extras  = comp[:, 132:132+self.n_extra]
        s_meta  = comp[:, 132+self.n_extra:132+self.n_extra+11]
        m2v_raw = comp[:, -200:]

        mag_tok  = self.tok_proj(magpie)
        ext_tok  = self.extra_proj(extras).unsqueeze(1)
        m2v_tok  = self.m2v_proj(m2v_raw).unsqueeze(1)
        comp_tok = torch.cat([mag_tok, ext_tok, m2v_tok], 1)

        comp_tok = comp_tok + self.type_embed.weight[0]
        atom_tok = atom_tok + self.type_embed.weight[1]

        all_tok = torch.cat([comp_tok, atom_tok], 1)
        full_mask = torch.cat([
            torch.zeros(B, 24, dtype=torch.bool, device=dev), atom_mask
        ], 1)

        struct_ctx = self.struct_proj(s_meta)

        # ══════════════════════════════════════════════════════
        #  STAGE 2: THE BRAIN (16 Shared TRM Cycles)
        # ══════════════════════════════════════════════════════
        z = torch.zeros(B, self.d, device=dev)
        y = torch.zeros(B, self.d, device=dev)
        preds = []

        for cyc in range(self.n_cycles):
            sa_out = self.sa(all_tok, all_tok, all_tok, key_padding_mask=full_mask)[0]
            all_tok = self.sa_n(all_tok + sa_out)
            all_tok = self.sa_fn(all_tok + self.sa_ff(all_tok))

            comp_tok = all_tok[:, :24]
            atom_cur = all_tok[:, 24:]

            ca_out = self.ca(comp_tok, atom_cur, atom_cur,
                             key_padding_mask=atom_mask)[0]
            comp_tok = self.ca_n(comp_tok + ca_out)

            all_tok = torch.cat([comp_tok, atom_cur], 1)

            xp = comp_tok.mean(dim=1)
            z = z + self.z_up(torch.cat([xp, struct_ctx, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))

            preds.append(self.head(y).squeeze(-1))

        return preds if deep_supervision else preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ═══════════════════════════════════════════════════════════════
# LOSS + UTILS
# ═══════════════════════════════════════════════════════════════

def deep_sup_loss(preds, targets):
    p = torch.stack(preds)
    w = torch.arange(1, p.shape[0]+1, device=p.device, dtype=p.dtype)
    w = w / w.sum()
    return (w * (p - targets.unsqueeze(0)).abs().mean(1)).sum()

def strat_split(t, vf=0.15, seed=42):
    bins = np.digitize(t, np.percentile(t, [25,50,75]))
    tr, vl = [], []
    rng = np.random.RandomState(seed)
    for b in range(4):
        m = np.where(bins==b)[0]
        if len(m)==0: continue
        n = max(1, int(len(m)*vf))
        c = rng.choice(m, n, replace=False)
        vl.extend(c.tolist()); tr.extend(np.setdiff1d(m, c).tolist())
    return np.array(tr), np.array(vl)

# ═══════════════════════════════════════════════════════════════
# DATALOADER (updated for angular features)
# ═══════════════════════════════════════════════════════════════

def collate(graphs, comp, targets, indices, device):
    zs, eis, efs, evs, bs, nas = [], [], [], [], [], []
    trips, afs, coords = [], [], []
    atom_off, edge_off = 0, 0

    for k, i in enumerate(indices):
        g = graphs[i]
        na = g['na']
        ne = g['ei'].shape[1]

        zs.append(g['z'])
        eis.append(g['ei'] + atom_off)
        efs.append(g['ef'])
        evs.append(g['ev'])
        bs.append(torch.full((na,), k, dtype=torch.long))
        nas.append(na)

        # Offset triplet edge-indices by the batched edge count
        trips.append(g['triplets'] + edge_off)
        afs.append(g['angle_feat'])
        coords.append(g['coord'])

        atom_off += na
        edge_off += ne

    return (
        comp[indices].to(device),
        {'z': torch.cat(zs).to(device), 'ei': torch.cat(eis, 1).to(device),
         'ef': torch.cat(efs).to(device), 'ev': torch.cat(evs).to(device),
         'batch': torch.cat(bs).to(device), 'n_crystals': len(indices),
         'n_atoms': nas,
         'triplets': torch.cat(trips, 1).to(device),
         'angle_feat': torch.cat(afs).to(device),
         'coord': torch.cat(coords).to(device)},
        targets[indices].to(device)
    )

class Loader:
    def __init__(self, graphs, comp, tgt, idx, bs, dev, shuf=False):
        self.g,self.c,self.t = graphs, comp, tgt
        self.idx,self.bs,self.dev,self.shuf = np.array(idx), bs, dev, shuf
    def __iter__(self):
        i = self.idx.copy()
        if self.shuf: np.random.shuffle(i)
        self._b = [i[j:j+self.bs] for j in range(0,len(i),self.bs)]
        self._p = 0; return self
    def __next__(self):
        if self._p >= len(self._b): raise StopIteration
        b = self._b[self._p]; self._p += 1
        return collate(self.g, self.c, self.t, b, self.dev)
    def __len__(self): return (len(self.idx)+self.bs-1)//self.bs

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train_fold(model, tr, vl, device, fold, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
    bv, bw = float('inf'), None
    bar = tqdm(range(EPOCHS), desc=f"  [V4|s{seed}] F{fold}/5", leave=False, ncols=120)
    for ep in bar:
        model.train(); te, tn = 0., 0
        for cb, gb, tb in tr:
            sp = model(cb, gb, True)
            loss = deep_sup_loss(sp, tb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            with torch.no_grad(): te += (sp[-1]-tb).abs().sum().item(); tn += len(tb)
        model.eval(); ve, vn = 0., 0
        with torch.inference_mode():
            for cb, gb, tb in vl:
                ve += (model(cb, gb)-tb).abs().sum().item(); vn += len(tb)
        tl, vl_ = te/tn, ve/vn; sch.step()
        if vl_ < bv: bv = vl_; bw = copy.deepcopy(model.state_dict())
        if ep%20==0 or ep==EPOCHS-1: bar.set_postfix(Best=f'{bv:.1f}',Tr=f'{tl:.1f}',V=f'{vl_:.1f}')
    model.load_state_dict(bw); return bv, model

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  TRIADS V4 — Angular GNN + Attention-TRM                ║
  ║  "The Phonon Simulator v4"                              ║
  ║                                                         ║
  ║  NEW: Edges learn from BOND ANGLES (i→j→k triplets)     ║
  ║  NEW: Coordination number per atom                      ║
  ║  Sensors: 3 unshared AngularGNN (d={D_GNN})               ║
  ║  Brain:   SA + CA + TRM ({N_CYCLES} shared cycles, d={D})     ║
  ║  Budget: <200K params                                   ║
  ╚══════════════════════════════════════════════════════════╝
    """)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True

    if os.path.exists("phonons_dataset.pt"):
        print("  Loading phonons_dataset.pt ...")
        data = torch.load("phonons_dataset.pt", weights_only=False)
    else:
        data = build_dataset()

    graphs, comp_all, tgt_all = data['graphs'], data['comp'], data['targets']
    cd, N = data['comp_dim'], len(graphs)
    print(f"  Dataset: {N} | comp_dim: {cd}")

    # ── Precompute angular features (one-time, ~30s) ─────────
    if 'triplets' not in graphs[0]:
        precompute_angular(graphs)
        print("  ✅ Angular features ready\n")
    else:
        print("  Angular features already present\n")

    tm = PhononV4(comp_dim=cd); np_ = tm.count_parameters()
    print(f"  Model: {np_:,} params")
    assert np_ < 250_000, f"OVER BUDGET: {np_:,}"
    print(f"  ✓ Under 250K ({np_:,})\n"); del tm

    kf = KFold(5, shuffle=True, random_state=FOLD_SEED); folds = list(kf.split(range(N)))
    for tv, te in folds: assert len(set(tv)&set(te))==0
    print("  5 folds: zero leakage ✓\n")

    tnp = tgt_all.numpy(); all_maes = {}
    for seed in SEEDS:
        print(f"  {'─'*3} Seed {seed} {'─'*45}"); ts = time.time(); sm = {}
        for fi, (tv, te) in enumerate(folds):
            tri, vli = strat_split(tnp[tv], 0.15, seed+fi)
            sc = StandardScaler().fit(comp_all[tv[tri]].numpy())
            cs = torch.tensor(np.nan_to_num(sc.transform(comp_all.numpy()),nan=0.).astype(np.float32))
            trl = Loader(graphs, cs, tgt_all, tv[tri], BATCH_SIZE, device, True)
            vll = Loader(graphs, cs, tgt_all, tv[vli], BATCH_SIZE, device, False)
            tel = Loader(graphs, cs, tgt_all, te, BATCH_SIZE, device, False)
            torch.manual_seed(seed+fi); np.random.seed(seed+fi)
            if device.type=='cuda': torch.cuda.manual_seed(seed+fi)
            model = PhononV4(comp_dim=cd).to(device)
            _, model = train_fold(model, trl, vll, device, fi+1, seed)
            model.eval(); ee, en_ = 0., 0
            with torch.inference_mode():
                for cb, gb, tb in tel:
                    ee += (model(cb,gb)-tb).abs().sum().item(); en_ += len(tb)
            mae = ee/en_; sm[fi] = mae; print(f"    Fold {fi+1}: MAE = {mae:.2f} cm⁻¹")
            del model
            if device.type=='cuda': torch.cuda.empty_cache()
        avg = np.mean(list(sm.values())); all_maes[seed] = sm
        print(f"\n  Seed {seed} avg: {avg:.2f} ({time.time()-ts:.0f}s)")

    fa = np.mean([np.mean(list(v.values())) for v in all_maes.values()])
    print(f"\n{'='*60}\n  FINAL — V4 Angular-GNN + Attn-TRM\n{'='*60}")
    print(f"\n  {'Model':<40} {'MAE':>10}\n  {'─'*52}")
    for n, v in sorted(BASELINES.items(), key=lambda x:x[1]):
        print(f"  {n:<40} {v:>10.2f}{' ← BEATEN!' if fa<v else ''}")
    print(f"  {'V4 AngGNN-SA-CA-TRM ('+str(np_//1000)+'K)':<40} {fa:>10.2f} ← US")
    print(f"  {'─'*52}\n  Time: {(time.time()-t0)/60:.1f} min")
    with open('phonons_v4_results.json','w') as f:
        json.dump({'model':'V4-AngularGNN-SA-CA-TRM','params':np_,'final_avg':round(fa,2),
                   'per_fold':{str(s):{str(k):round(v,2) for k,v in m.items()}
                               for s,m in all_maes.items()}},f,indent=2)
    print("  Saved: phonons_v4_results.json")

if __name__ == '__main__':
    main()
