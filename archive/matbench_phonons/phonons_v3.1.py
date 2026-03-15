
"""
+=============================================================+
|  TRIADS V3.1 — GNN-Attention-TRM for matbench_phonons       |
|                                                             |
|  Architecture:                                              |
|    Stage 1: 3 Unshared GNN Layers ("Sensors")               |
|       → N per-atom tokens with ionic/electronic physics     |
|    Stage 2: 16 Shared TRM Cycles ("Brain")                  |
|       ① Self-Attention [24 comp + N atom tokens] + type_emb |
|       ② Cross-Attention  comp queries atoms specifically    |
|       ③ TRM z,y update with Channel C struct metadata       |
|                                                             |
|  ~175K params | d_gnn=48, d=64 | 16 cycles                  |
+=============================================================+
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

D       = 64          # TRM / token hidden dim
D_GNN   = 48          # GNN hidden dim (smaller, projected to D after)
N_GNN   = 3           # Unshared GNN layers
N_CYCLES = 16         # TRM reasoning cycles (shared weights)
BATCH_SIZE = 32
EPOCHS  = 350
LR      = 1e-3
SEEDS   = [42]
FOLD_SEED = 18012019
N_GAUSS  = 40
MAX_NBR  = 12
CUTOFF   = 8.0

BASELINES = {
    'MEGNet': 28.76, 'ALIGNN': 29.34, 'MODNet': 45.39,
    'CrabNet': 47.09, 'TRIADS V1': 71.82, 'Dummy': 323.76,
}

# ═══════════════════════════════════════════════════════════════
# PHYSICS LOOKUP TABLES (built once from pymatgen)
# ═══════════════════════════════════════════════════════════════

def _build_phys_table():
    """Build (103, 6) table: [1/√m, EN, radius, valence_e, group, period]."""
    from pymatgen.core.periodic_table import Element
    rows = [[0.]*6]  # Z=0 placeholder
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
# DATASET BUILDER (runs once, saves phonons_dataset.pt)
# ═══════════════════════════════════════════════════════════════

def gaussian_expand(d, n=N_GAUSS, lo=0., hi=CUTOFF):
    c = torch.linspace(lo, hi, n)
    g = 1. / ((hi - lo) / n) ** 2
    return torch.exp(-g * (d.unsqueeze(-1) - c) ** 2)

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
# GNN LAYER (Unshared — instantiated N_GNN times)
# ═══════════════════════════════════════════════════════════════

class GNNLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.msg  = nn.Sequential(nn.Linear(d*3, d), nn.SiLU(), nn.Linear(d, d))
        self.gate = nn.Sequential(nn.Linear(d*3, d), nn.Sigmoid())
        self.up   = nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.SiLU())

    def forward(self, x, ei, edge_h):
        inp = torch.cat([x[ei[0]], x[ei[1]], edge_h], -1)
        agg = scatter_sum(self.msg(inp) * self.gate(inp), ei[1], x.size(0))
        return x + self.up(torch.cat([x, agg], -1))

# ═══════════════════════════════════════════════════════════════
# V3.1 MODEL  (~175K params)
# ═══════════════════════════════════════════════════════════════

class PhononGNNTRM(nn.Module):
    """
    Stage 1 — The Sensors (3 Unshared GNN Layers):
        Process raw crystal geometry with per-atom ionic physics.
        Output: N distinct atom tokens (per-atom, NOT pooled).

    Stage 2 — The Brain (16 Shared TRM Cycles):
        Each cycle:
          ① Self-Attention [24 comp + N atom tokens] with type embeddings
          ② Cross-Attention: comp tokens query atom tokens
          ③ Pool → z_up(xp, struct_ctx, y, z) → y_up(y, z) → predict
    """

    def __init__(self, comp_dim, d=D, d_gnn=D_GNN, n_gnn=N_GNN, n_cycles=N_CYCLES):
        super().__init__()
        self.d = d
        self.n_cycles = n_cycles
        # Feature split: comp = [Magpie(132) | Extras(E) | Struct(11) | M2V(200)]
        self.n_extra = comp_dim - 343  # E = comp_dim - 132 - 11 - 200

        # ── Physics table (register_buffer → saved in state_dict) ──
        self.register_buffer('phys_table', _build_phys_table())

        # ── GNN Input Projections ────────────────────────────
        self.atom_embed = nn.Embedding(103, d_gnn)
        self.phys_proj  = nn.Linear(6, d_gnn)
        self.edge_enc   = nn.Linear(N_GAUSS, d_gnn)
        self.vec_enc    = nn.Linear(3, d_gnn)
        self.ang_enc    = nn.Linear(9, d_gnn)

        # ── 3 Unshared GNN Layers ────────────────────────────
        self.gnn_layers = nn.ModuleList([GNNLayer(d_gnn) for _ in range(n_gnn)])
        self.gnn_out = nn.Sequential(nn.Linear(d_gnn, d), nn.LayerNorm(d), nn.SiLU())

        # ── Composition Token Projections ────────────────────
        self.tok_proj   = nn.Linear(6, d)               # Magpie stat → d
        self.extra_proj = nn.Linear(self.n_extra, d)     # extras → 1 token
        self.m2v_proj   = nn.Linear(200, d)              # Mat2Vec → 1 token

        # ── Channel C: Structural Metadata → direct z_up ────
        self.struct_proj = nn.Linear(11, d)

        # ── Token Type Embeddings ────────────────────────────
        self.type_embed = nn.Embedding(2, d)  # 0=comp, 1=atom

        # ── Shared Self-Attention (inside loop) ──────────────
        self.sa   = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
        self.sa_n = nn.LayerNorm(d)
        self.sa_ff = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1),
                                   nn.Linear(d*2, d))
        self.sa_fn = nn.LayerNorm(d)

        # ── Shared Cross-Attention (inside loop) ─────────────
        self.ca   = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
        self.ca_n = nn.LayerNorm(d)

        # ── Shared TRM State Update (inside loop) ────────────
        # z_up takes [pooled(d) + struct_ctx(d) + y(d) + z(d)] = 4d
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
        #  STAGE 1: THE SENSORS (3 Unshared GNN Layers)
        # ══════════════════════════════════════════════════════

        # Per-atom physics from lookup table
        phys = self.phys_table[g['z']]                          # (N_total, 6)
        atoms = self.atom_embed(g['z']) + self.phys_proj(phys)  # (N_total, d_gnn)

        # Edge features
        edge_h = self.edge_enc(g['ef']) * torch.tanh(self.vec_enc(g['ev']))

        # Angular context (outer product of neighbor vectors → scatter to nodes)
        vo = (g['ev'].unsqueeze(-1) * g['ev'].unsqueeze(-2)).view(-1, 9)
        atoms = atoms + self.ang_enc(scatter_sum(vo, ei[1], atoms.size(0)))

        # 3 unshared message-passing layers
        for layer in self.gnn_layers:
            atoms = layer(atoms, ei, edge_h)

        # Project d_gnn → d
        atoms = self.gnn_out(atoms)  # (N_total, d)

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
        magpie   = comp[:, :132].view(B, 22, 6)
        extras   = comp[:, 132:132+self.n_extra]
        s_meta   = comp[:, 132+self.n_extra:132+self.n_extra+11]
        m2v_raw  = comp[:, -200:]

        mag_tok  = self.tok_proj(magpie)                      # (B, 22, d)
        ext_tok  = self.extra_proj(extras).unsqueeze(1)       # (B, 1, d)
        m2v_tok  = self.m2v_proj(m2v_raw).unsqueeze(1)        # (B, 1, d)
        comp_tok = torch.cat([mag_tok, ext_tok, m2v_tok], 1)  # (B, 24, d)

        # ── Type Embeddings ──────────────────────────────────
        comp_tok = comp_tok + self.type_embed.weight[0]
        atom_tok = atom_tok + self.type_embed.weight[1]

        # ── Full Token Set ───────────────────────────────────
        all_tok = torch.cat([comp_tok, atom_tok], 1)  # (B, 24+ma, d)
        full_mask = torch.cat([
            torch.zeros(B, 24, dtype=torch.bool, device=dev), atom_mask
        ], 1)  # (B, 24+ma)

        # ── Channel C: Structural Metadata ───────────────────
        struct_ctx = self.struct_proj(s_meta)  # (B, d)

        # ══════════════════════════════════════════════════════
        #  STAGE 2: THE BRAIN (16 Shared TRM Cycles)
        # ══════════════════════════════════════════════════════
        z = torch.zeros(B, self.d, device=dev)
        y = torch.zeros(B, self.d, device=dev)
        preds = []

        for cyc in range(self.n_cycles):
            # ① Self-Attention (all tokens see each other)
            sa_out = self.sa(all_tok, all_tok, all_tok, key_padding_mask=full_mask)[0]
            all_tok = self.sa_n(all_tok + sa_out)
            all_tok = self.sa_fn(all_tok + self.sa_ff(all_tok))

            # Split
            comp_tok = all_tok[:, :24]
            atom_cur = all_tok[:, 24:]

            # ② Cross-Attention (comp queries atoms)
            ca_out = self.ca(comp_tok, atom_cur, atom_cur,
                             key_padding_mask=atom_mask)[0]
            comp_tok = self.ca_n(comp_tok + ca_out)

            # Re-assemble (comp updated, atoms unchanged by CA)
            all_tok = torch.cat([comp_tok, atom_cur], 1)

            # ③ Pool comp tokens → TRM state update
            xp = comp_tok.mean(dim=1)  # (B, d)
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
# DATALOADER
# ═══════════════════════════════════════════════════════════════

def collate(graphs, comp, targets, indices, device):
    zs, eis, efs, evs, bs, nas = [], [], [], [], [], []
    off = 0
    for k, i in enumerate(indices):
        g = graphs[i]; na = g['na']
        zs.append(g['z']); eis.append(g['ei']+off)
        efs.append(g['ef']); evs.append(g['ev'])
        bs.append(torch.full((na,), k, dtype=torch.long))
        nas.append(na); off += na
    return (
        comp[indices].to(device),
        {'z': torch.cat(zs).to(device), 'ei': torch.cat(eis,1).to(device),
         'ef': torch.cat(efs).to(device), 'ev': torch.cat(evs).to(device),
         'batch': torch.cat(bs).to(device), 'n_crystals': len(indices),
         'n_atoms': nas},
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
    bar = tqdm(range(EPOCHS), desc=f"  [V3.1|s{seed}] F{fold}/5", leave=False, ncols=120)
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
  ║  TRIADS V3.1 — GNN-Attention-TRM  (Phonon Simulator)    ║
  ║                                                         ║
  ║  Sensors: 3 unshared GNN + per-atom ionic physics        ║
  ║  Brain:   SA + CA + TRM (16 shared cycles)              ║
  ║  Tokens:  24 comp + N atoms (padded, masked)            ║
  ║  d_gnn={D_GNN} | d={D} | budget <200K                    ║
  ╚══════════════════════════════════════════════════════════╝
    """)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True

    if os.path.exists("phonons_dataset.pt"):
        print("  Loading phonons_dataset.pt ..."); data = torch.load("phonons_dataset.pt", weights_only=False)
    else:
        data = build_dataset()

    graphs, comp_all, tgt_all = data['graphs'], data['comp'], data['targets']
    cd, N = data['comp_dim'], len(graphs)
    print(f"  Dataset: {N} | comp_dim: {cd}")

    tm = PhononGNNTRM(comp_dim=cd); np_ = tm.count_parameters()
    print(f"  Model: {np_:,} params"); assert np_ < 200_000, f"OVER: {np_:,}"; del tm
    print(f"  ✓ Under 200K\n")

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
            model = PhononGNNTRM(comp_dim=cd).to(device)
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
    print(f"\n{'='*60}\n  FINAL — V3.1 GNN-Attn-TRM\n{'='*60}")
    print(f"\n  {'Model':<40} {'MAE':>10}\n  {'─'*52}")
    for n, v in sorted(BASELINES.items(), key=lambda x:x[1]):
        print(f"  {n:<40} {v:>10.2f}{' ← BEATEN!' if fa<v else ''}")
    print(f"  {'V3.1 GNN-SA-CA-TRM ('+str(np_//1000)+'K)':<40} {fa:>10.2f} ← US")
    print(f"  {'─'*52}\n  Time: {(time.time()-t0)/60:.1f} min")
    with open('phonons_v31_results.json','w') as f:
        json.dump({'model':'V3.1-GNN-SA-CA-TRM','params':np_,'final_avg':round(fa,2),
                   'per_fold':{str(s):{str(k):round(v,2) for k,v in m.items()}
                               for s,m in all_maes.items()}},f,indent=2)
    print("  Saved: phonons_v31_results.json")

if __name__ == '__main__':
    main()
