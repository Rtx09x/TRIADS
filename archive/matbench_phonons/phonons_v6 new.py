"""
+=============================================================+
|  TRIADS V6 — Graph-TRM + Order 3 Dihedral MP                |
|                                                              |
|  Single model: Fixed 12 cycles, gated residuals              |
|  d=64, 4 heads, 3-order message passing, deep supervision    |
|  SWA last 50 ep | 200 epochs                                 |
|                                                              |
|  Loads: phonons_v6_dataset.pt                                |
+=============================================================+

DEPENDENCIES (dataset already pre-computed, no matminer needed):
    pip install torch numpy scikit-learn tqdm
    (all pre-installed on Kaggle)

USAGE:
    python phonons_v6.py
"""

import os, copy, json, time, math, warnings, threading, zipfile
from collections import defaultdict
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.preprocessing import StandardScaler

# Notebook dashboard (IPython is always available on Kaggle)
try:
    from IPython.display import display, HTML, clear_output
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

D             = 64
N_HEADS       = 4
N_WARMUP      = 1        # 1 unshared warm-up (param budget)
N_ANGLE_RBF   = 8
N_DIHED_RBF   = 8        # Order 3 dihedral RBF dim
DROPOUT       = 0.1
BATCH_SIZE    = 32
MAX_QUADS_B   = 512         # cap quads/crystal in collator (OOM guard)
EPOCHS        = 200
SWA_START     = 150
LR            = 5e-4
WD            = 1e-4
SEEDS         = [42]

# Fixed cycles (no halting — let the system reign)
CYCLES        = 12

BASELINES = {
    'MEGNet': 28.76, 'ALIGNN': 29.34, 'MODNet': 45.39,
    'CrabNet': 47.09, 'V6-GatedTRM (d=64, 8c)': 41.91,
    'V6-GateHalt (d=56, 4-16c)': 43.29,
    'TRIADS V4': 56.33, 'TRIADS V3.1': 63.00,
    'TRIADS V1': 71.82, 'Dummy': 323.76,
}


# ═══════════════════════════════════════════════════════════════
# SCATTER
# ═══════════════════════════════════════════════════════════════

def scatter_sum(src, idx, dim_size):
    out = torch.zeros(dim_size, src.shape[-1], dtype=src.dtype, device=src.device)
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
    return out


# ═══════════════════════════════════════════════════════════════
# COLLATION + DATALOADER (with Order 3)
# ═══════════════════════════════════════════════════════════════

def collate(graphs, comp, glob_phys, targets, indices, device):
    az, af = [], []
    ei, rb, vc, ph = [], [], [], []
    tr, an = [], []
    qi, dh = [], []          # Order 3: quad_index, dihedral_rbf
    ba, na_list = [], []
    a_off, e_off, t_off = 0, 0, 0

    for k, i in enumerate(indices):
        g = graphs[i]
        na, ne, nt = g['n_atoms'], g['n_edges'], g['n_triplets']
        az.append(g['atom_z'])
        af.append(g['atom_features'])
        ei.append(g['edge_index'] + a_off)
        rb.append(g['edge_rbf']); vc.append(g['edge_vec']); ph.append(g['edge_physics'])
        tr.append(g['triplet_index'] + e_off)
        an.append(g['angle_rbf'])
        # Order 3 — subsample quads to cap memory
        nq = g['quad_index'].shape[1]
        if nq > MAX_QUADS_B:
            sel = torch.randperm(nq)[:MAX_QUADS_B]
            qi.append(g['quad_index'][:, sel] + t_off)
            dh.append(g['dihedral_rbf'][sel])
        else:
            qi.append(g['quad_index'] + t_off)
            dh.append(g['dihedral_rbf'])
        ba.append(torch.full((na,), k, dtype=torch.long))
        na_list.append(na)
        a_off += na; e_off += ne; t_off += nt

    return (
        comp[indices].to(device),
        glob_phys[indices].to(device),
        {
            'atom_z': torch.cat(az).to(device),
            'atom_feat': torch.cat(af).to(device),
            'ei': torch.cat(ei, 1).to(device),
            'rbf': torch.cat(rb).to(device),
            'vec': torch.cat(vc).to(device),
            'phys': torch.cat(ph).to(device),
            'triplets': torch.cat(tr, 1).to(device),
            'angle_feat': torch.cat(an).to(device),
            'quads': torch.cat(qi, 1).to(device),
            'dihedral_feat': torch.cat(dh).to(device),
            'batch': torch.cat(ba).to(device),
            'n_crystals': len(indices),
            'n_atoms': na_list,
        },
        targets[indices].to(device),
    )


class Loader:
    def __init__(self, graphs, comp, gp, tgt, idx, bs, dev, shuf=False):
        self.g, self.c, self.gp, self.t = graphs, comp, gp, tgt
        self.idx, self.bs, self.dev, self.shuf = np.array(idx), bs, dev, shuf

    def __iter__(self):
        i = self.idx.copy()
        if self.shuf: np.random.shuffle(i)
        self._b = [i[j:j+self.bs] for j in range(0, len(i), self.bs)]
        self._p = 0; return self

    def __next__(self):
        if self._p >= len(self._b): raise StopIteration
        b = self._b[self._p]; self._p += 1
        return collate(self.g, self.c, self.gp, self.t, b, self.dev)

    def __len__(self): return (len(self.idx) + self.bs - 1) // self.bs


# ═══════════════════════════════════════════════════════════════
# GRAPH MESSAGE PASSING LAYER (3-Order: Dihedral → Bond → Atom)
# ═══════════════════════════════════════════════════════════════

class GraphMPLayer(nn.Module):
    """3-order message passing: Dihedral → Bond → Atom."""

    def __init__(self, d, n_angle=N_ANGLE_RBF, n_dihed=N_DIHED_RBF, dropout=DROPOUT):
        super().__init__()
        # Phase 0: Angle update from dihedral neighbors (Order 3)
        self.dihed_msg  = nn.Sequential(nn.Linear(d*2 + n_dihed, d), nn.SiLU())
        self.dihed_gate = nn.Sequential(nn.Linear(d*2 + n_dihed, d), nn.Sigmoid())
        self.dihed_up   = nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(dropout))

        # Phase 1: Bond update from angular neighbors (Order 2)
        self.bond_msg  = nn.Sequential(nn.Linear(d*2 + n_angle, d), nn.SiLU())
        self.bond_gate = nn.Sequential(nn.Linear(d*2 + n_angle, d), nn.Sigmoid())
        self.bond_up   = nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(dropout))

        # Phase 2: Atom update from bonds (Order 1)
        self.atom_msg  = nn.Sequential(nn.Linear(d*3, d), nn.SiLU())
        self.atom_gate = nn.Sequential(nn.Linear(d*3, d), nn.Sigmoid())
        self.atom_up   = nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(dropout))

    def forward(self, atoms, bonds, ei, triplets, angle_feat, quads, dihedral_feat):
        # ── Phase 0: Angles learn from dihedral neighbors (Order 3) ──
        # We need per-angle representations. Derive them from the bond pair:
        if triplets.shape[1] > 0 and quads.shape[1] > 0:
            # Angle representations = concat of the two bonds forming the angle
            angle_reps = bonds[triplets[0]] + bonds[triplets[1]]  # [n_triplets, d]

            # Dihedral message passing: angle ← dihedral neighbors
            a_src = angle_reps[quads[0]]
            a_dst = angle_reps[quads[1]]
            inp_d = torch.cat([a_src, a_dst, dihedral_feat], -1)
            msg_d = self.dihed_msg(inp_d) * self.dihed_gate(inp_d)
            agg_d = torch.zeros(angle_reps.size(0), angle_reps.size(1),
                                dtype=torch.float32, device=msg_d.device)
            agg_d.scatter_add_(0, quads[0].unsqueeze(-1).expand_as(msg_d), msg_d)
            angle_reps = angle_reps + self.dihed_up(torch.cat([angle_reps, agg_d], -1))

            # Feed enriched angle info back into bonds via the existing triplet structure
            # Each triplet (bond_ij, bond_kj) now carries dihedral-enriched angle context
            # We scatter the angle updates back to bonds[triplet[0]]
            bond_dihed_agg = torch.zeros_like(bonds)
            bond_dihed_agg.scatter_add_(0, triplets[0].unsqueeze(-1).expand_as(angle_reps), angle_reps)
            bonds = bonds + bond_dihed_agg * 0.1  # gentle injection

        # ── Phase 1: Bonds learn from angular neighbors (Order 2) ──
        if triplets.shape[1] > 0:
            b_ij, b_kj = bonds[triplets[0]], bonds[triplets[1]]
            inp = torch.cat([b_ij, b_kj, angle_feat], -1)
            msg = self.bond_msg(inp) * self.bond_gate(inp)
            agg = torch.zeros(bonds.size(0), bonds.size(1), dtype=torch.float32, device=msg.device)
            agg.scatter_add_(0, triplets[0].unsqueeze(-1).expand_as(msg), msg)
            bonds = bonds + self.bond_up(torch.cat([bonds, agg], -1))

        # ── Phase 2: Atoms aggregate from bonds (Order 1) ──
        inp = torch.cat([atoms[ei[0]], atoms[ei[1]], bonds], -1)
        msg = self.atom_msg(inp) * self.atom_gate(inp)
        agg = scatter_sum(msg, ei[1], atoms.size(0))
        atoms = atoms + self.atom_up(torch.cat([atoms, agg], -1))
        return atoms, bonds


# ═══════════════════════════════════════════════════════════════
# PHONON V6 MODEL (Single Model, Fixed Cycles, Order 3)
# ═══════════════════════════════════════════════════════════════

class PhononV6(nn.Module):
    """
    Graph Attention TRM for phonon prediction.
    Fixed n_cycles cycles, gated residuals, 3-order message passing.
    """

    def __init__(self, comp_dim, global_phys_dim=15, d=D,
                 n_cycles=CYCLES, n_warmup=N_WARMUP,
                 n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        self.d = d
        self.total_cycles = n_cycles

        # Feature layout (from V6 dataset: 132 magpie + extras + 11 struct + 200 m2v)
        self.n_magpie = 132
        self.n_extra = comp_dim - 132 - 11 - 200
        self.n_comp_tokens = 22 + 1 + 1  # 22 magpie + 1 extra + 1 m2v = 24

        # ── Input Encoding ────────────────────────────────────
        self.atom_embed = nn.Embedding(103, d)
        self.atom_feat_proj = nn.Linear(18, d)
        self.rbf_enc  = nn.Linear(40, d)
        self.vec_enc  = nn.Linear(3, d)
        self.phys_enc = nn.Linear(8, d)

        # ── Composition Token Projections ─────────────────────
        self.magpie_proj = nn.Linear(6, d)
        self.extra_proj  = nn.Linear(max(self.n_extra, 1), d)
        self.m2v_proj    = nn.Linear(200, d)

        # ── Context (structural + global physics) ─────────────
        self.ctx_proj = nn.Linear(11 + global_phys_dim, d)

        # ── Token Type Embeddings ─────────────────────────────
        self.type_embed = nn.Embedding(2, d)

        # ── Warm-up Layers (unshared) ─────────────────────────
        self.warmup = nn.ModuleList([
            GraphMPLayer(d, N_ANGLE_RBF, N_DIHED_RBF, dropout) for _ in range(n_warmup)
        ])
        self.warmup_out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.SiLU())

        # ── Shared TRM Block ──────────────────────────────────
        self.trm_gnn = GraphMPLayer(d, N_ANGLE_RBF, N_DIHED_RBF, dropout)

        # Self-Attention
        self.sa   = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.sa_n = nn.LayerNorm(d)
        self.sa_ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.sa_fn = nn.LayerNorm(d)

        # Cross-Attention
        self.ca   = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d)

        # ── State Update (Gated Residuals) ───────────────────
        self.z_proj = nn.Linear(d*3, d)
        self.z_up   = nn.Sequential(nn.Linear(d*2, d), nn.SiLU(), nn.Linear(d, d))
        self.z_gate = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())
        self.y_up   = nn.Sequential(nn.Linear(d*2, d), nn.SiLU(), nn.Linear(d, d))
        self.y_gate = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())

        # ── Output Head ───────────────────────────────────────
        self.head = nn.Sequential(nn.Linear(d, d//2), nn.SiLU(), nn.Linear(d//2, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, comp, glob_phys, g, deep_supervision=False):
        B   = g['n_crystals']
        ei  = g['ei']
        dev = comp.device

        # ══════════════════════════════════════════════════════
        #  INPUT ENCODING
        # ══════════════════════════════════════════════════════

        atoms = self.atom_embed(g['atom_z'].clamp(0, 102)) + self.atom_feat_proj(g['atom_feat'])
        bonds = self.rbf_enc(g['rbf']) * torch.tanh(self.vec_enc(g['vec'])) + self.phys_enc(g['phys'])

        triplets     = g['triplets']
        angle_feat   = g['angle_feat']
        quads        = g['quads']
        dihedral_feat = g['dihedral_feat']

        # ══════════════════════════════════════════════════════
        #  WARM-UP (unshared graph layers)
        # ══════════════════════════════════════════════════════

        for layer in self.warmup:
            atoms, bonds = layer(atoms, bonds, ei, triplets, angle_feat, quads, dihedral_feat)
        atoms = self.warmup_out(atoms)

        # ══════════════════════════════════════════════════════
        #  COMPOSITION TOKENS (24 total)
        # ══════════════════════════════════════════════════════

        magpie = comp[:, :132].view(B, 22, 6)
        extras = comp[:, 132:132+self.n_extra]
        s_meta = comp[:, 132+self.n_extra:132+self.n_extra+11]
        m2v    = comp[:, -200:]

        mag_tok = self.magpie_proj(magpie)                    # [B, 22, d]
        ext_tok = self.extra_proj(extras).unsqueeze(1)        # [B, 1, d]
        m2v_tok = self.m2v_proj(m2v).unsqueeze(1)             # [B, 1, d]
        comp_tok = torch.cat([mag_tok, ext_tok, m2v_tok], 1)  # [B, 24, d]

        comp_tok = comp_tok + self.type_embed.weight[0]

        # Context vector (structural + global physics)
        ctx = self.ctx_proj(torch.cat([s_meta, glob_phys], -1))  # [B, d]

        # ══════════════════════════════════════════════════════
        #  TRM REASONING LOOP (12 fixed cycles)
        # ══════════════════════════════════════════════════════

        z = torch.zeros(B, self.d, device=dev)
        y = torch.zeros(B, self.d, device=dev)
        preds = []
        n_atoms = g['n_atoms']

        for cyc in range(self.total_cycles):
            # ── Phase 1+2+3: 3-Order Graph MP (shared weights) ─
            atoms, bonds = self.trm_gnn(atoms, bonds, ei, triplets, angle_feat, quads, dihedral_feat)

            # ── Pad atoms for attention ───────────────────────
            ma = max(n_atoms)
            atom_tok = atoms.new_zeros(B, ma, self.d)
            atom_mask = torch.ones(B, ma, dtype=torch.bool, device=dev)
            off = 0
            for i, na in enumerate(n_atoms):
                atom_tok[i, :na] = atoms[off:off+na]
                atom_mask[i, :na] = False
                off += na
            atom_tok = atom_tok + self.type_embed.weight[1]

            # ── Joint Self-Attention ──────────────────────────
            all_tok = torch.cat([comp_tok, atom_tok], 1)
            full_mask = torch.cat([
                torch.zeros(B, self.n_comp_tokens, dtype=torch.bool, device=dev),
                atom_mask
            ], 1)

            sa_out = self.sa(all_tok, all_tok, all_tok, key_padding_mask=full_mask)[0]
            all_tok = self.sa_n(all_tok + sa_out)
            all_tok = self.sa_fn(all_tok + self.sa_ff(all_tok))

            comp_tok = all_tok[:, :self.n_comp_tokens]
            atom_tok = all_tok[:, self.n_comp_tokens:]

            # ── Cross-Attention (comp queries atoms) ──────────
            ca_out = self.ca(comp_tok, atom_tok, atom_tok, key_padding_mask=atom_mask)[0]
            comp_tok = self.ca_n(comp_tok + ca_out)

            # ── Unpad atoms back to flat ──────────────────────
            parts = [atom_tok[i, :n_atoms[i]] for i in range(B)]
            atoms = torch.cat(parts, 0)

            # ── State Update (Gated Residuals) ────────────────
            xp = comp_tok.mean(dim=1)  # [B, d]

            z_inp = self.z_proj(torch.cat([xp, ctx, y], -1))
            z_cand = self.z_up(torch.cat([z_inp, z], -1))
            z_g = self.z_gate(torch.cat([z_inp, z], -1))
            z = z + z_g * z_cand

            y_cand = self.y_up(torch.cat([y, z], -1))
            y_g = self.y_gate(torch.cat([y, z], -1))
            y = y + y_g * y_cand

            preds.append(self.head(y).squeeze(-1))

        return preds if deep_supervision else preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def deep_sup_loss(preds_list, targets):
    """Linearly-weighted deep supervision: later cycles get more weight."""
    p = torch.stack(preds_list)
    w = torch.arange(1, p.shape[0]+1, device=p.device, dtype=p.dtype)
    w = w / w.sum()
    return (w * (p - targets.unsqueeze(0)).abs().mean(1)).sum()


# ═══════════════════════════════════════════════════════════════
# STRATIFIED SPLIT (within train fold → train/val)
# ═══════════════════════════════════════════════════════════════

def strat_split(t, vf=0.15, seed=42):
    bins = np.digitize(t, np.percentile(t, [25, 50, 75]))
    tr, vl = [], []
    rng = np.random.RandomState(seed)
    for b in range(4):
        m = np.where(bins == b)[0]
        if len(m) == 0: continue
        n = max(1, int(len(m) * vf))
        c = rng.choice(m, n, replace=False)
        vl.extend(c.tolist())
        tr.extend(np.setdiff1d(m, c).tolist())
    return np.array(tr), np.array(vl)


# ═══════════════════════════════════════════════════════════════
# LIVE DASHBOARD (IPython HTML — works in Kaggle/Jupyter)
# ═══════════════════════════════════════════════════════════════

_print_lock = threading.Lock()

_dash_state = {
    'TRM': {'fold': 0, 'ep': 0, 'tr': float('inf'), 'val': float('inf'),
            'best': float('inf'), 'best_ep': 0, 'lr': 0., 'eta_m': 0,
            'ep_s': 0., 'swa': False, 'done': False, 'test_mae': None},
}
_dash_log = []


def _log(msg):
    with _print_lock:
        _dash_log.append(msg)
        if not IN_NOTEBOOK:
            print(msg, flush=True)


def _render_html():
    """Build an HTML table from _dash_state + recent log lines."""
    css = (
        '<style>'
        '.tv6{font-family:monospace;font-size:13px;border-collapse:collapse;width:100%}'
        '.tv6 th{background:#1a1a2e;color:#e94560;padding:6px 10px;text-align:right;border-bottom:2px solid #e94560}'
        '.tv6 td{padding:5px 10px;text-align:right;border-bottom:1px solid #333}'
        '.tv6 tr:nth-child(odd){background:#16213e}'
        '.tv6 tr:nth-child(even){background:#0f3460}'
        '.tv6 td:first-child,.tv6 th:first-child{text-align:left;font-weight:bold;color:#e9c46a}'
        '.tv6 .best{color:#2ecc71;font-weight:bold}'
        '.tv6 .done{color:#2ecc71}'
        '.tv6 .swa{color:#9b59b6}'
        '.tv6 .training{color:#f39c12}'
        '.tv6 .waiting{color:#636e72}'
        '.logbox{font-family:monospace;font-size:12px;color:#dfe6e9;background:#0a0a0a;'
        'padding:8px 12px;margin-top:8px;border-radius:6px;max-height:200px;overflow-y:auto}'
        '</style>'
    )
    rows = ''
    for name, s in _dash_state.items():
        if s['done'] and s['test_mae']:
            status = f'<span class="done">✅ {s["test_mae"]:.1f}</span>'
        elif s['swa']:
            status = '<span class="swa">SWA</span>'
        elif s['ep'] == 0:
            status = '<span class="waiting">Waiting</span>'
        else:
            status = '<span class="training">▶ Training</span>'
        ep_str   = f"{s['ep']}/{EPOCHS}" if s['ep'] else '-'
        tr_str   = f"{s['tr']:.1f}" if s['tr'] < 1e6 else '-'
        val_str  = f"{s['val']:.1f}" if s['val'] < 1e6 else '-'
        best_str = f'<span class="best">{s["best"]:.1f}@{s["best_ep"]}</span>' if s['best'] < 1e6 else '-'
        lr_str   = f"{s['lr']:.0e}" if s['lr'] > 0 else '-'
        eps_str  = f"{s['ep_s']:.1f}" if s['ep_s'] > 0 else '-'
        eta_str  = f"{s['eta_m']:.0f}m" if s['eta_m'] > 0 else '-'
        fold_str = str(s['fold']) if s['fold'] else '-'
        rows += (f'<tr><td>{name}</td><td>{fold_str}</td><td>{ep_str}</td>'
                 f'<td>{tr_str}</td><td>{val_str}</td><td>{best_str}</td>'
                 f'<td>{lr_str}</td><td>{eps_str}</td><td>{eta_str}</td>'
                 f'<td>{status}</td></tr>')
    table = (
        f'{css}<h3 style="color:#e94560;font-family:monospace;margin:4px 0">⚡ TRIADS V6 — Live Dashboard</h3>'
        f'<table class="tv6"><tr><th>Model</th><th>Fold</th><th>Epoch</th>'
        f'<th>Train MAE</th><th>Val MAE</th><th>Best MAE</th>'
        f'<th>LR</th><th>s/ep</th><th>ETA</th><th>Status</th></tr>{rows}</table>'
    )
    if _dash_log:
        log_html = '<br>'.join(_dash_log[-8:])
        table += f'<div class="logbox">{log_html}</div>'
    return table


class Dashboard:
    """Background thread that re-renders an HTML table every 5s in-place."""
    def __init__(self):
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not IN_NOTEBOOK:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not IN_NOTEBOOK or self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=10)
        clear_output(wait=True)
        display(HTML(_render_html()))

    def _run(self):
        while not self._stop.is_set():
            try:
                clear_output(wait=True)
                display(HTML(_render_html()))
            except Exception:
                pass
            self._stop.wait(5)


_dashboard = Dashboard()


def train_fold_core(model, tr_loader, vl_loader, device, fold, seed,
                    model_name, tgt_mean=0., tgt_std=1., log_every=10):
    """
    Train one model on one device. Returns (best_val_mae, model_with_best_weights).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    WARMUP_EP = 10
    def lr_lambda(ep):
        if ep < WARMUP_EP: return (ep + 1) / WARMUP_EP
        progress = (ep - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
        return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - 1e-5/LR) + 1e-5/LR
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    swa_model = AveragedModel(model)
    swa_sch = SWALR(opt, swa_lr=1e-4)

    bv, bw, best_ep = float('inf'), None, 0
    fold_start = time.time()

    for ep in range(EPOCHS):
        ep_start = time.time()
        use_swa = ep >= SWA_START

        # ── TRAIN ─────────────────────────────────────────────
        model.train()
        te, tn = 0., 0
        for cb, gb, g_batch, tb in tr_loader:
            sp = model(cb, gb, g_batch, True)
            loss = deep_sup_loss(sp, tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                te += ((sp[-1] * tgt_std + tgt_mean) - (tb * tgt_std + tgt_mean)).abs().sum().item()
                tn += len(tb)

        if use_swa:
            swa_model.update_parameters(model)
            swa_sch.step()
        else:
            sch.step()

        # ── VALIDATE ──────────────────────────────────────────
        eval_m = swa_model if use_swa and ep == EPOCHS - 1 else model
        eval_m.eval()
        ve, vn = 0., 0
        with torch.inference_mode():
            for cb, gb, g_batch, tb in vl_loader:
                pred = eval_m(cb, gb, g_batch)
                ve += ((pred * tgt_std + tgt_mean) - (tb * tgt_std + tgt_mean)).abs().sum().item()
                vn += len(tb)

        train_mae = te / max(tn, 1)
        val_mae = ve / max(vn, 1)
        ep_time = time.time() - ep_start

        if val_mae < bv:
            bv = val_mae
            bw = copy.deepcopy(model.state_dict())
            best_ep = ep + 1

        # ── UPDATE DASHBOARD STATE ───────────────────────────
        lr_now = opt.param_groups[0]['lr']
        eta_m = (EPOCHS - ep - 1) * ep_time / 60
        _dash_state[model_name].update({
            'fold': fold, 'ep': ep + 1,
            'tr': train_mae, 'val': val_mae,
            'best': bv, 'best_ep': best_ep,
            'lr': lr_now, 'ep_s': ep_time,
            'eta_m': eta_m, 'swa': use_swa,
        })

        if not IN_NOTEBOOK and ((ep + 1) % log_every == 0 or ep == 0 or ep == EPOCHS - 1):
            swa_tag = ' SWA' if use_swa else ''
            _log(f"    [{model_name}|F{fold}] ep {ep+1:>3}/{EPOCHS}"
                 f" │ Tr={train_mae:>6.1f}  Val={val_mae:>6.1f}"
                 f"  Best={bv:>6.1f}@{best_ep:<3}"
                 f" │ lr={lr_now:.0e}{swa_tag}"
                 f" │ {ep_time:.1f}s/ep  ETA {eta_m:.0f}m")

    model.load_state_dict(bw)
    total_time = time.time() - fold_start
    _log(f"    [{model_name}|F{fold}] ✅ Done in {total_time/60:.1f}m │ Best Val MAE = {bv:.2f} @ epoch {best_ep}")

    return bv, model


def evaluate_model(model, test_loader, device, tgt_mean=0., tgt_std=1.):
    """Evaluate model MAE on test set (returns MAE in original scale)."""
    model.eval()
    ee, en_ = 0., 0
    with torch.inference_mode():
        for cb, gb, g_batch, tb in test_loader:
            pred = model(cb, gb, g_batch) * tgt_std + tgt_mean
            real = tb * tgt_std + tgt_mean
            ee += (pred - real).abs().sum().item()
            en_ += len(tb)
    return ee / max(en_, 1)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  TRIADS V6 — Graph-TRM + Order 3 Dihedral MP            ║
  ║                                                          ║
  ║  Fixed {CYCLES} cycles, d={D}, 3-order message passing           ║
  ║  Deep supervision │ SWA (last {EPOCHS-SWA_START} ep) │ {EPOCHS} ep          ║
  ╚══════════════════════════════════════════════════════════╝
    """)

    device = torch.device('cuda:0' if n_gpus > 0 else 'cpu')
    if n_gpus > 0:
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name} ({mem:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        print("  ⚠ No GPU — training will be slow")

    # ── LOAD DATASET ──────────────────────────────────────────
    kaggle_path = "/kaggle/input/datasets/rudratiwari0099x/phonons-training-dataset/phonons_v6_dataset.pt"
    local_path = "phonons_v6_dataset.pt"
    ds_path = kaggle_path if os.path.exists(kaggle_path) else local_path
    print(f"  Loading {ds_path}...")
    data = torch.load(ds_path, weights_only=False)
    graphs = data['graphs']
    comp_all = data['comp_features']
    glob_phys = data['global_physics']
    tgt_all = data['targets']
    fold_indices = data['fold_indices']
    N = data['n_samples']
    comp_dim = comp_all.shape[1]
    gp_dim = glob_phys.shape[1]
    print(f"  Dataset: {N} samples | comp_dim: {comp_dim} | global_phys: {gp_dim}")

    # ── VERIFY FOLDS ──────────────────────────────────────────
    for fi, (tr, te) in enumerate(fold_indices):
        assert len(set(tr) & set(te)) == 0, f"LEAK in fold {fi}!"
    print("  5 folds: zero leakage ✓")

    # ── MODEL SIZE CHECK ─────────────────────────────────────
    m_test = PhononV6(comp_dim, gp_dim, n_cycles=CYCLES)
    n_params = m_test.count_parameters()
    print(f"  Model (Graph-TRM + Order 3): {n_params:,} params")
    del m_test
    print()

    # ── TRAINING ──────────────────────────────────────────────
    tnp = tgt_all.numpy()
    results = {}

    _dashboard.start()
    try:
        for seed in SEEDS:
            print(f"  {'═'*3} Seed {seed} {'═'*55}")
            ts = time.time()
            fold_maes = {}

            for fi, (tv_idx, te_idx) in enumerate(fold_indices):
                tv_idx, te_idx = np.array(tv_idx), np.array(te_idx)
                print(f"\n  ┌─ Fold {fi+1}/5 {'─'*50}")

                tri, vli = strat_split(tnp[tv_idx], 0.15, seed + fi)

                tgt_mean = float(tgt_all[tv_idx[tri]].mean())
                tgt_std  = float(tgt_all[tv_idx[tri]].std()) + 1e-8
                tgt_norm = (tgt_all - tgt_mean) / tgt_std
                print(f"  │ Target norm: mean={tgt_mean:.1f} std={tgt_std:.1f}")

                sc = StandardScaler().fit(comp_all[tv_idx[tri]].numpy())
                cs = torch.tensor(
                    np.nan_to_num(sc.transform(comp_all.numpy()), nan=0.).astype(np.float32)
                )
                sc_gp = StandardScaler().fit(glob_phys[tv_idx[tri]].numpy())
                gps = torch.tensor(
                    np.nan_to_num(sc_gp.transform(glob_phys.numpy()), nan=0.).astype(np.float32)
                )

                torch.manual_seed(seed + fi)
                np.random.seed(seed + fi)
                if n_gpus > 0:
                    torch.cuda.manual_seed_all(seed + fi)

                model = PhononV6(comp_dim, gp_dim, n_cycles=CYCLES).to(device)

                trl = Loader(graphs, cs, gps, tgt_norm, tv_idx[tri], BATCH_SIZE, device, True)
                vll = Loader(graphs, cs, gps, tgt_norm, tv_idx[vli], BATCH_SIZE, device, False)
                tel = Loader(graphs, cs, gps, tgt_norm, te_idx,      BATCH_SIZE, device, False)

                _dash_state['TRM']['done'] = False

                _, best_model = train_fold_core(
                    model, trl, vll, device, fi+1, seed, "TRM",
                    tgt_mean=tgt_mean, tgt_std=tgt_std
                )
                mae = evaluate_model(best_model, tel, device, tgt_mean, tgt_std)
                fold_maes[fi] = mae
                _dash_state['TRM']['test_mae'] = mae
                _dash_state['TRM']['done'] = True
                _log(f"    [TRM|F{fi+1}] 🏆 Test MAE = {mae:.2f} cm⁻¹")

                # Save fold weights
                wt_path = f"v6_fold{fi+1}_seed{seed}.pt"
                torch.save({
                    'model_state_dict': best_model.state_dict(),
                    'fold': fi+1, 'seed': seed,
                    'test_mae': mae, 'val_mae': fold_maes[fi],
                    'tgt_mean': tgt_mean, 'tgt_std': tgt_std,
                    'config': {'d': D, 'cycles': CYCLES, 'n_heads': N_HEADS,
                               'comp_dim': comp_dim, 'gp_dim': gp_dim},
                }, wt_path)
                print(f"  │ Saved weights: {wt_path}")

                print(f"  └─ Fold {fi+1} done │ MAE = {fold_maes[fi]:.2f} cm⁻¹")

                del model, best_model
                if n_gpus > 0: torch.cuda.empty_cache()

            avg = np.mean(list(fold_maes.values()))
            results[seed] = fold_maes
            elapsed = time.time() - ts
            print(f"\n  Seed {seed} │ Avg MAE: {avg:.2f} │ {elapsed/60:.1f} min")

    finally:
        _dashboard.stop()

    # ── FINAL RESULTS ─────────────────────────────────────────
    fa = np.mean([np.mean(list(v.values())) for v in results.values()])

    print(f"""
{'='*62}
  FINAL RESULTS — V6 Graph-TRM + Order 3
{'='*62}

  {'Model':<45} {'MAE':>10}
  {'─'*57}""")
    for n, v in sorted(BASELINES.items(), key=lambda x: x[1]):
        beaten = ' ← BEATEN!' if fa < v else ''
        print(f"  {n:<45} {v:>10.2f}{beaten}")
    print(f"  {'V6 Order3-TRM ('+str(n_params//1000)+'K, '+str(CYCLES)+' cycles)':<45} {fa:>10.2f} ← OURS")
    print(f"  {'─'*57}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    # ── SAVE ──────────────────────────────────────────────────
    res = {
        'model': 'V6-Order3-TRM', 'params': n_params,
        'cycles': CYCLES,
        'avg_mae': round(fa, 2),
        'per_fold': {str(s): {str(k): round(v, 2) for k,v in m.items()}
                     for s,m in results.items()},
    }
    with open('phonons_v6_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print("  Saved: phonons_v6_results.json")

    # ── ZIP ALL FOLD WEIGHTS ──────────────────────────────────
    zip_name = 'phonons_v6_weights.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for seed in SEEDS:
            for fi in range(5):
                wt = f"v6_fold{fi+1}_seed{seed}.pt"
                if os.path.exists(wt):
                    zf.write(wt)
                    os.remove(wt)  # clean up individual files
        zf.write('phonons_v6_results.json')
    print(f"  📦 Zipped: {zip_name} ({os.path.getsize(zip_name)/1e6:.1f} MB)")
    print("  Download from Kaggle Output tab!\n")


if __name__ == '__main__':
    main()
