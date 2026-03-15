"""
+=============================================================+
|  TRIADS V3 on matbench_phonons — SOTA Attempt              |
|  Angular Physics GNN + DeepHybridTRM Hybrid                |
|                                                             |
|  Features:                                                  |
|  1. Physics Injection: Atomic mass denominator (1/sqrt(m))  |
|  2. Angular Context: Implicit bond-angle symmetry tensor    |
|  3. Deep Gating: 6-layer Gated Message Passing              |
|                                                             |
|  Total: <120K params | Single Seed [42]                     |
|  Requires: phonons_dataset.pt (run build_phonons_dataset.py)|
+=============================================================+
"""

import torch, copy, time, json, shutil
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np

# V3 Architecture Configuration (<120K Total)
GNN_CFG = dict(d_gnn=40, n_gaussian=40, n_layers=6)
TRM_CFG = dict(d_attn=32, nhead=4, d_hidden=64, ff_dim=96, dropout=0.15, max_steps=12)
MATBENCH_FOLD_SEED = 18012019

# Global Physics Data (Denominators for ω ∝ sqrt(k/m))
MASSES = torch.tensor([1.0, 1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.18, 22.99, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.63, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 98, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.6, 126.9, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24, 145, 150.36, 151.96, 157.25, 158.93, 162.5, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2, 208.98, 209, 210, 222, 223, 226, 227, 232.04, 231.04, 238.03, 237, 244, 243, 247, 247, 251, 252, 257, 258, 259, 262])
INV_SQRT_M = (1.0 / torch.sqrt(MASSES + 1e-8)).unsqueeze(-1)

def scatter_sum(src, index, dim_size):
    out = torch.zeros(dim_size, src.shape[1], device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    return out

class AngularPhysicsGNN(nn.Module):
    def __init__(self, d_gnn=40, n_gaussian=40, n_layers=6):
        super().__init__()
        self.register_buffer('inv_sqrt_mass', INV_SQRT_M)
        self.z_embed = nn.Embedding(103, d_gnn)
        self.m_embed = nn.Linear(1, d_gnn)
        self.edge_enc = nn.Linear(n_gaussian, d_gnn)
        self.vec_enc = nn.Linear(3, d_gnn)
        self.ang_enc = nn.Linear(9, d_gnn)
        self.layers = nn.ModuleList([self._make_layer(d_gnn) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_gnn)

    def _make_layer(self, d):
        return nn.ModuleDict({
            'msg': nn.Sequential(nn.Linear(d*3, d), nn.SiLU(), nn.Linear(d, d)),
            'gate': nn.Sequential(nn.Linear(d*3, d), nn.Sigmoid()),
            'up': nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.SiLU())
        })

    def forward(self, z, idx, f, v, b, n):
        x = self.z_embed(z) + self.m_embed(self.inv_sqrt_mass[z])
        e = self.edge_enc(f) * torch.tanh(self.vec_enc(v))
        v_outer = (v.unsqueeze(-1) * v.unsqueeze(-2)).view(-1, 9)
        x = x + self.ang_enc(scatter_sum(v_outer, idx[1], x.size(0)))
        for layer in self.layers:
            inputs = torch.cat([x[idx[0]], x[idx[1]], e], -1)
            m = layer['msg'](inputs) * layer['gate'](inputs)
            agg = scatter_sum(m, idx[1], x.size(0))
            x = x + layer['up'](torch.cat([x, agg], -1))
        return scatter_sum(self.norm(x), b, n) / scatter_sum(torch.ones_like(x[:,:1]), b, n).clamp(min=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DeepHybridTRM(nn.Module):
    def __init__(self, n_extra=69, d_attn=32, nhead=4, d_hidden=64, ff_dim=96, dropout=0.15, max_steps=12):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.tok_proj = nn.Linear(6, d_attn); self.m2v_proj = nn.Linear(200, d_attn)
        self.sa = nn.MultiheadAttention(d_attn, nhead, dropout=dropout, batch_first=True)
        self.pool = nn.Linear(d_attn + n_extra, d_hidden)
        self.z_up = nn.Linear(d_hidden*3, d_hidden); self.y_up = nn.Linear(d_hidden*2, d_hidden)
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, x, deep=False):
        B = x.size(0); tok = self.tok_proj(x[:, :132].view(B, 22, 6)); ctx = self.m2v_proj(x[:, -200:]).unsqueeze(1)
        tok = tok + self.sa(tok, tok, tok)[0]; tok = tok + self.sa(tok, ctx, ctx)[0]
        xp = F.gelu(self.pool(torch.cat([tok.mean(1), x[:, 132:-200]], -1)))
        z = y = torch.zeros(B, self.D, device=x.device); preds = []
        for s in range(self.max_steps):
            z = z + F.silu(self.z_up(torch.cat([xp, y, z], -1)))
            y = y + F.silu(self.y_up(torch.cat([y, z], -1))); preds.append(self.head(y).squeeze(1))
        return preds if deep else preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class HybridModel(nn.Module):
    def __init__(self, gnn, trm):
        super().__init__(); self.gnn = gnn; self.trm = trm
    def forward(self, comp, g, deep=False):
        gnn_e = self.gnn(g['z'], g['idx'], g['f'], g['v'], g['b'], g['n'])
        return self.trm(torch.cat([comp[:, :161], gnn_e, comp[:, 161:]], 1), deep)

    def count_parameters(self):
        return self.gnn.count_parameters() + self.trm.count_parameters()

def strat_split(y, seed):
    bins = np.digitize(y, np.percentile(y, [25, 50, 75]))
    tr, vl = [], []
    for b in range(4):
        m = np.where(bins == b)[0]
        n = max(1, int(len(m) * 0.15))
        c = np.random.RandomState(seed).choice(m, n, replace=False)
        vl.extend(c); tr.extend(np.setdiff1d(m, c))
    return tr, vl

def run_benchmark():
    data = torch.load("phonons_dataset.pt", map_location='cpu'); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kf = KFold(n_splits=5, shuffle=True, random_state=MATBENCH_FOLD_SEED)
    maes = []
    for fi, (tv, te) in enumerate(kf.split(range(len(data['targets'])))):
        tri, vli = strat_split(data['targets'][tv], 42+fi)
        sc = StandardScaler().fit(data['comp_features'][tv][tri])
        def get_dl(idx):
            comp = torch.tensor(sc.transform(data['comp_features'][idx]), dtype=torch.float32)
            return [(comp[i:i+64], data['targets'][idx][i:i+64], [data['graphs'][j] for j in idx[i:i+64]]) for i in range(0, len(idx), 64)]
        
        tr_dl, vl_dl, te_dl = get_dl(tv[tri]), get_dl(tv[vli]), get_dl(te)
        mod = HybridModel(AngularPhysicsGNN(**GNN_CFG), DeepHybridTRM(n_extra=29+GNN_CFG['d_gnn'])).to(device)
        print(f"Fold {fi+1} | Params: {mod.count_parameters():,}")
        opt = torch.optim.AdamW(mod.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=5e-5)
        
        pbar = tqdm(range(300), desc=f"Fold {fi+1}")
        for ep in pbar:
            mod.train(); tl = 0
            for bc, bt, bg in tr_dl:
                z, idx, f, v, b, n = torch.cat([g['atom_numbers'] for g in bg]), [], [], [], [], len(bg)
                off = 0
                for i, g in enumerate(bg):
                    m = len(g['atom_numbers'])
                    idx.append(g['edge_index']+off); f.append(g['edge_feats']); v.append(g['edge_vectors'])
                    b.append(torch.full((m,), i, dtype=torch.long)); off += m
                gb = {'z':z.to(device), 'idx':torch.cat(idx, 1).to(device), 'f':torch.cat(f).to(device), 'v':torch.cat(v).to(device), 'b':torch.cat(b).to(device), 'n':n}
                preds = mod(bc.to(device), gb, True)
                loss = torch.stack([(p - bt.to(device)).abs().mean() for p in preds]).mean()
                opt.zero_grad(); loss.backward(); opt.step(); tl += (preds[-1]-bt.to(device)).abs().mean().item()
            sch.step(); mod.eval(); vl = 0
            with torch.no_grad():
                for bc, bt, bg in vl_dl:
                    z, idx, f, v, b, n = torch.cat([g['atom_numbers'] for g in bg]), [], [], [], [], len(bg)
                    off = 0
                    for i, g in enumerate(bg):
                        m = len(g['atom_numbers'])
                        idx.append(g['edge_index']+off); f.append(g['edge_feats']); v.append(g['edge_vectors'])
                        b.append(torch.full((m,), i, dtype=torch.long)); off += m
                    gb = {'z':z.to(device), 'idx':torch.cat(idx, 1).to(device), 'f':torch.cat(f).to(device), 'v':torch.cat(v).to(device), 'b':torch.cat(b).to(device), 'n':n}
                    vl += (mod(bc.to(device), gb)-bt.to(device)).abs().mean().item()
            pbar.set_postfix(tr=tl/len(tr_dl), vl=vl/len(vl_dl))
        
        mod.eval(); te_mae = 0
        with torch.no_grad():
            for bc, bt, bg in te_dl:
                z, idx, f, v, b, n = torch.cat([g['atom_numbers'] for g in bg]), [], [], [], [], len(bg)
                off = 0
                for i, g in enumerate(bg):
                    m = len(g['atom_numbers'])
                    idx.append(g['edge_index']+off); f.append(g['edge_feats']); v.append(g['edge_vectors'])
                    b.append(torch.full((m,), i, dtype=torch.long)); off += m
                gb = {'z':z.to(device), 'idx':torch.cat(idx, 1).to(device), 'f':torch.cat(f).to(device), 'v':torch.cat(v).to(device), 'b':torch.cat(b).to(device), 'n':n}
                te_mae += (mod(bc.to(device), gb)-bt.to(device)).abs().mean().item()
        maes.append(te_mae/len(te_dl))
        print(f"Fold {fi+1} MAE: {maes[-1]}")
    print(f"Final MAE: {np.mean(maes)}")

if __name__ == '__main__':
    run_benchmark()
