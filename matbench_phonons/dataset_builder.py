"""
+=============================================================+
|  V6 Physics-Featurized Phonon Dataset Builder                |
|  Architecture-Agnostic | Rich Physics | 3-Order Graphs       |
|                                                              |
|  Features per atom:  18d (element physics + coords + local)  |
|  Features per bond:   8d physics + 40d RBF + 3d direction    |
|  Order 2 (angles):    8d angle RBF                           |
|  Order 3 (dihedrals): 8d dihedral RBF                        |
|  Composition:         MAGPIE + mat2vec + matminer extras     |
|  Global physics:      Debye temp, force constants, etc.      |
|                                                              |
|  ⚠ NO SCALING — raw features. Scale at training time only.  |
+=============================================================+

DEPENDENCIES:
    pip install matminer pymatgen gensim tqdm scikit-learn torch numpy

USAGE:
    python build_phonons_v6_dataset.py
    -> Outputs: phonons_v6_dataset.pt
"""

import os, time, math, warnings, urllib.request, logging
from collections import defaultdict
warnings.filterwarnings('ignore')

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
log = logging.getLogger("V6-BUILD")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CUTOFF          = 8.0
MAX_NEIGHBORS   = 12
N_RBF_DIST      = 40
N_RBF_ANGLE     = 8
N_RBF_DIHEDRAL  = 8
MAX_QUADS       = 50000     # cap dihedrals per crystal for memory
FOLD_SEED       = 18012019  # matbench v0.1 protocol
N_FOLDS         = 5

N_ELEM_FEAT     = 12        # from lookup table
N_ATOM_COMPUTED = 6         # frac_coords(3) + coord_num(1) + avg_nn(1) + std_nn(1)
N_ATOM_FEAT     = N_ELEM_FEAT + N_ATOM_COMPUTED  # 18
N_BOND_PHYSICS  = 8
N_GLOBAL_PHYS   = 15


# ═══════════════════════════════════════════════════════════════
# GAUSSIAN RADIAL BASIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def gaussian_rbf(values, n_bins, vmin, vmax):
    """Fixed Gaussian expansion. No learnable parameters."""
    centers = torch.linspace(vmin, vmax, n_bins)
    gamma = 1.0 / ((vmax - vmin) / n_bins) ** 2
    return torch.exp(-gamma * (values.unsqueeze(-1) - centers.unsqueeze(0)) ** 2)


# ═══════════════════════════════════════════════════════════════
# ELEMENT PHYSICS LOOKUP TABLE
# ═══════════════════════════════════════════════════════════════

def build_element_table():
    """
    Build [103, 12] lookup table of per-element physical properties.
    Z=0 is padding. Uses pymatgen Element data.

    Columns: mass, 1/sqrt(mass), electronegativity, atomic_radius,
             covalent_radius, ionization_energy, electron_affinity,
             valence_electrons, group, period, block, is_metal
    """
    from pymatgen.core.periodic_table import Element

    block_map = {'s': 0., 'p': 1., 'd': 2., 'f': 3.}
    table = torch.zeros(103, N_ELEM_FEAT)

    for z in range(1, 103):
        try:
            el = Element.from_Z(z)
            mass = float(el.atomic_mass) if el.atomic_mass else 1.0
            chi = float(el.X) if el.X is not None else 0.0
            ar = float(el.atomic_radius) if el.atomic_radius is not None else 1.5
            # Covalent radius proxy
            try:
                cr = float(el.average_ionic_radius) if el.average_ionic_radius and float(el.average_ionic_radius) > 0 else ar
            except:
                cr = ar
            # First ionization energy
            ie = 0.0
            try:
                ies = el.ionization_energies
                if isinstance(ies, dict) and 1 in ies and ies[1] is not None:
                    ie = float(ies[1])
                elif isinstance(ies, (list, tuple)) and len(ies) > 1 and ies[1] is not None:
                    ie = float(ies[1])
            except:
                pass
            # Electron affinity
            ea = 0.0
            try:
                if el.electron_affinity is not None:
                    ea = float(el.electron_affinity)
            except:
                pass
            # Group, period, valence electrons
            g = int(el.group) if el.group is not None else 0
            p = int(el.row) if el.row is not None else 0
            ve = g if g <= 2 else (g - 10 if g >= 13 else 2)
            bl = block_map.get(el.block, 0.) if hasattr(el, 'block') and el.block else 0.
            im = 1.0 if el.is_metal else 0.0

            table[z] = torch.tensor([
                mass, 1.0 / math.sqrt(max(mass, 0.01)), chi, ar, cr,
                ie, ea, float(ve), float(g), float(p), bl, im
            ])
        except:
            table[z] = torch.tensor([1., 1., 0., 1.5, 1.5, 0., 0., 0., 0., 0., 0., 0.])

    return table


# ═══════════════════════════════════════════════════════════════
# CRYSTAL GRAPH BUILDER (Orders 1, 2, 3)
# ═══════════════════════════════════════════════════════════════

def _empty_graph(atom_z, atom_features, n_atoms):
    """Fallback for crystals with no neighbors found."""
    return {
        'atom_z': atom_z,
        'atom_features': atom_features,
        'n_atoms': n_atoms,
        'edge_index': torch.zeros(2, 1, dtype=torch.long),
        'edge_dist': torch.zeros(1),
        'edge_rbf': torch.zeros(1, N_RBF_DIST),
        'edge_vec': torch.zeros(1, 3),
        'edge_physics': torch.zeros(1, N_BOND_PHYSICS),
        'n_edges': 1,
        'triplet_index': torch.zeros(2, 0, dtype=torch.long),
        'angle_rbf': torch.zeros(0, N_RBF_ANGLE),
        'n_triplets': 0,
        'quad_index': torch.zeros(2, 0, dtype=torch.long),
        'dihedral_rbf': torch.zeros(0, N_RBF_DIHEDRAL),
        'n_quads': 0,
    }


def build_crystal_graph(structure, elem_table):
    """
    Build a complete 3-order crystal graph for a single structure.

    Returns dict with atom features, edge features + physics,
    triplets (angles), and quads (dihedrals).

    ✅ ZERO DATA LEAKAGE: uses ONLY this structure's geometry.
    """
    n_atoms = len(structure)
    atom_z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    # Element lookup features [N, 12]
    atom_elem_feat = elem_table[atom_z.clamp(0, 102)]

    # Fractional coordinates [N, 3]
    frac_coords = torch.tensor(
        [site.frac_coords for site in structure], dtype=torch.float32
    )

    # ── NEIGHBOR FINDING ──────────────────────────────────────
    src_list, dst_list, dist_list, vec_list = [], [], [], []
    nn_dists_per_atom = defaultdict(list)

    try:
        all_nbrs = structure.get_all_neighbors(CUTOFF)
        for i, nbrs in enumerate(all_nbrs):
            nbrs_sorted = sorted(nbrs, key=lambda x: x.nn_distance)[:MAX_NEIGHBORS]
            for nbr in nbrs_sorted:
                src_list.append(i)
                dst_list.append(nbr.index)
                dist_list.append(nbr.nn_distance)
                vec_list.append(nbr.coords - structure[i].coords)
                nn_dists_per_atom[i].append(nbr.nn_distance)
    except Exception as e:
        log.warning(f"  Neighbor finding failed: {e}")

    # Per-atom coordination stats
    coord_nums = torch.zeros(n_atoms)
    avg_nn_dists = torch.zeros(n_atoms)
    std_nn_dists = torch.zeros(n_atoms)
    for i in range(n_atoms):
        ds = nn_dists_per_atom.get(i, [])
        coord_nums[i] = len(ds)
        if ds:
            avg_nn_dists[i] = np.mean(ds)
            std_nn_dists[i] = np.std(ds) if len(ds) > 1 else 0.0

    # Combined atom features [N, 18]
    atom_features = torch.cat([
        atom_elem_feat,                                          # [N, 12]
        frac_coords,                                             # [N, 3]
        coord_nums.unsqueeze(-1),                                # [N, 1]
        avg_nn_dists.unsqueeze(-1),                              # [N, 1]
        std_nn_dists.unsqueeze(-1),                              # [N, 1]
    ], dim=-1)  # [N, 18]

    if len(src_list) == 0:
        return _empty_graph(atom_z, atom_features, n_atoms)

    # ── EDGE FEATURES (Order 1) ───────────────────────────────
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_dist = torch.tensor(dist_list, dtype=torch.float32)
    raw_vecs = torch.tensor(np.array(vec_list), dtype=torch.float32)
    n_edges = edge_index.shape[1]

    edge_rbf = gaussian_rbf(edge_dist, N_RBF_DIST, 0.0, CUTOFF)
    norms = raw_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    edge_vec = raw_vecs / norms

    # ── BOND PHYSICS FEATURES [E, 8] ─────────────────────────
    z_src = atom_z[edge_index[0]]  # [E]
    z_dst = atom_z[edge_index[1]]  # [E]

    m_src = elem_table[z_src.clamp(0, 102), 0]   # mass
    m_dst = elem_table[z_dst.clamp(0, 102), 0]
    chi_src = elem_table[z_src.clamp(0, 102), 2]  # electronegativity
    chi_dst = elem_table[z_dst.clamp(0, 102), 2]
    r_src = elem_table[z_src.clamp(0, 102), 3]    # atomic radius
    r_dst = elem_table[z_dst.clamp(0, 102), 3]

    d = edge_dist.clamp(min=0.01)

    # Vectorized bond physics computation
    chi_prod = (chi_src * chi_dst).clamp(min=0.01)
    k_est = torch.sqrt(chi_prod) / (d * d)                       # force constant
    mu = (m_src * m_dst) / (m_src + m_dst).clamp(min=0.01)       # reduced mass
    omega = torch.sqrt(k_est / mu.clamp(min=0.01))               # Einstein freq
    delta_chi = (chi_src - chi_dst).abs()                         # EN difference
    ionicity = delta_chi * delta_chi                              # bond ionicity
    r_ratio = (r_src + r_dst) / d                                 # radius sum ratio
    m_ratio = torch.min(m_src, m_dst) / torch.max(m_src, m_dst).clamp(min=0.01)
    inv_d = 1.0 / d                                               # inverse distance

    edge_physics = torch.stack([
        k_est, mu, omega, delta_chi, ionicity, r_ratio, m_ratio, inv_d
    ], dim=-1)  # [E, 8]

    # ── TRIPLETS / ANGLES (Order 2) ───────────────────────────
    dst_np = edge_index[1].numpy()
    dest_to_edges = defaultdict(list)
    for e_idx in range(n_edges):
        dest_to_edges[int(dst_np[e_idx])].append(e_idx)

    trip_ij, trip_kj = [], []
    for j, edge_list in dest_to_edges.items():
        for idx_ij in edge_list:
            for idx_kj in edge_list:
                if idx_ij != idx_kj:
                    trip_ij.append(idx_ij)
                    trip_kj.append(idx_kj)

    if trip_ij:
        triplet_index = torch.tensor([trip_ij, trip_kj], dtype=torch.long)
        v_ij = edge_vec[triplet_index[0]]
        v_kj = edge_vec[triplet_index[1]]
        cos_theta = (v_ij * v_kj).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
        angles = torch.acos(cos_theta)
        angle_rbf_t = gaussian_rbf(angles, N_RBF_ANGLE, 0.0, math.pi)
        n_triplets = triplet_index.shape[1]
    else:
        triplet_index = torch.zeros(2, 0, dtype=torch.long)
        angle_rbf_t = torch.zeros(0, N_RBF_ANGLE)
        n_triplets = 0

    # ── QUADS / DIHEDRALS (Order 3) ───────────────────────────
    quad_index, dihedral_rbf_t, n_quads = _compute_quads(
        triplet_index, n_triplets, edge_vec, trip_ij, trip_kj
    )

    return {
        'atom_z': atom_z,
        'atom_features': atom_features,
        'n_atoms': n_atoms,
        'edge_index': edge_index,
        'edge_dist': edge_dist,
        'edge_rbf': edge_rbf,
        'edge_vec': edge_vec,
        'edge_physics': edge_physics,
        'n_edges': n_edges,
        'triplet_index': triplet_index,
        'angle_rbf': angle_rbf_t,
        'n_triplets': n_triplets,
        'quad_index': quad_index,
        'dihedral_rbf': dihedral_rbf_t,
        'n_quads': n_quads,
    }


def _compute_quads(triplet_index, n_triplets, edge_vec, trip_ij, trip_kj):
    """Compute Order 3: pairs of triplets sharing a bond (dihedrals)."""
    if n_triplets == 0:
        return (torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, N_RBF_DIHEDRAL), 0)

    # For each edge, which triplets reference it?
    edge_to_trips = defaultdict(list)
    for t_idx in range(n_triplets):
        edge_to_trips[trip_ij[t_idx]].append(t_idx)
        edge_to_trips[trip_kj[t_idx]].append(t_idx)

    quad_src, quad_dst = [], []
    for edge_idx, tlist in edge_to_trips.items():
        for i in range(len(tlist)):
            for j in range(len(tlist)):
                if tlist[i] != tlist[j]:
                    quad_src.append(tlist[i])
                    quad_dst.append(tlist[j])
                    if len(quad_src) >= MAX_QUADS:
                        break
            if len(quad_src) >= MAX_QUADS:
                break
        if len(quad_src) >= MAX_QUADS:
            break

    if not quad_src:
        return (torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, N_RBF_DIHEDRAL), 0)

    quad_index = torch.tensor([quad_src, quad_dst], dtype=torch.long)

    # Dihedral angle = angle between planes of the two triplets
    v_a1 = edge_vec[triplet_index[0, quad_index[0]]]
    v_a2 = edge_vec[triplet_index[1, quad_index[0]]]
    v_b1 = edge_vec[triplet_index[0, quad_index[1]]]
    v_b2 = edge_vec[triplet_index[1, quad_index[1]]]

    n_a = torch.cross(v_a1, v_a2, dim=-1)
    n_b = torch.cross(v_b1, v_b2, dim=-1)
    n_a = n_a / n_a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    n_b = n_b / n_b.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cos_dih = (n_a * n_b).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    dihedrals = torch.acos(cos_dih)
    dihedral_rbf_t = gaussian_rbf(dihedrals, N_RBF_DIHEDRAL, 0.0, math.pi)

    return quad_index, dihedral_rbf_t, quad_index.shape[1]


# ═══════════════════════════════════════════════════════════════
# GLOBAL PHYSICS FEATURES (per crystal)
# ═══════════════════════════════════════════════════════════════

def compute_global_physics(graph, structure, elem_table):
    """
    Compute 15 global physics features from a crystal graph.

    Features:
      0: avg_force_constant       7: avg_coordination
      1: std_force_constant       8: density
      2: avg_reduced_mass         9: volume_per_atom
      3: mass_variance           10: packing_fraction
      4: avg_einstein_freq       11: avg_bond_length
      5: electronegativity_var   12: std_bond_length
      6: debye_temp_estimate     13: max_atomic_mass
                                 14: min_atomic_mass
    """
    ep = graph['edge_physics']  # [E, 8]
    n_atoms = graph['n_atoms']
    atom_z = graph['atom_z']

    # From bond physics
    k_vals = ep[:, 0]   # force constants
    mu_vals = ep[:, 1]   # reduced masses
    omega_vals = ep[:, 2]  # Einstein frequencies
    dists = graph['edge_dist']

    feats = torch.zeros(N_GLOBAL_PHYS)

    if graph['n_edges'] > 0 and dists.shape[0] > 0:
        feats[0] = k_vals.mean()
        feats[1] = k_vals.std() if k_vals.shape[0] > 1 else 0.0
        feats[2] = mu_vals.mean()
        feats[4] = omega_vals.mean()
        feats[11] = dists.mean()
        feats[12] = dists.std() if dists.shape[0] > 1 else 0.0

    # Mass statistics
    masses = elem_table[atom_z.clamp(0, 102), 0]
    feats[3] = masses.var() if n_atoms > 1 else 0.0
    feats[13] = masses.max()
    feats[14] = masses.min()

    # Electronegativity variance
    chis = elem_table[atom_z.clamp(0, 102), 2]
    feats[5] = chis.var() if n_atoms > 1 else 0.0

    # Debye temperature estimate: Θ_D ∝ sqrt(k_avg / m_avg)
    m_avg = masses.mean()
    k_avg = feats[0]
    feats[6] = math.sqrt(float(k_avg / max(m_avg, 0.01)))

    # Coordination
    feats[7] = graph['atom_features'][:, N_ELEM_FEAT + 3].mean()  # coord_num column

    # Structural
    try:
        feats[8] = structure.density
        feats[9] = structure.volume / max(n_atoms, 1)
        # Packing fraction
        total_vol = sum(
            (4 / 3) * math.pi * (float(site.specie.atomic_radius) ** 3)
            for site in structure
            if hasattr(site.specie, 'atomic_radius') and site.specie.atomic_radius is not None
        )
        feats[10] = total_vol / structure.volume if structure.volume > 0 else 0.0
    except:
        pass

    return feats


# ═══════════════════════════════════════════════════════════════
# STRUCTURAL FEATURES (per crystal)
# ═══════════════════════════════════════════════════════════════

def compute_structural_features(structure):
    """
    Compute 11 structural features: lattice params + symmetry.
    Same as previous versions for backward compatibility.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    feats = np.zeros(11, dtype=np.float32)
    try:
        lat = structure.lattice
        feats[0:6] = [lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma]
        feats[6] = structure.volume / max(len(structure), 1)
        feats[7] = structure.density
        feats[8] = float(len(structure))
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            feats[9] = float(sga.get_space_group_number())
        except:
            feats[9] = 0.0
        try:
            total_vol = sum(
                (4 / 3) * np.pi * site.specie.atomic_radius ** 3
                for site in structure
                if hasattr(site.specie, 'atomic_radius') and site.specie.atomic_radius is not None
            )
            feats[10] = total_vol / structure.volume if structure.volume > 0 else 0.0
        except:
            feats[10] = 0.0
    except:
        pass
    return feats


# ═══════════════════════════════════════════════════════════════
# COMPOSITION FEATURIZER (MAGPIE + mat2vec + matminer extras)
# ═══════════════════════════════════════════════════════════════

class CompositionFeaturizer:
    """
    Builds rich composition features per crystal:
    - MAGPIE elemental properties (132d: 22 props × 6 stats)
    - Extra matminer (Stoichiometry, ValenceOrbital, IonProperty, TMetalFraction)
    - Structural features (11d)
    - mat2vec embeddings (200d)

    ✅ ALL features are deterministic per-sample. No cross-sample info.
    """
    M2V_URL = "https://storage.googleapis.com/mat2vec/"
    M2V_FILES = [
        "pretrained_embeddings",
        "pretrained_embeddings.wv.vectors.npy",
        "pretrained_embeddings.trainables.syn1neg.npy",
    ]

    def __init__(self, cache="mat2vec_cache"):
        from matminer.featurizers.composition import (
            ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
        )
        from matminer.featurizers.composition.element import TMetalFraction
        from gensim.models import Word2Vec

        self.ep_magpie = ElementProperty.from_preset("magpie")
        self.n_magpie = len(self.ep_magpie.feature_labels())

        self.extra_ftzrs = [
            ("Stoichiometry", Stoichiometry()),
            ("ValenceOrbital", ValenceOrbital()),
            ("IonProperty", IonProperty()),
            ("TMetalFraction", TMetalFraction()),
        ]
        self._extra_sizes = {}
        for name, ft in self.extra_ftzrs:
            try:
                self._extra_sizes[name] = len(ft.feature_labels())
            except:
                self._extra_sizes[name] = None

        # Download mat2vec
        os.makedirs(cache, exist_ok=True)
        for f in self.M2V_FILES:
            p = os.path.join(cache, f)
            if not os.path.exists(p):
                log.info(f"  Downloading mat2vec: {f}...")
                urllib.request.urlretrieve(self.M2V_URL + f, p)
        m2v = Word2Vec.load(os.path.join(cache, "pretrained_embeddings"))
        self.emb = {w: m2v.wv[w] for w in m2v.wv.index_to_key}

        self.n_extra = None  # determined on first call

    def _pool_m2v(self, comp):
        v, t = np.zeros(200, np.float32), 0.0
        for s, f in comp.get_el_amt_dict().items():
            if s in self.emb:
                v += f * self.emb[s]
                t += f
        return v / max(t, 1e-8)

    def _featurize_extras(self, comp):
        parts = []
        for name, ft in self.extra_ftzrs:
            try:
                vals = np.array(ft.featurize(comp), np.float32)
                parts.append(np.nan_to_num(vals, nan=0.0))
                if self._extra_sizes.get(name) is None:
                    self._extra_sizes[name] = len(vals)
            except:
                sz = self._extra_sizes.get(name, 0) or 1
                parts.append(np.zeros(sz, np.float32))
        return np.concatenate(parts)

    def featurize_all(self, compositions, structures):
        """Return [N, D_comp] array of all composition features."""
        # Determine dimensions from first sample
        test_extras = self._featurize_extras(compositions[0])
        self.n_extra = len(test_extras)
        struct_feats_dim = 11
        total_dim = self.n_magpie + self.n_extra + struct_feats_dim + 200

        log.info(f"  Composition features: {self.n_magpie} MAGPIE + "
                 f"{self.n_extra} Extras + 11 Structural + 200 mat2vec = {total_dim}d")

        out = []
        for i, comp in enumerate(tqdm(compositions, desc="  Featurizing compositions", leave=False)):
            # MAGPIE
            try:
                mg = np.array(self.ep_magpie.featurize(comp), np.float32)
            except:
                mg = np.zeros(self.n_magpie, np.float32)
            mg = np.nan_to_num(mg, nan=0.0)

            # Extra matminer
            ex = self._featurize_extras(comp)

            # Structural
            sf = compute_structural_features(structures[i])

            # mat2vec
            m2v = self._pool_m2v(comp)

            out.append(np.concatenate([mg, ex, sf, m2v]))

        return np.array(out, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# MAIN — BUILD AND SAVE
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("""
  +==========================================================+
  |  V6 Physics-Featurized Phonon Dataset Builder             |
  |  3-Order Graphs | Bond Physics | Architecture-Agnostic    |
  |  ⚠ NO SCALING — raw features only                       |
  +==========================================================+
    """)

    # ── LOAD MATBENCH DATA ────────────────────────────────────
    print("  Loading matbench_phonons...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_phonons")
    targets = np.array(df['last phdos peak'].tolist(), np.float32)
    structures = df['structure'].tolist()
    compositions = [s.composition for s in structures]
    N = len(structures)
    print(f"  Loaded: {N} samples")
    print(f"  Target range: {targets.min():.1f} – {targets.max():.1f} cm⁻¹")

    # ── BUILD ELEMENT TABLE ───────────────────────────────────
    print("\n  Building element physics table...")
    elem_table = build_element_table()
    print(f"  Element table: {elem_table.shape} (Z=0..102, {N_ELEM_FEAT} features)")

    # ── BUILD CRYSTAL GRAPHS ─────────────────────────────────
    print(f"\n  Building 3-order crystal graphs ({MAX_NEIGHBORS}-NN, cutoff={CUTOFF}Å)...")
    graphs = []
    global_physics_list = []

    for i, struct in enumerate(tqdm(structures, desc="  Building graphs")):
        g = build_crystal_graph(struct, elem_table)
        gp = compute_global_physics(g, struct, elem_table)
        graphs.append(g)
        global_physics_list.append(gp)

    # Stats
    n_atoms_list = [g['n_atoms'] for g in graphs]
    n_edges_list = [g['n_edges'] for g in graphs]
    n_trips_list = [g['n_triplets'] for g in graphs]
    n_quads_list = [g['n_quads'] for g in graphs]
    print(f"  Graphs built:")
    print(f"    Atoms/crystal:    min={min(n_atoms_list)}, max={max(n_atoms_list)}, "
          f"mean={np.mean(n_atoms_list):.1f}")
    print(f"    Edges/crystal:    min={min(n_edges_list)}, max={max(n_edges_list)}, "
          f"mean={np.mean(n_edges_list):.1f}")
    print(f"    Triplets/crystal: min={min(n_trips_list)}, max={max(n_trips_list)}, "
          f"mean={np.mean(n_trips_list):.1f}")
    print(f"    Quads/crystal:    min={min(n_quads_list)}, max={max(n_quads_list)}, "
          f"mean={np.mean(n_quads_list):.1f}")

    global_physics = torch.stack(global_physics_list)
    print(f"  Global physics: {global_physics.shape}")

    # ── COMPOSITION FEATURES ─────────────────────────────────
    print("\n  Computing composition features...")
    feat = CompositionFeaturizer()
    comp_features = feat.featurize_all(compositions, structures)
    print(f"  Composition features shape: {comp_features.shape}")

    # ── FOLD INDICES (strict matbench protocol) ──────────────
    print(f"\n  Computing 5-fold split indices (seed={FOLD_SEED})...")
    kf = KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    fold_indices = [(train_idx.tolist(), test_idx.tolist())
                    for train_idx, test_idx in kf.split(range(N))]

    # Verify zero leakage
    for fi, (tr, te) in enumerate(fold_indices):
        overlap = set(tr) & set(te)
        assert len(overlap) == 0, f"DATA LEAK in fold {fi}: {len(overlap)} shared indices!"
        assert len(tr) + len(te) == N, f"Fold {fi}: missing samples!"
    print("  ✅ All folds verified: ZERO data leakage")

    # ── FEATURE DIMENSION INFO ───────────────────────────────
    n_magpie = feat.n_magpie
    n_extra = feat.n_extra
    feature_info = {
        'atom_features_dim': N_ATOM_FEAT,
        'atom_features_layout': [
            'mass', '1/sqrt_mass', 'electronegativity', 'atomic_radius',
            'covalent_radius', 'ionization_energy', 'electron_affinity',
            'valence_electrons', 'group', 'period', 'block', 'is_metal',
            'frac_x', 'frac_y', 'frac_z',
            'coordination_num', 'avg_nn_dist', 'std_nn_dist',
        ],
        'edge_physics_dim': N_BOND_PHYSICS,
        'edge_physics_layout': [
            'force_constant', 'reduced_mass', 'einstein_freq',
            'en_difference', 'ionicity', 'radius_sum_ratio',
            'mass_ratio', 'inverse_distance',
        ],
        'edge_rbf_dim': N_RBF_DIST,
        'angle_rbf_dim': N_RBF_ANGLE,
        'dihedral_rbf_dim': N_RBF_DIHEDRAL,
        'global_physics_dim': N_GLOBAL_PHYS,
        'global_physics_layout': [
            'avg_force_constant', 'std_force_constant', 'avg_reduced_mass',
            'mass_variance', 'avg_einstein_freq', 'en_variance',
            'debye_temp_estimate', 'avg_coordination', 'density',
            'volume_per_atom', 'packing_fraction', 'avg_bond_length',
            'std_bond_length', 'max_atomic_mass', 'min_atomic_mass',
        ],
        'comp_magpie_range': (0, n_magpie),
        'comp_extras_range': (n_magpie, n_magpie + n_extra),
        'comp_structural_range': (n_magpie + n_extra, n_magpie + n_extra + 11),
        'comp_mat2vec_range': (n_magpie + n_extra + 11, n_magpie + n_extra + 11 + 200),
        'comp_total_dim': comp_features.shape[1],
    }

    # ── SAVE ─────────────────────────────────────────────────
    save_path = "phonons_v6_dataset.pt"
    save_data = {
        # Per-crystal data
        'graphs': graphs,
        'comp_features': torch.tensor(comp_features, dtype=torch.float32),
        'global_physics': global_physics,
        'targets': torch.tensor(targets, dtype=torch.float32),

        # Fold indices
        'fold_indices': fold_indices,
        'fold_seed': FOLD_SEED,

        # Metadata
        'n_samples': N,
        'feature_info': feature_info,
        'element_table': elem_table,
        'config': {
            'cutoff': CUTOFF,
            'max_neighbors': MAX_NEIGHBORS,
            'n_rbf_dist': N_RBF_DIST,
            'n_rbf_angle': N_RBF_ANGLE,
            'n_rbf_dihedral': N_RBF_DIHEDRAL,
            'max_quads': MAX_QUADS,
            'fold_seed': FOLD_SEED,
            'n_folds': N_FOLDS,
        },
    }
    torch.save(save_data, save_path)

    size_mb = os.path.getsize(save_path) / 1e6
    dt = time.time() - t0
    print(f"\n  ✅ Saved: {save_path} ({size_mb:.1f} MB)")
    print(f"  Total time: {dt:.1f}s")

    # ── SUMMARY ──────────────────────────────────────────────
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  Dataset Summary                                        ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Samples:            {N:>6}                              ║
  ║  Atom features:      {N_ATOM_FEAT:>6}d (12 elem + 3 coord + 3 local)  ║
  ║  Bond RBF:           {N_RBF_DIST:>6}d                              ║
  ║  Bond physics:       {N_BOND_PHYSICS:>6}d (k, μ, ω, Δχ, ...)          ║
  ║  Angle RBF:          {N_RBF_ANGLE:>6}d                              ║
  ║  Dihedral RBF:       {N_RBF_DIHEDRAL:>6}d                              ║
  ║  Composition:        {comp_features.shape[1]:>6}d (MAGPIE+extras+struct+m2v)║
  ║  Global physics:     {N_GLOBAL_PHYS:>6}d                              ║
  ║  Folds:              {N_FOLDS:>6} (seed={FOLD_SEED})         ║
  ║  File size:          {size_mb:>5.1f} MB                            ║
  ╚══════════════════════════════════════════════════════════╝

  ⚠ Remember: NO scaling applied. Apply StandardScaler at
    training time using ONLY train-fold indices!

  Architecture-agnostic: plug ANY model on top of this dataset.
    """)


if __name__ == '__main__':
    main()
