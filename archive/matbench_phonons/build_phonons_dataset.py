"""
+=============================================================+
|  TRIADS — Phonons Dataset Builder                           |
|  Builds crystal graphs + composition features from          |
|  matbench_phonons for the hybrid GNN+TRM training script.   |
|                                                             |
|  ⚠ NO SCALING / NO TRAINING / NO MODEL WEIGHTS HERE ⚠     |
|  All features are deterministic per-sample lookups.         |
|  StandardScaler is applied ONLY during training.            |
+=============================================================+

DEPENDENCIES:
    pip install matminer pymatgen gensim tqdm scikit-learn torch numpy

USAGE:
    python build_phonons_dataset.py
    -> Outputs: phonons_dataset.pt (~20 MB)
"""

import os, time, warnings, urllib.request, logging
warnings.filterwarnings('ignore')

import numpy as np
import torch
from tqdm import tqdm

from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
log = logging.getLogger("BUILD")


# ======================================================================
# GRAPH CONSTRUCTION (per-structure, NO cross-sample info)
# ======================================================================

def gaussian_expand(distances, n_bins=40, d_min=0.0, d_max=8.0):
    """Fixed Gaussian radial basis expansion. No learnable parameters."""
    centers = torch.linspace(d_min, d_max, n_bins)
    gamma = 1.0 / ((d_max - d_min) / n_bins) ** 2
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers.unsqueeze(0)) ** 2)


def build_crystal_graph(structure, max_neighbors=12, cutoff=8.0, n_gaussian=40):
    """
    Build a crystal graph for a SINGLE structure.
    Returns atom numbers, edge index, Gaussian-expanded edge features,
    and normalized edge vectors (for directional/angular info).

    ✅ ZERO DATA LEAKAGE: uses only this structure's own geometry.
    """
    n_atoms = len(structure)
    atom_numbers = torch.tensor(
        [site.specie.Z for site in structure], dtype=torch.long
    )

    src_list, dst_list, dist_list, vec_list = [], [], [], []

    try:
        all_nbrs = structure.get_all_neighbors(cutoff)
        for i, nbrs in enumerate(all_nbrs):
            # Sort by distance, keep closest max_neighbors
            nbrs_sorted = sorted(nbrs, key=lambda x: x.nn_distance)[:max_neighbors]
            for nbr in nbrs_sorted:
                src_list.append(i)
                dst_list.append(nbr.index)
                dist_list.append(nbr.nn_distance)
                vec = nbr.coords - structure[i].coords
                vec_list.append(vec)
    except Exception as e:
        log.warning(f"  Neighbor finding failed: {e}. Using empty graph.")

    if len(src_list) == 0:
        # Fallback: self-loop for isolated structure
        edge_index = torch.zeros(2, 1, dtype=torch.long)
        edge_feats = torch.zeros(1, n_gaussian, dtype=torch.float32)
        edge_vectors = torch.zeros(1, 3, dtype=torch.float32)
        return atom_numbers, edge_index, edge_feats, edge_vectors

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    distances = torch.tensor(dist_list, dtype=torch.float32)
    vectors = torch.tensor(np.array(vec_list), dtype=torch.float32)

    # Gaussian expand distances (fixed basis, no parameters)
    edge_feats = gaussian_expand(distances, n_gaussian)

    # Normalize edge vectors to unit direction
    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    edge_vectors = vectors / norms

    return atom_numbers, edge_index, edge_feats, edge_vectors


# ======================================================================
# COMPOSITION FEATURIZER (identical to phonons_v1.py)
# ======================================================================

class PhononFeaturizer:
    """
    ~361d composition + macro-structural features.
    ✅ ALL features are deterministic per-sample lookups. No cross-sample info.
    """
    GCS = "https://storage.googleapis.com/mat2vec/"
    FILES = ["pretrained_embeddings",
             "pretrained_embeddings.wv.vectors.npy",
             "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache="mat2vec_cache"):
        from matminer.featurizers.composition import (
            Stoichiometry, ValenceOrbital, IonProperty
        )
        from matminer.featurizers.composition.element import TMetalFraction
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        self.SpacegroupAnalyzer = SpacegroupAnalyzer

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

    def _extract_structural(self, structure):
        feats = []
        try:
            lat = structure.lattice
            feats.extend([lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma])
            feats.append(structure.volume / max(len(structure), 1))
            feats.append(structure.density)
            feats.append(float(len(structure)))
            try:
                sga = self.SpacegroupAnalyzer(structure, symprec=0.1)
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
            parts.append(self._extract_structural(structure))
        else:
            parts.append(np.zeros(11, np.float32))
        return np.concatenate(parts)

    def featurize_all(self, comps, structures=None):
        out = []
        test_struct = structures[0] if structures else None
        test_ex = self._featurize_extra(comps[0], test_struct)
        self.n_extra = len(test_ex)
        total = self.n_mg + self.n_extra + 200
        log.info(f"  Composition features: {self.n_mg} Magpie + "
                 f"{self.n_extra} Extra + 200 Mat2Vec = {total}d")
        for i, c in enumerate(tqdm(comps, desc="  Featurizing compositions", leave=False)):
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


# ======================================================================
# MAIN — BUILD AND SAVE
# ======================================================================

def main():
    t0 = time.time()
    print("""
  +==========================================================+
  |  TRIADS Phonons — Dataset Builder                        |
  |  Builds crystal graphs + composition features            |
  |  ⚠ NO SCALING HERE — done during training only ⚠       |
  +==========================================================+
    """)

    # ── LOAD ──────────────────────────────────────────────────────────
    print("  Loading matbench_phonons...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_phonons")
    targets = np.array(df['last phdos peak'].tolist(), np.float32)
    structures = df['structure'].tolist()
    comps = [s.composition for s in structures]
    print(f"  Loaded: {len(structures)} samples")
    print(f"  Target range: {targets.min():.1f} – {targets.max():.1f} 1/cm")

    # ── BUILD CRYSTAL GRAPHS ─────────────────────────────────────────
    print("\n  Building crystal graphs (12-NN, cutoff=8Å, 40 Gaussian bins)...")
    graphs = []
    for i, struct in enumerate(tqdm(structures, desc="  Building graphs")):
        atom_nums, edge_idx, edge_feats, edge_vecs = build_crystal_graph(
            struct, max_neighbors=12, cutoff=8.0, n_gaussian=40
        )
        graphs.append({
            'atom_numbers': atom_nums,
            'edge_index': edge_idx,
            'edge_feats': edge_feats,
            'edge_vectors': edge_vecs,
            'n_atoms': len(atom_nums),
            'n_edges': edge_idx.shape[1],
        })

    n_atoms_list = [g['n_atoms'] for g in graphs]
    n_edges_list = [g['n_edges'] for g in graphs]
    print(f"  Graphs built:")
    print(f"    Atoms/crystal:  min={min(n_atoms_list)}, max={max(n_atoms_list)}, "
          f"mean={np.mean(n_atoms_list):.1f}")
    print(f"    Edges/crystal:  min={min(n_edges_list)}, max={max(n_edges_list)}, "
          f"mean={np.mean(n_edges_list):.1f}")

    # ── COMPOSITION FEATURES ─────────────────────────────────────────
    print("\n  Computing composition features...")
    feat = PhononFeaturizer()
    comp_features = feat.featurize_all(comps, structures)
    n_extra = feat.n_extra
    print(f"  Composition features shape: {comp_features.shape}")
    print(f"  n_extra (non-Magpie, non-Mat2Vec): {n_extra}")

    # ── SAVE ─────────────────────────────────────────────────────────
    save_path = "phonons_dataset.pt"
    torch.save({
        'graphs': graphs,
        'comp_features': torch.tensor(comp_features, dtype=torch.float32),
        'targets': torch.tensor(targets, dtype=torch.float32),
        'n_samples': len(structures),
        'n_extra_comp': n_extra,  # original n_extra from composition featurizer
        'n_magpie': feat.n_mg,
        'n_gaussian': 40,
        'max_neighbors': 12,
        'cutoff': 8.0,
    }, save_path)

    size_mb = os.path.getsize(save_path) / 1e6
    dt = time.time() - t0
    print(f"\n  ✅ Saved: {save_path} ({size_mb:.1f} MB)")
    print(f"  Total time: {dt:.1f}s")
    print(f"\n  Next: Run phonons_v2.py to train the hybrid GNN+TRM model.")


if __name__ == '__main__':
    main()
