"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRIADS V13A — SOTA Evaluation Script                               ║
║  Reproduces the 91.20 MPa MAE on matbench_steels                    ║
║                                                                      ║
║  Usage:                                                              ║
║    python evaluate.py                                                ║
║    python evaluate.py --checkpoint path/to/triads_v13a_ensemble.pt   ║
║                                                                      ║
║  This script will:                                                   ║
║    1. Download the checkpoint from HuggingFace (if not provided)     ║
║    2. Load the matbench_steels dataset (312 samples)                 ║
║    3. Compute expanded features (Magpie + Mat2Vec + Matminer)        ║
║    4. Run official 5-fold nested CV with the 5-seed ensemble         ║
║    5. Report per-fold and overall MAE                                ║
║                                                                      ║
║  Expected output: ~91.20 MPa MAE                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from model_arch import DeepHybridTRM, ExpandedFeaturizer, DSData


def evaluate(checkpoint_path=None):
    # ── 1. Load checkpoint ────────────────────────────────────────────
    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download
        print("Downloading checkpoint from HuggingFace...")
        checkpoint_path = hf_hub_download(
            repo_id="Rtx09/TRIADS",
            filename="triads_v13a_ensemble.pt"
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    seeds = ckpt["seeds"]
    n_folds = ckpt["n_folds"]
    ensemble_weights = ckpt["ensemble_weights"]
    print(f"  {ckpt['model_name']} — {len(ensemble_weights)} models "
          f"({len(seeds)} seeds × {n_folds} folds)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── 2. Load matbench_steels ───────────────────────────────────────
    print("\nLoading matbench_steels dataset...")
    from matminer.datasets import load_dataset
    from pymatgen.core import Composition

    df = load_dataset("matbench_steels")
    comps_raw = df["composition"].tolist()
    targets_all = np.array(df["yield strength"].tolist(), np.float32)
    comps_all = [Composition(c) for c in comps_raw]
    print(f"  {len(comps_all)} samples loaded")

    # ── 3. Featurize ──────────────────────────────────────────────────
    print("\nComputing expanded features (Magpie + Mat2Vec + Matminer)...")
    feat = ExpandedFeaturizer()
    X_all = feat.featurize_all(comps_all)
    print(f"  Feature shape: {X_all.shape}")

    # ── 4. Official 5-fold CV (same split as training) ────────────────
    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))

    print(f"\n{'═'*60}")
    print(f"  Running 5-Fold Evaluation with {len(seeds)}-Seed Ensemble")
    print(f"{'═'*60}\n")

    fold_maes = []

    for fi, (tv_i, te_i) in enumerate(folds):
        # Fit scaler on training data for this fold
        feat.fit_scaler(X_all[tv_i])
        te_s = feat.transform(X_all[te_i])

        te_dl = DataLoader(
            DSData(te_s, targets_all[te_i]),
            batch_size=32, shuffle=False, num_workers=0
        )
        te_tgt = torch.tensor(targets_all[te_i], dtype=torch.float32)

        # Collect predictions from all seeds for this fold
        seed_preds = []
        for seed in seeds:
            key = f"seed{seed}_fold{fi+1}"
            if key not in ensemble_weights:
                print(f"  WARNING: Missing {key}, skipping")
                continue

            # Force n_extra=200 to match pool.0.weight shape [96, 264] (64+200=264)
            model = DeepHybridTRM(n_extra=200).to(device)
            model.load_state_dict(ensemble_weights[key])
            model.eval()

            preds = []
            with torch.no_grad():
                for bx, _ in te_dl:
                    preds.append(model(bx.to(device)).cpu())
            seed_preds.append(torch.cat(preds))

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Ensemble: average across seeds
        ensemble_pred = torch.stack(seed_preds).mean(dim=0)
        fold_mae = F.l1_loss(ensemble_pred, te_tgt).item()
        fold_maes.append(fold_mae)

        print(f"  Fold {fi+1}/5: MAE = {fold_mae:.2f} MPa  "
              f"({len(seed_preds)} seeds)")

    # ── 5. Report ─────────────────────────────────────────────────────
    avg_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)

    print(f"\n{'═'*60}")
    print(f"  TRIADS V13A — Final Result")
    print(f"  {'─'*40}")
    print(f"  Per-fold MAE: {[f'{m:.2f}' for m in fold_maes]}")
    print(f"  Average MAE:  {avg_mae:.2f} ± {std_mae:.2f} MPa")
    print(f"  Best fold:    {min(fold_maes):.2f} MPa")
    print(f"{'═'*60}")

    return avg_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce TRIADS V13A SOTA result (91.20 MPa)")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to triads_v13a_ensemble.pt "
             "(downloads from HuggingFace if not provided)")
    args = parser.parse_args()
    evaluate(args.checkpoint)
