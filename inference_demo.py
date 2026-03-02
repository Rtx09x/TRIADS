"""
TRIADS Inference Demo
=====================
Demonstrates the featurization pipeline and model architecture for
steel yield strength prediction.

Usage:
    python inference_demo.py --composition "Fe0.8Cr0.1Ni0.1"

For full ensemble inference with trained weights, load the model
checkpoints from the models/ directory.
"""

import torch
import argparse
from v13a import DeepHybridTRM, ExpandedFeaturizer
from pymatgen.core import Composition


def run_inference(comp_str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Featurizer (downloads mat2vec embeddings on first run)
    feat = ExpandedFeaturizer()
    comp = [Composition(comp_str)]
    X = feat.featurize_all(comp)

    print(f"\n{'='*60}")
    print(f"  TRIADS Inference Demo")
    print(f"{'='*60}")
    print(f"\n  Composition:          {comp_str}")
    print(f"  Feature Vector Shape: {X.shape}")
    print(f"  Feature Dimensions:   {X.shape[1]} (Magpie + Mat2Vec + Matminer)")
    print(f"\n  Model Architecture:   DeepHybridTRM (V13A)")
    print(f"  Parameters:           224,685")
    print(f"  Recursion Steps:      20 (Deep Supervised)")
    print(f"  Ensemble Seeds:       5 (42, 123, 7, 0, 99)")
    print(f"\n  Note: Load ensemble weights from models/ for")
    print(f"        production inference (91.20 MPa MAE).")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRIADS: Predict steel alloy yield strength from composition"
    )
    parser.add_argument(
        "--composition",
        type=str,
        default="Fe0.7Cr0.15Ni0.15",
        help="Chemical composition string (e.g., 'Fe0.8Cr0.1Ni0.1')"
    )
    args = parser.parse_args()

    run_inference(args.composition)
