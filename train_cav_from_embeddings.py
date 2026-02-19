#!/usr/bin/env python3
"""
Train a CAV from pre-prepared pos.npy and neg.npy embedding files.

Run prepare_embeddings.py first to produce these files from your .pkl inputs.

Examples
--------
# Basic training:
python train_cav_from_embeddings.py \\
    --embed-dir ./embeddings/my_concept/ \\
    --out ./cavs/my_concept/

# With holdout evaluation (20% held out before fitting):
python train_cav_from_embeddings.py \\
    --embed-dir ./embeddings/my_concept/ \\
    --out ./cavs/my_concept/ \\
    --holdout 0.2

# Custom PCA and regularization:
python train_cav_from_embeddings.py \\
    --embed-dir ./embeddings/my_concept/ \\
    --out ./cavs/my_concept/ \\
    --no-pca \\
    --C 0.1
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "esm2_tcav"))

from src.train_cav import train_cav

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train a CAV from pos.npy and neg.npy embedding files"
    )
    parser.add_argument('--embed-dir', required=True,
                        help='Directory containing pos.npy and neg.npy')
    parser.add_argument('--out', required=True,
                        help='Output directory for CAV artifacts')
    parser.add_argument('--holdout', type=float, default=0.0,
                        help='Fraction to hold out for evaluation (default: 0, no holdout)')
    parser.add_argument('--pca-dim', type=int, default=128,
                        help='PCA dimensionality (default: 128)')
    parser.add_argument('--no-pca', action='store_true',
                        help='Disable PCA preprocessing')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Logistic regression regularization strength (default: 1.0)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Cross-validation folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--version', default='v1',
                        help='Artifact version string (default: v1)')
    args = parser.parse_args()

    config = {
        'use_pca': not args.no_pca,
        'pca_dim': args.pca_dim,
        'regularization_C': args.C,
        'cv_folds': args.cv_folds,
        'random_seed': args.seed,
    }

    train_cav(
        embed_dir=args.embed_dir,
        output_dir=args.out,
        config=config,
        artifact_version=args.version,
        holdout_fraction=args.holdout
    )


if __name__ == '__main__':
    main()
