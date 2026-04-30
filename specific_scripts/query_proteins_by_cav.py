#!/usr/bin/env python3
"""
Query a protein embedding pkl for the top-k proteins most aligned with a CAV.

Scores every protein by its projection (dot product) onto the CAV direction
and returns the top-k hits as a ranked TSV.

--cav can be either:
  - A CAV artifact directory (produced by train_cav_from_embeddings.py), which
    contains the concept .npy + scaler + PCA. Preprocessing is applied before
    projection, matching how the CAV was trained.
  - A bare .npy file. The vector is used directly with no preprocessing — only
    do this if the embeddings and CAV are already in the same space.

Usage
-----
# From a CAV artifact directory (recommended):
python specific_scripts/query_proteins_by_cav.py \\
    --cav  cavs/my_concept/ \\
    --pkl  embeddings/proteins.pkl \\
    --out  results/top_proteins.tsv

# From a bare .npy:
python specific_scripts/query_proteins_by_cav.py \\
    --cav  cavs/my_concept/concept_v1.npy \\
    --pkl  embeddings/proteins.pkl \\
    --out  results/top_proteins.tsv
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tcav"))

from src.utils.data_loader import load_sequence_embeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cav(cav_path: str, version: str = 'v1'):
    """
    Load CAV vector and optional preprocessing artifacts.

    Returns (cav_vec, scaler, pca). scaler and pca are None when given a bare .npy.
    """
    p = Path(cav_path)

    if p.is_dir():
        from src.evaluate import load_cav_artifacts
        artifacts = load_cav_artifacts(str(p), version=version)
        return artifacts['concept_cav'], artifacts['scaler'], artifacts['pca']

    # Bare .npy
    cav_vec = np.load(p)
    logger.warning(
        "Loading CAV from a bare .npy — no scaler/PCA will be applied. "
        "Make sure the embeddings and CAV are already in the same space."
    )
    return cav_vec, None, None


def score_and_rank(embs: np.ndarray, cav_vec: np.ndarray, scaler, pca) -> np.ndarray:
    """Apply preprocessing and return per-protein projection scores."""
    X = embs.astype('float32')
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return X @ cav_vec


def main():
    parser = argparse.ArgumentParser(
        description="Rank proteins by projection onto a CAV direction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--cav', required=True,
                        help='CAV artifact directory or bare concept .npy file.')
    parser.add_argument('--pkl', required=True,
                        help='Protein embeddings pkl (sequence_embeddings key).')
    parser.add_argument('--out', required=True,
                        help='Output TSV path.')
    parser.add_argument('--k', type=int, default=100,
                        help='Number of top proteins to return (default: 100; '
                             'use -1 for all proteins ranked).')
    parser.add_argument('--version', default='v1',
                        help='CAV artifact version suffix (default: v1).')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load CAV
    # ------------------------------------------------------------------ #
    logger.info(f"Loading CAV from {args.cav}")
    cav_vec, scaler, pca = load_cav(args.cav, version=args.version)
    logger.info(f"  CAV dim: {cav_vec.shape[0]}")

    # ------------------------------------------------------------------ #
    # Load embeddings
    # ------------------------------------------------------------------ #
    logger.info(f"Loading embeddings from {args.pkl}")
    embs, protein_ids = load_sequence_embeddings(args.pkl)
    logger.info(f"  {len(protein_ids):,} proteins, hidden_dim={embs.shape[1]}")

    # ------------------------------------------------------------------ #
    # Score
    # ------------------------------------------------------------------ #
    scores = score_and_rank(embs, cav_vec, scaler, pca)
    logger.info(f"  Score range: min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")

    # ------------------------------------------------------------------ #
    # Rank and select top-k
    # ------------------------------------------------------------------ #
    order = np.argsort(scores)[::-1]
    if args.k > 0:
        order = order[:args.k]

    df = pd.DataFrame({
        'rank': np.arange(1, len(order) + 1),
        'protein_id': [protein_ids[i] for i in order],
        'cav_score': scores[order].astype(float),
    })

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False, float_format='%.4f')
    logger.info(f"Saved {len(df):,} results to {out_path}")

    print(f"\n--- Top {min(10, len(df))} proteins ---")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
