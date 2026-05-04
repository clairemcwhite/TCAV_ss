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


def parse_fasta_lengths(fasta_path: str) -> dict:
    """Return {accession: length} parsed from a FASTA file."""
    lengths = {}
    current_id = None
    current_len = 0
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    lengths[current_id] = current_len
                current_id = line[1:].split()[0]
                current_len = 0
            else:
                current_len += len(line)
    if current_id is not None:
        lengths[current_id] = current_len
    return lengths


def length_correct_scores(
    raw_scores: np.ndarray,
    protein_ids: list,
    protein_lengths: dict,
    K: int,
    ref_mean: np.ndarray,
    cav_vec: np.ndarray,
    scaler,
    pca,
) -> np.ndarray:
    """
    Correct CAV scores for motif dilution in whole-sequence embeddings.

    Models full_seq ≈ (K/L)*span + ((L-K)/L)*μ, rearranges to estimate the
    span embedding, and scores that instead. In the preprocessed CAV space:

        corrected_score = (L/K) * raw_score - ((L-K)/K) * μ_proj

    where μ_proj is the reference mean projected onto the CAV direction.
    """
    mu = ref_mean.reshape(1, -1).astype('float32')
    if scaler is not None:
        mu = scaler.transform(mu)
    if pca is not None:
        mu = pca.transform(mu)
    mu_proj = (mu @ cav_vec).item()

    corrected = np.empty_like(raw_scores)
    missing = []
    for i, pid in enumerate(protein_ids):
        L = protein_lengths.get(pid)
        if L is None:
            missing.append(pid)
            corrected[i] = raw_scores[i]
            continue
        corrected[i] = (L / K) * raw_scores[i] - ((L - K) / K) * mu_proj

    if missing:
        logger.warning(
            f"{len(missing)} proteins not found in FASTA (lengths unknown); "
            "raw scores used for those entries."
        )
    return corrected


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
    parser.add_argument('--length-correct', action='store_true',
                        help='Correct scores for motif dilution in whole-sequence embeddings. '
                             'Requires --span-length, --fasta, and --ref-pkl.')
    parser.add_argument('--span-length', type=int, default=None,
                        help='Length K of the motif span used for CAV training (number of residues).')
    parser.add_argument('--fasta', default=None,
                        help='FASTA file for the search proteome, used to look up protein lengths.')
    parser.add_argument('--ref-pkl', default=None,
                        help='Reference negative pkl used to compute the background mean embedding μ.')
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
    # Length correction (optional)
    # ------------------------------------------------------------------ #
    if args.length_correct:
        missing = [a for a in ('span_length', 'fasta', 'ref_pkl')
                   if getattr(args, a) is None]
        if missing:
            raise ValueError(
                f"--length-correct requires: {', '.join('--' + a.replace('_','-') for a in missing)}"
            )
        logger.info("Applying length correction to scores")
        protein_lengths = parse_fasta_lengths(args.fasta)
        logger.info(f"  Loaded lengths for {len(protein_lengths):,} proteins from {args.fasta}")
        ref_embs, _ = load_sequence_embeddings(args.ref_pkl)
        ref_mean = ref_embs.mean(axis=0)
        logger.info(f"  Reference mean computed from {len(ref_embs):,} sequences ({args.ref_pkl})")
        scores = length_correct_scores(
            scores, protein_ids, protein_lengths,
            K=args.span_length,
            ref_mean=ref_mean,
            cav_vec=cav_vec,
            scaler=scaler,
            pca=pca,
        )
        logger.info(f"  Corrected score range: min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")

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
