#!/usr/bin/env python3
"""
Combine two CAV score TSVs with a sigmoid gate.

Computes:
    final_score = specific_score * sigmoid(beta * (filter_score - threshold))

This upweights proteins that score high on both the specific CAV and the
filter CAV, and suppresses proteins that are weak on the filter (e.g.
non-bHLH hits in a MAD1-specific search).

Usage
-----
python specific_scripts/combine_cav_scores.py \\
    --specific  results/MAD1_specific_hits.tsv \\
    --filter    results/HLH_hits.tsv \\
    --out       results/MAD1_bHLH_hits.tsv

# Tune the gate:
python specific_scripts/combine_cav_scores.py \\
    --specific   results/MAD1_specific_hits.tsv \\
    --filter     results/HLH_hits.tsv \\
    --out        results/MAD1_bHLH_hits.tsv \\
    --threshold  0.5   # filter_score below this is suppressed
    --beta       10    # steepness of the sigmoid gate (higher = harder cutoff)
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(
        description="Gate a specific CAV score by a filter CAV score via sigmoid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--specific', required=True,
                        help='TSV of specific CAV scores (e.g. MAD1_specific_hits.tsv).')
    parser.add_argument('--filter', required=True,
                        help='TSV of filter CAV scores (e.g. HLH_hits.tsv).')
    parser.add_argument('--out', required=True,
                        help='Output TSV path.')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='filter_score value at which the sigmoid gate = 0.5 '
                             '(default: 0.0, i.e. the mean of the score distribution).')
    parser.add_argument('--beta', type=float, default=5.0,
                        help='Sigmoid steepness: higher = harder cutoff (default: 5.0).')
    parser.add_argument('--k', type=int, default=100,
                        help='Top-k results to output (default: 100; -1 for all).')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #
    specific = pd.read_csv(args.specific, sep='\t')
    filter_  = pd.read_csv(args.filter,   sep='\t')

    for df, name in [(specific, '--specific'), (filter_, '--filter')]:
        if 'protein_id' not in df.columns or 'cav_score' not in df.columns:
            raise ValueError(f"{name} TSV must have 'protein_id' and 'cav_score' columns.")

    # ------------------------------------------------------------------ #
    # Merge on protein_id
    # ------------------------------------------------------------------ #
    merged = specific[['protein_id', 'cav_score']].merge(
        filter_[['protein_id', 'cav_score']],
        on='protein_id',
        suffixes=('_specific', '_filter'),
        how='inner',
    )
    n_dropped = len(specific) - len(merged)
    if n_dropped:
        logger.warning(f"{n_dropped} proteins in --specific not found in --filter and dropped.")
    logger.info(f"  {len(merged):,} proteins after merge")

    # ------------------------------------------------------------------ #
    # Compute final score
    # ------------------------------------------------------------------ #
    gate = sigmoid(args.beta * (merged['cav_score_filter'] - args.threshold))
    merged['gate'] = gate
    merged['final_score'] = merged['cav_score_specific'] * gate

    logger.info(f"  threshold={args.threshold}, beta={args.beta}")
    logger.info(f"  gate range: {gate.min():.3f} – {gate.max():.3f}")

    # ------------------------------------------------------------------ #
    # Rank and select top-k
    # ------------------------------------------------------------------ #
    merged = merged.sort_values('final_score', ascending=False).reset_index(drop=True)
    merged.insert(0, 'rank', np.arange(1, len(merged) + 1))

    if args.k > 0:
        merged = merged.head(args.k)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep='\t', index=False, float_format='%.4f')
    logger.info(f"Saved {len(merged):,} results to {out_path}")

    print(f"\n--- Top {min(10, len(merged))} proteins ---")
    print(merged.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
