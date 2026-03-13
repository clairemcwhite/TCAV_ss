#!/usr/bin/env python3
"""
Summarize CAV scores by metadata groups.

Joins a CAV scores TSV (from score_samples_against_cavs.py) with
cell-level metadata from an h5ad, then prints mean CAV scores broken
down by any combination of metadata columns.

Usage
-----
python specific_scripts/summarize_cav_scores.py \
    --scores  results/fallopian_10k_cav_scores.tsv \
    --h5ad    bd314446-2d21-4675-ab1a-320391cbfe52.h5ad \
    --group   cell_type menstrual_phase_at_collection \
    --cavs    menstrual proliferative secretory \
    --out     results/fallopian_10k_summary.tsv
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize CAV scores broken down by metadata groups.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--scores', required=True,
                        help='CAV scores TSV from score_samples_against_cavs.py.')
    parser.add_argument('--h5ad', required=True,
                        help='h5ad file whose obs contains the metadata columns.')
    parser.add_argument('--obs-cols', nargs='+',
                        default=['cell_type', 'menstrual_phase_at_collection',
                                 'donor_id', 'donor_menopausal_status',
                                 'author_cell_type'],
                        help='obs columns to pull from h5ad and join to scores.')
    parser.add_argument('--group', nargs='+',
                        default=['cell_type', 'menstrual_phase_at_collection'],
                        help='Columns to group by when computing mean scores.')
    parser.add_argument('--cavs', nargs='+',
                        default=['menstrual', 'proliferative', 'secretory'],
                        help='CAV score column names to summarize.')
    parser.add_argument('--min-cells', type=int, default=5,
                        help='Skip group combinations with fewer than this '
                             'many cells (default: 5).')
    parser.add_argument('--out', default=None,
                        help='Save summary TSV to this path (optional).')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load scores
    # ------------------------------------------------------------------ #
    scores = pd.read_csv(args.scores, sep='\t')
    print(f"Scores loaded: {len(scores)} cells")

    cav_cols = [c for c in args.cavs if c in scores.columns]
    if not cav_cols:
        print(f"ERROR: none of {args.cavs} found in scores columns: "
              f"{scores.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Load obs metadata from h5ad
    # ------------------------------------------------------------------ #
    print(f"Loading obs from {args.h5ad} ...")
    import anndata as ad
    adata = ad.read_h5ad(args.h5ad, backed='r')  # backed mode — no X loaded

    available = [c for c in args.obs_cols if c in adata.obs.columns]
    missing   = [c for c in args.obs_cols if c not in adata.obs.columns]
    if missing:
        print(f"  WARNING: obs columns not found, skipping: {missing}")

    meta = (adata.obs[available]
            .reset_index()
            .rename(columns={'index': 'sample_id'}))
    adata.file.close()

    # ------------------------------------------------------------------ #
    # 3. Join
    # ------------------------------------------------------------------ #
    combined = scores.merge(meta, on='sample_id', how='left')
    n_matched = combined[available[0]].notna().sum() if available else 0
    print(f"Cells matched to metadata: {n_matched} / {len(scores)}")

    # ------------------------------------------------------------------ #
    # 4. Validate group columns
    # ------------------------------------------------------------------ #
    group_cols = [c for c in args.group if c in combined.columns]
    missing_group = [c for c in args.group if c not in combined.columns]
    if missing_group:
        print(f"WARNING: group columns not found: {missing_group}")
    if not group_cols:
        print("ERROR: no valid group columns found.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 5. Summary table: mean + count per group combination
    # ------------------------------------------------------------------ #
    summary = (combined
               .groupby(group_cols, observed=True)[cav_cols]
               .agg(['mean', 'count'])
               .round(3))

    # Flatten multi-level columns: (cav, stat) -> "cav_mean" / "cav_n"
    summary.columns = [f"{cav}_{stat}" if stat != 'count'
                       else f"n_cells"
                       for cav, stat in summary.columns]

    # Keep only one n_cells column (they're all the same)
    n_col = [c for c in summary.columns if c == 'n_cells']
    other = [c for c in summary.columns if c != 'n_cells']
    summary = summary[other + [n_col[0]]].copy()
    summary.columns = other + ['n_cells']

    # Filter small groups
    summary = summary[summary['n_cells'] >= args.min_cells]

    print(f"\n{'='*70}")
    print(f"Mean CAV scores by: {' x '.join(group_cols)}")
    print(f"{'='*70}")
    print(summary.to_string())

    # ------------------------------------------------------------------ #
    # 6. Pivot view: one CAV at a time, last group column as pivot
    # ------------------------------------------------------------------ #
    if len(group_cols) >= 2:
        row_col  = group_cols[0]
        pivot_col = group_cols[-1]

        for cav in cav_cols:
            col_name = f"{cav}_mean"
            if col_name not in summary.reset_index().columns:
                continue
            try:
                pivot = summary.reset_index().pivot_table(
                    values=col_name,
                    index=row_col,
                    columns=pivot_col,
                    aggfunc='mean'
                ).round(3)
                counts = summary.reset_index().pivot_table(
                    values='n_cells',
                    index=row_col,
                    columns=pivot_col,
                    aggfunc='sum'
                ).fillna(0).astype(int)

                print(f"\n--- CAV: {cav} (mean score) ---")
                print(pivot.to_string())
                print(f"\n--- CAV: {cav} (n cells) ---")
                print(counts.to_string())
            except Exception as e:
                print(f"  Could not pivot {cav}: {e}")

    # ------------------------------------------------------------------ #
    # 7. Save
    # ------------------------------------------------------------------ #
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, sep='\t')
        print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
