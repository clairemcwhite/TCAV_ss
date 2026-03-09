#!/usr/bin/env python3
"""
Generate positive/negative spans files for CAV training from scRNA-seq metadata.

Each output file is a plain list of cell IDs (one per line, no tab columns)
— i.e. whole-sequence mode — ready to be passed to prepare_embeddings.py.

Concepts
--------
Two concept axes are supported, alone or combined:

  --phase    <phase_label>          e.g. "secretory_early"
  --phases   <label> [<label> ...]  pool multiple phases into one group
  --cell-type <cell_type>           e.g. "Stromal fibroblasts"

Positives are always cells that match the requested label(s).
Negatives are cells that match the contrast label(s):
  --neg-phase    <label>            specific contrast phase
  --neg-phases   <label> [...]      pool multiple phases as the negative group
  --neg-cell-type <label>           specific contrast cell type
  (if omitted, all non-matching cells are used as negatives)

Pooled-phase convenience groups (can be used with --phases / --neg-phases):
  proliferative  →  proliferative_early, proliferative_late
  secretory      →  secretory_early, secretory_mid, secretory_late

Confounder control
------------------
  --restrict-cell-type <type>  only use cells of this type (removes cell-type
                                confound when training phase CAVs)
  --restrict-phase <phase>     only use cells from this phase (removes phase
                                confound when training cell-type CAVs)

Donor hold-out
--------------
  --holdout-donor <donor_id>   exclude this donor from BOTH sets (use for
                                held-out evaluation; can be repeated)

Examples
--------
# Proliferative vs secretory (recommended first CAV), Stromal fibroblasts:
python specific_scripts/make_scrnaseq_spans.py \\
    --metadata data/GSE111976_summary_C1_day_donor_ctype.csv \\
    --donor-phase data/GSE111976_summary_C1_donor_phase.csv \\
    --phases proliferative \\
    --neg-phases secretory \\
    --restrict-cell-type "Stromal fibroblasts" \\
    --out-pos spans/prolif_vs_sec_stromal_pos.txt \\
    --out-neg spans/prolif_vs_sec_stromal_neg.txt

# Same contrast with donor hold-out for validation:
python specific_scripts/make_scrnaseq_spans.py \\
    --metadata data/GSE111976_summary_C1_day_donor_ctype.csv \\
    --donor-phase data/GSE111976_summary_C1_donor_phase.csv \\
    --phases proliferative \\
    --neg-phases secretory \\
    --restrict-cell-type "Stromal fibroblasts" \\
    --holdout-donor 40 --holdout-donor 56 \\
    --out-pos spans/train_pos.txt \\
    --out-neg spans/train_neg.txt

# Held-out evaluation set:
python specific_scripts/make_scrnaseq_spans.py \\
    --metadata data/GSE111976_summary_C1_day_donor_ctype.csv \\
    --donor-phase data/GSE111976_summary_C1_donor_phase.csv \\
    --phases proliferative \\
    --neg-phases secretory \\
    --restrict-cell-type "Stromal fibroblasts" \\
    --keep-only-donors 40 56 \\
    --out-pos spans/eval_pos.txt \\
    --out-neg spans/eval_neg.txt

# Single-phase contrast (e.g. secretory_early vs menstrual):
python specific_scripts/make_scrnaseq_spans.py \\
    --metadata data/GSE111976_summary_C1_day_donor_ctype.csv \\
    --donor-phase data/GSE111976_summary_C1_donor_phase.csv \\
    --phase secretory_early \\
    --neg-phase menstrual \\
    --restrict-cell-type "Stromal fibroblasts" \\
    --out-pos spans/se_vs_men_pos.txt \\
    --out-neg spans/se_vs_men_neg.txt

# Cell type CAV (Macrophages vs Stromal fibroblasts), proliferative phase only:
python specific_scripts/make_scrnaseq_spans.py \\
    --metadata data/GSE111976_summary_C1_day_donor_ctype.csv \\
    --donor-phase data/GSE111976_summary_C1_donor_phase.csv \\
    --cell-type Macrophages \\
    --neg-cell-type "Stromal fibroblasts" \\
    --restrict-phases proliferative \\
    --out-pos spans/ctype_macro_pos.txt \\
    --out-neg spans/ctype_stromal_neg.txt
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Canonical phase order for reference
PHASE_ORDER = [
    'menstrual',
    'proliferative_early',
    'proliferative_late',
    'secretory_early',
    'secretory_mid',
    'secretory_late',
]

# Convenience aliases: a short name expands to a list of canonical phase labels.
# "menstrual" is intentionally left out of both groups — it is a distinct
# transitional state (tissue shedding) that should usually be excluded when
# training a proliferative-vs-secretory CAV.
PHASE_GROUPS: dict[str, list[str]] = {
    'proliferative': ['proliferative_early', 'proliferative_late'],
    'secretory':     ['secretory_early', 'secretory_mid', 'secretory_late'],
}


def expand_phases(labels: list[str]) -> list[str]:
    """
    Expand convenience group names (e.g. 'proliferative') to their constituent
    canonical phase labels.  Unknown labels are passed through unchanged so
    that single canonical names (e.g. 'secretory_early') still work.
    """
    expanded = []
    for label in labels:
        expanded.extend(PHASE_GROUPS.get(label, [label]))
    return expanded


def load_metadata(metadata_path: Path, donor_phase_path: Path | None) -> pd.DataFrame:
    """
    Load the per-cell metadata CSV (GSE111976_summary_*_day_donor_ctype.csv).
    Optionally join phase_canonical from the per-donor phase CSV.

    The cell-level CSVs already contain 'cell_type', 'donor', and 'cell_name'.
    If phase_canonical is not a column in the cell CSV, supply --donor-phase
    to join it from the donor-level file.
    """
    meta = pd.read_csv(metadata_path, index_col=0)
    meta.index = meta.index.astype(str)

    if 'phase_canonical' not in meta.columns:
        if donor_phase_path is None:
            raise ValueError(
                "'phase_canonical' column not found in metadata. "
                "Supply --donor-phase to join it from the donor-level CSV."
            )
        dphase = pd.read_csv(donor_phase_path, index_col=0)
        # Donor phase CSVs have 'donor' and 'phase_canonical'
        if 'donor' in dphase.columns:
            dphase = dphase.set_index('donor')
        dphase = dphase[['phase_canonical']]
        meta = meta.join(dphase, on='donor')
        logger.info("Joined phase_canonical from donor-phase file")

    logger.info(
        f"Loaded {len(meta)} cells. "
        f"Phases: {sorted(meta['phase_canonical'].dropna().unique())}. "
        f"Cell types: {sorted(meta['cell_type'].dropna().unique())}. "
        f"Donors: {sorted(meta['donor'].unique())}."
    )
    return meta


def apply_filters(
    meta: pd.DataFrame,
    pos_phases: list[str],          # expanded canonical phase labels for positives
    neg_phases: list[str],          # expanded canonical phase labels for negatives
    cell_type: str | None,
    restrict_cell_type: str | None,
    restrict_phases: list[str],     # expanded; restrict BOTH sets to these phases
    holdout_donors: list,
    keep_only_donors: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split metadata into positive and negative sets.
    Returns (positives_df, negatives_df).
    """
    df = meta.copy()

    # ---- donor filtering ------------------------------------------------
    if holdout_donors:
        holdout_set = {str(d) for d in holdout_donors}
        df = df[~df['donor'].astype(str).isin(holdout_set)]
        logger.info(f"Held out donors {holdout_donors}: {len(df)} cells remaining")

    if keep_only_donors:
        keep_set = {str(d) for d in keep_only_donors}
        df = df[df['donor'].astype(str).isin(keep_set)]
        logger.info(f"Keeping only donors {keep_only_donors}: {len(df)} cells")

    # ---- confounder restrictions -----------------------------------------
    if restrict_cell_type:
        df = df[df['cell_type'] == restrict_cell_type]
        logger.info(f"Restricted to cell type '{restrict_cell_type}': {len(df)} cells")
        if len(df) == 0:
            raise ValueError(
                f"No cells remain after restricting to cell type '{restrict_cell_type}'. "
                f"Available: {sorted(meta['cell_type'].unique())}"
            )

    if restrict_phases:
        df = df[df['phase_canonical'].isin(restrict_phases)]
        logger.info(f"Restricted to phases {restrict_phases}: {len(df)} cells")
        if len(df) == 0:
            raise ValueError(
                f"No cells remain after restricting to phases {restrict_phases}. "
                f"Available: {sorted(meta['phase_canonical'].unique())}"
            )

    # ---- positive / negative masks --------------------------------------
    pos_mask = pd.Series(True, index=df.index)
    neg_mask = pd.Series(True, index=df.index)

    if pos_phases:
        pos_mask &= df['phase_canonical'].isin(pos_phases)
        # Default negatives: anything not in pos_phases
        # (overridden below if neg_phases is specified)
        neg_mask &= ~df['phase_canonical'].isin(pos_phases)

    if neg_phases:
        neg_mask = df['phase_canonical'].isin(neg_phases)

    if cell_type:
        pos_mask &= df['cell_type'] == cell_type
        neg_mask &= df['cell_type'] != cell_type

    positives = df[pos_mask]
    negatives = df[neg_mask]

    return positives, negatives


def summarize(label: str, df: pd.DataFrame) -> None:
    """Log a breakdown of a cell set by phase and cell type."""
    logger.info(f"--- {label}: {len(df)} cells ---")
    if len(df) == 0:
        logger.warning("  (empty set)")
        return
    by_phase = df.groupby('phase_canonical').size()
    for phase, n in by_phase.items():
        logger.info(f"  phase={phase}: {n} cells")
    by_ctype = df.groupby('cell_type').size()
    for ct, n in by_ctype.items():
        logger.info(f"  cell_type={ct}: {n} cells")
    by_donor = df.groupby('donor').size()
    n_donors  = len(by_donor)
    logger.info(f"  {n_donors} donors: {dict(by_donor)}")


def write_spans(cell_ids: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(cell_ids) + '\n')
    logger.info(f"Wrote {len(cell_ids)} cell IDs → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate positive/negative spans files for CAV training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument('--metadata', required=True,
                        help='Per-cell metadata CSV (GSE111976_summary_*_day_donor_ctype.csv). '
                             'Row index = cell ID, columns include cell_type, donor, '
                             'and optionally phase_canonical.')
    parser.add_argument('--donor-phase',
                        help='Per-donor phase CSV (GSE111976_summary_*_donor_phase.csv). '
                             'Required only if phase_canonical is absent from --metadata.')

    # Concept definition — phase axis
    phase_grp = parser.add_mutually_exclusive_group()
    phase_grp.add_argument('--phase',
                           help='Single positive phase (e.g. "secretory_early").')
    phase_grp.add_argument('--phases', nargs='+', metavar='PHASE',
                           help='One or more positive phases to pool. '
                                'Use the shorthand "proliferative" or "secretory" '
                                'to expand to all sub-phases automatically.')

    neg_phase_grp = parser.add_mutually_exclusive_group()
    neg_phase_grp.add_argument('--neg-phase',
                               help='Single negative phase.')
    neg_phase_grp.add_argument('--neg-phases', nargs='+', metavar='PHASE',
                               help='One or more negative phases to pool. '
                                    'Accepts "proliferative" / "secretory" shorthands.')

    # Concept definition — cell type axis
    parser.add_argument('--cell-type',
                        help='Positive cell type label (e.g. "Stromal fibroblasts").')
    parser.add_argument('--neg-cell-type',
                        help='Negative cell type label. Defaults to all non-matching types.')

    # Confounder control
    parser.add_argument('--restrict-cell-type',
                        help='Only use cells of this type (controls cell-type confound '
                             'when training phase CAVs).')
    restrict_phase_grp = parser.add_mutually_exclusive_group()
    restrict_phase_grp.add_argument('--restrict-phase',
                                    help='Only use cells from this single phase.')
    restrict_phase_grp.add_argument('--restrict-phases', nargs='+', metavar='PHASE',
                                    help='Only use cells from these phases (pooled). '
                                         'Accepts "proliferative" / "secretory" shorthands.')

    # Donor filtering
    parser.add_argument('--holdout-donor', action='append', dest='holdout_donors',
                        metavar='DONOR_ID', default=[],
                        help='Exclude this donor from both sets (for held-out evaluation). '
                             'Repeat to hold out multiple donors.')
    parser.add_argument('--keep-only-donors', nargs='+', metavar='DONOR_ID',
                        help='Keep ONLY these donors (use to build a held-out eval set).')

    # Output
    parser.add_argument('--out-pos', required=True,
                        help='Output path for positive spans file.')
    parser.add_argument('--out-neg', required=True,
                        help='Output path for negative spans file.')

    args = parser.parse_args()

    if not args.phase and not args.phases and not args.cell_type:
        parser.error("Specify at least one of --phase/--phases or --cell-type.")

    # ------------------------------------------------------------------
    # Resolve phase lists (expand shorthands, merge single/multi args)
    # ------------------------------------------------------------------
    pos_phases = expand_phases(
        ([args.phase] if args.phase else []) +
        (args.phases  if args.phases  else [])
    )
    neg_phases = expand_phases(
        ([args.neg_phase] if args.neg_phase else []) +
        (args.neg_phases  if args.neg_phases  else [])
    )
    restrict_phases = expand_phases(
        ([args.restrict_phase]  if args.restrict_phase  else []) +
        (args.restrict_phases   if args.restrict_phases else [])
    )

    if pos_phases:
        logger.info(f"Positive phases: {pos_phases}")
    if neg_phases:
        logger.info(f"Negative phases: {neg_phases}")
    if restrict_phases:
        logger.info(f"Restricting both sets to phases: {restrict_phases}")

    # ------------------------------------------------------------------
    # Load metadata (join phase if needed)
    # ------------------------------------------------------------------
    meta = load_metadata(
        Path(args.metadata),
        Path(args.donor_phase) if args.donor_phase else None,
    )

    # ------------------------------------------------------------------
    # Split into positive / negative
    # ------------------------------------------------------------------
    positives, negatives = apply_filters(
        meta=meta,
        pos_phases=pos_phases,
        neg_phases=neg_phases,
        cell_type=args.cell_type,
        restrict_cell_type=args.restrict_cell_type,
        restrict_phases=restrict_phases,
        holdout_donors=args.holdout_donors,
        keep_only_donors=args.keep_only_donors,
    )

    if args.neg_cell_type:
        negatives = negatives[negatives['cell_type'] == args.neg_cell_type]
        logger.info(
            f"Restricted negatives to cell type '{args.neg_cell_type}': {len(negatives)} cells"
        )

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if len(positives) == 0:
        raise ValueError("Positive set is empty — check your --phase / --cell-type labels.")
    if len(negatives) == 0:
        raise ValueError("Negative set is empty — check your --neg-phase / --neg-cell-type labels.")

    overlap = set(positives.index) & set(negatives.index)
    if overlap:
        raise ValueError(
            f"{len(overlap)} cell IDs appear in both positive and negative sets. "
            "Check your concept definition for contradictions."
        )

    ratio = len(positives) / len(negatives)
    if ratio > 5 or ratio < 0.2:
        logger.warning(
            f"Imbalanced sets: {len(positives)} positives vs {len(negatives)} negatives "
            f"(ratio {ratio:.1f}x). Consider subsampling the larger set for CAV training."
        )

    # ------------------------------------------------------------------
    # Log summaries
    # ------------------------------------------------------------------
    summarize("POSITIVES", positives)
    summarize("NEGATIVES", negatives)

    # ------------------------------------------------------------------
    # Write spans files
    # ------------------------------------------------------------------
    write_spans(list(positives.index), Path(args.out_pos))
    write_spans(list(negatives.index), Path(args.out_neg))

    logger.info("Done.")


if __name__ == '__main__':
    main()
