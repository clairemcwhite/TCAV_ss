#!/usr/bin/env python3
"""
summarize_temporal_eval.py

Compute global and per-GO-term summary statistics from eval_go_temporal.py output.

For each GO term:
  - AUC treating validation proteins as true positives and test negatives
    as true negatives (uses val_cav_score vs test neg cav_scores)

Global stats reported:
  - Macro-averaged AUC across GO terms
  - % validation protein-term pairs with LLR > 0
  - Median test_pos_percentile
  - Median test_neg_percentile

Usage
-----
python specific_scripts/summarize_temporal_eval.py \\
    --results  results/temporal_eval_mf/eval_temporal_results.tsv \\
    --out-dir  results/temporal_eval_mf/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_go_term_auc(val_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """
    AUC treating val_scores as true positives and neg_scores as true negatives.
    Returns nan if either set is empty or all one class.
    """
    if len(val_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    y_true  = np.concatenate([np.ones(len(val_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([val_scores, neg_scores])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", required=True,
                        help="Path to eval_temporal_results.tsv.")
    parser.add_argument("--out-dir", required=True,
                        help="Directory containing {go_id}_test_neg_scores.tsv files "
                             "and where per-term summary will be saved.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = pd.read_csv(args.results, sep="\t")
    logger.info(f"Loaded {len(results)} protein-term pairs across "
                f"{results['go_term'].nunique()} GO terms")

    # ------------------------------------------------------------------
    # Per-GO-term stats
    # ------------------------------------------------------------------
    per_term_rows = []

    for go_id, grp in results.groupby("go_term"):
        neg_file = out_dir / f"{go_id}_test_neg_scores.tsv"
        if not neg_file.exists():
            logger.warning(f"{go_id}: neg scores file not found — skipping AUC")
            neg_scores = np.array([])
        else:
            neg_scores = pd.read_csv(neg_file, sep="\t")["cav_score"].values.astype(float)

        val_scores = grp["val_cav_score"].values.astype(float)

        auc = compute_go_term_auc(val_scores, neg_scores)

        per_term_rows.append({
            "go_term":                  go_id,
            "n_val_proteins":           len(grp),
            "n_test_neg":               len(neg_scores),
            "auc_val_vs_test_neg":      auc,
            "pct_llr_positive":         float((grp["llr"] > 0).mean() * 100),
            "median_val_cav_score":     float(grp["val_cav_score"].median()),
            "median_test_pos_pct":      float(grp["test_pos_percentile"].median()),
            "median_test_neg_pct":      float(grp["test_neg_percentile"].median()),
            "median_llr":               float(grp["llr"].median()),
            "median_test_pos_zscore":   float(grp["test_pos_zscore"].median()),
        })

    per_term = pd.DataFrame(per_term_rows).sort_values("auc_val_vs_test_neg", ascending=False)

    per_term_file = out_dir / "eval_temporal_per_term_summary.tsv"
    per_term.to_csv(per_term_file, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Per-term summary saved to {per_term_file}")

    # ------------------------------------------------------------------
    # Global stats
    # ------------------------------------------------------------------
    valid_aucs = per_term["auc_val_vs_test_neg"].dropna()
    macro_auc  = float(valid_aucs.mean())

    pct_llr_pos        = float((results["llr"] > 0).mean() * 100)
    median_pos_pct     = float(results["test_pos_percentile"].median())
    median_neg_pct     = float(results["test_neg_percentile"].median())
    median_llr         = float(results["llr"].median())

    print(f"\n{'='*60}")
    print(f"Temporal holdout evaluation — global summary")
    print(f"{'='*60}")
    print(f"  GO terms evaluated          : {results['go_term'].nunique()}")
    print(f"  Protein-term pairs          : {len(results)}")
    print(f"  Unique validation proteins  : {results['protein_id'].nunique()}")
    print(f"\n  Macro-averaged AUC          : {macro_auc:.3f}  "
          f"(n={len(valid_aucs)} GO terms with valid AUC)")
    print(f"  % pairs with LLR > 0        : {pct_llr_pos:.1f}%")
    print(f"  Median LLR                  : {median_llr:.2f}")
    print(f"  Median test_pos_percentile  : {median_pos_pct:.1f}")
    print(f"  Median test_neg_percentile  : {median_neg_pct:.1f}")

    print(f"\n--- AUC distribution across GO terms ---")
    print(valid_aucs.describe().to_string())

    print(f"\n--- Top 10 GO terms by AUC ---")
    print(per_term.head(10)[
        ["go_term", "n_val_proteins", "auc_val_vs_test_neg",
         "pct_llr_positive", "median_test_pos_pct", "median_test_neg_pct"]
    ].to_string(index=False))

    print(f"\n--- Bottom 10 GO terms by AUC ---")
    print(per_term.tail(10)[
        ["go_term", "n_val_proteins", "auc_val_vs_test_neg",
         "pct_llr_positive", "median_test_pos_pct", "median_test_neg_pct"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
