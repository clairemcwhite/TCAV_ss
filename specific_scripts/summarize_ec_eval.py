#!/usr/bin/env python3
"""
summarize_ec_eval.py

Compute global and per-EC-term summary statistics from eval_ec.py output.

Usage
-----
python specific_scripts/summarize_ec_eval.py \\
    --results  results/ec_eval/eval_ec_results.tsv \\
    --out-dir  results/ec_eval/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------

def compute_auc(val_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(val_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    y_true  = np.concatenate([np.ones(len(val_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([val_scores, neg_scores])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", required=True,
                        help="Path to eval_ec_results.tsv.")
    parser.add_argument("--out-dir", required=True,
                        help="Directory containing {ec_cav_id}_test_neg_scores.tsv files.")
    parser.add_argument("--exclude", default=None,
                        help="Optional TSV with ec_number and protein_id columns to exclude "
                             "(e.g. train/val overlap pairs).")
    parser.add_argument("--figure-data-dir", default=None,
                        help="If provided, write figure-ready CSVs to this directory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = pd.read_csv(args.results, sep="\t")
    logger.info(f"Loaded {len(results)} protein-term pairs across "
                f"{results['ec_number'].nunique()} EC terms")

    if args.exclude:
        excl = pd.read_csv(args.exclude, sep="\t")[["ec_number", "protein_id"]]
        excl["_drop"] = True
        before = len(results)
        results = results.merge(excl, on=["ec_number", "protein_id"], how="left")
        results = results[results["_drop"].isna()].drop(columns="_drop").reset_index(drop=True)
        logger.info(f"Excluded {before - len(results)} pairs via --exclude; "
                    f"{len(results)} pairs remain")

    # ------------------------------------------------------------------
    # Per-EC-term stats
    # ------------------------------------------------------------------
    per_term_rows = []

    for ec_num, grp in results.groupby("ec_number"):
        ec_cav_id = grp["ec_cav_id"].iloc[0]
        neg_file  = out_dir / f"{ec_cav_id}_test_neg_scores.tsv"

        if not neg_file.exists():
            logger.warning(f"{ec_num}: neg scores file not found — skipping AUC")
            neg_scores = np.array([])
        else:
            neg_scores = pd.read_csv(neg_file, sep="\t")["cav_score"].values.astype(float)

        val_scores = grp["val_cav_score"].values.astype(float)
        auc        = compute_auc(val_scores, neg_scores)

        per_term_rows.append({
            "ec_number":              ec_num,
            "ec_cav_id":              ec_cav_id,
            "n_val_proteins":         len(grp),
            "n_test_neg":             len(neg_scores),
            "auc_val_vs_test_neg":    auc,
            "pct_llr_positive":       float((grp["llr"] > 0).mean() * 100),
            "median_val_cav_score":   float(grp["val_cav_score"].median()),
            "median_test_pos_pct":    float(grp["test_pos_percentile"].median()),
            "median_test_neg_pct":    float(grp["test_neg_percentile"].median()),
            "median_llr":             float(grp["llr"].median()),
            "median_test_pos_zscore": float(grp["test_pos_zscore"].median()),
        })

    per_term = pd.DataFrame(per_term_rows).sort_values("auc_val_vs_test_neg", ascending=False)

    per_term_file = out_dir / "eval_ec_per_term_summary.tsv"
    per_term.to_csv(per_term_file, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Per-term summary saved to {per_term_file}")

    if args.figure_data_dir:
        fig_dir = Path(args.figure_data_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / "ec_per_term_summary.csv"
        per_term.to_csv(out_path, index=False, float_format="%.4f")
        logger.info(f"Figure data written to {out_path}")

    # ------------------------------------------------------------------
    # Global stats
    # ------------------------------------------------------------------
    valid_aucs  = per_term["auc_val_vs_test_neg"].dropna()
    macro_auc   = float(valid_aucs.mean()) if len(valid_aucs) else float("nan")
    pct_llr_pos = float((results["llr"] > 0).mean() * 100)
    median_llr  = float(results["llr"].median())

    print(f"\n{'='*60}")
    print(f"EC evaluation — global summary")
    print(f"{'='*60}")
    print(f"  EC terms evaluated          : {results['ec_number'].nunique()}")
    print(f"  Protein-term pairs          : {len(results)}")
    print(f"  Unique validation proteins  : {results['protein_id'].nunique()}")
    print(f"\n  Macro-averaged AUC          : {macro_auc:.3f}  "
          f"(n={len(valid_aucs)} EC terms with neg scores)")
    print(f"  % pairs with LLR > 0        : {pct_llr_pos:.1f}%")
    print(f"  Median LLR                  : {median_llr:.2f}")
    print(f"  Median test_pos_percentile  : {results['test_pos_percentile'].median():.1f}")
    print(f"  Median test_neg_percentile  : {results['test_neg_percentile'].median():.1f}")

    print(f"\n--- AUC distribution across EC terms ---")
    if len(valid_aucs):
        print(valid_aucs.describe().to_string())
    else:
        print("  (no neg score files found — run eval_ec.py to generate them)")

    print(f"\n--- LLR distribution ---")
    print(results["llr"].describe().to_string())

    # ------------------------------------------------------------------
    # Top / bottom tables
    # ------------------------------------------------------------------
    cols = ["ec_number", "ec_cav_id", "n_val_proteins", "n_test_neg",
            "auc_val_vs_test_neg", "pct_llr_positive",
            "median_llr", "median_test_pos_pct", "median_test_neg_pct"]
    cols = [c for c in cols if c in per_term.columns]

    n_show = min(10, len(per_term))

    print(f"\n--- Top {n_show} EC terms by AUC ---")
    print(per_term.head(n_show)[cols].to_string(index=False))

    if len(per_term) > n_show:
        print(f"\n--- Bottom {n_show} EC terms by AUC ---")
        print(per_term.tail(n_show)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
