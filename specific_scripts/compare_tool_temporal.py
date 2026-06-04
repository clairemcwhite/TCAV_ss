#!/usr/bin/env python3
"""
compare_tool_temporal.py

Compare an external tool's predictions against CAV temporal holdout results.

The external tool provides a wide-format sparse CSV:
  - Header row: protein_id, GO:0000001, GO:0000016, ...
  - Data rows:  protein_id, [score or empty], ...
  - Empty cells = not predicted (below threshold).

The tool output is assumed to cover only the validation proteins; test negatives
are not in the file. Two complementary metrics are reported:

  1. Spearman correlation between CAV score and tool score across all
     (protein, GO term) validation pairs — requires no negatives.

  2. AUC (val positives vs test negatives), where test negatives receive
     tool score = 0.  This is reasonable because the tool did not flag those
     proteins; it becomes optimistic only if those proteins were never shown
     to the tool.  Treat accordingly.

  3. Recall at threshold: fraction of true val pairs where tool score > 0.

Usage
-----
python specific_scripts/compare_tool_temporal.py \\
    --tool-predictions  path/to/external_predictions.csv \\
    --results           results/temporal_eval_mf/eval_temporal_results.tsv \\
    --per-term-summary  results/temporal_eval_mf/eval_temporal_per_term_summary.tsv \\
    --out-dir           results/temporal_eval_mf/ \\
    --output            results/temporal_eval_mf/tool_comparison.tsv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_SKIP_PARTS = {"sp", "tr", "sw", "ref"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_pid(pid: str) -> str:
    """'sp|Q15185|TEBP_HUMAN' → 'Q15185'  |  'Q15185' → 'Q15185'"""
    parts = [p for p in str(pid).split("|") if p and p not in _SKIP_PARTS]
    return parts[0] if parts else str(pid)


def compute_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    y_true  = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Load external tool predictions → long-format DataFrame
# ---------------------------------------------------------------------------

def load_tool_predictions(csv_path: str, go_terms_needed: set) -> pd.DataFrame:
    """Read the wide CSV, keep only relevant GO columns, melt to long format.

    Returns a DataFrame with columns: protein_id (normalised), go_term, tool_score.
    Only rows where tool_score > 0 are returned (sparse representation).
    """
    logger.info(f"Reading external tool predictions: {csv_path}")
    wide = pd.read_csv(csv_path)

    pid_col   = wide.columns[0]
    go_cols   = [c for c in wide.columns[1:] if c in go_terms_needed]
    n_all_go  = wide.shape[1] - 1
    logger.info(f"  GO terms in file : {n_all_go}  |  overlapping with eval : {len(go_cols)}")
    logger.info(f"  Proteins in file : {len(wide)}")

    # Normalise protein IDs
    wide[pid_col] = wide[pid_col].astype(str).apply(normalise_pid)

    # Keep only relevant GO columns + protein ID
    wide = wide[[pid_col] + go_cols].copy()
    wide = wide.rename(columns={pid_col: "protein_id"})

    # Melt to long format; drop NaN / zero entries
    long = wide.melt(id_vars="protein_id", var_name="go_term", value_name="tool_score")
    long = long.dropna(subset=["tool_score"])
    long = long[long["tool_score"] > 0].reset_index(drop=True)
    long["tool_score"] = long["tool_score"].astype(float)

    logger.info(f"  Non-zero (protein, GO) predictions: {len(long)}")
    return long


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tool-predictions", required=True,
                        help="Wide-format CSV from the external tool.")
    parser.add_argument("--results", required=True,
                        help="Path to eval_temporal_results.tsv (our method).")
    parser.add_argument("--per-term-summary", required=True,
                        help="Path to eval_temporal_per_term_summary.tsv (has CAV AUC).")
    parser.add_argument("--out-dir", required=True,
                        help="Directory containing {go_id}_test_neg_scores.tsv files.")
    parser.add_argument("--output", required=True,
                        help="Output TSV path for per-term comparison table.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # ------------------------------------------------------------------
    # Load our results
    # ------------------------------------------------------------------
    results = pd.read_csv(args.results, sep="\t")
    summary = pd.read_csv(args.per_term_summary, sep="\t")
    go_terms = set(results["go_term"].unique())
    logger.info(f"Loaded {len(results)} val pairs across {len(go_terms)} GO terms")

    results["protein_id_norm"] = results["protein_id"].apply(normalise_pid)

    # ------------------------------------------------------------------
    # Load tool predictions (long format, only nonzero scores)
    # ------------------------------------------------------------------
    tool_long = load_tool_predictions(args.tool_predictions, go_terms_needed=go_terms)

    # ------------------------------------------------------------------
    # Join tool scores onto our validation pairs
    # ------------------------------------------------------------------
    # results has one row per (protein, go_term) where protein truly has that term
    merged = results.merge(
        tool_long,
        left_on=["protein_id_norm", "go_term"],
        right_on=["protein_id",     "go_term"],
        how="left",
    )
    merged["tool_score"] = merged["tool_score"].fillna(0.0)

    # ------------------------------------------------------------------
    # Global Spearman: CAV score vs tool score across all val pairs
    # ------------------------------------------------------------------
    r_global, p_global = spstats.spearmanr(
        merged["val_cav_score"], merged["tool_score"]
    )
    overall_recall = float((merged["tool_score"] > 0).mean())

    # ------------------------------------------------------------------
    # Per-GO-term stats
    # ------------------------------------------------------------------
    rows = []

    for go_id, grp in merged.groupby("go_term"):
        val_cav  = grp["val_cav_score"].values.astype(float)
        val_tool = grp["tool_score"].values.astype(float)

        n_nonzero = int((val_tool > 0).sum())
        recall    = float(n_nonzero / len(grp)) if len(grp) else np.nan

        # Spearman per term (only meaningful when ≥3 pairs)
        if len(grp) >= 3:
            r_term, p_term = spstats.spearmanr(val_cav, val_tool)
        else:
            r_term, p_term = np.nan, np.nan

        # AUC: val positives vs test negatives (negatives scored 0)
        neg_file = out_dir / f"{go_id}_test_neg_scores.tsv"
        if not neg_file.exists():
            neg_tool = np.array([])
        else:
            neg_df   = pd.read_csv(neg_file, sep="\t")
            neg_pids = neg_df["protein_id"].apply(normalise_pid).tolist()
            # Test negatives were not in tool output → score 0
            neg_tool = np.zeros(len(neg_pids))

        tool_auc = compute_auc(val_tool, neg_tool)

        rows.append({
            "go_term":                      go_id,
            "n_val_proteins":               len(grp),
            "tool_auc_0fill_neg":           tool_auc,
            "tool_recall_at_threshold":     recall,
            "tool_n_val_nonzero":           n_nonzero,
            "spearman_r_cav_vs_tool":       r_term,
            "spearman_p_cav_vs_tool":       p_term,
        })

    term_df = pd.DataFrame(rows)

    # Merge with per-term summary (adds CAV AUC, go_term_name, depth, etc.)
    compare = summary.merge(term_df, on="go_term", how="left")
    compare["auc_diff_cav_minus_tool"] = (
        compare["auc_val_vs_test_neg"] - compare["tool_auc_0fill_neg"]
    )

    compare.sort_values("auc_val_vs_test_neg", ascending=False, inplace=True)
    compare.to_csv(args.output, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Comparison saved to {args.output}")

    # ------------------------------------------------------------------
    # Print global summary
    # ------------------------------------------------------------------
    valid = compare.dropna(subset=["auc_val_vs_test_neg", "tool_auc_0fill_neg"])
    n = len(valid)

    cav_macro  = float(valid["auc_val_vs_test_neg"].mean())
    tool_macro = float(valid["tool_auc_0fill_neg"].mean())
    mean_diff  = float(valid["auc_diff_cav_minus_tool"].mean())
    n_cav_wins  = int((valid["auc_diff_cav_minus_tool"] > 0).sum())
    n_tool_wins = int((valid["auc_diff_cav_minus_tool"] < 0).sum())

    print(f"\n{'='*60}")
    print(f"External tool vs CAV — temporal holdout comparison")
    print(f"{'='*60}")
    print(f"  GO terms compared                 : {n}")
    print(f"  Val (protein, GO) pairs           : {len(merged)}")

    print(f"\n--- Score agreement (no negatives needed) ---")
    print(f"  Spearman r  CAV vs tool score     : {r_global:+.3f}  (p={p_global:.2e})")
    print(f"  Tool recall at threshold          : {overall_recall:.1%}  "
          f"(val pairs with tool score > 0)")

    print(f"\n--- AUC comparison (test negatives scored 0 for tool) ---")
    print(f"  Macro-AUC  CAV                    : {cav_macro:.3f}")
    print(f"  Macro-AUC  tool (0-fill negs)     : {tool_macro:.3f}")
    print(f"  Mean AUC diff (CAV − tool)        : {mean_diff:+.3f}")
    print(f"  GO terms where CAV wins           : {n_cav_wins} / {n}")
    print(f"  GO terms where tool wins          : {n_tool_wins} / {n}")

    print(f"\n--- CAV AUC distribution ---")
    print(valid["auc_val_vs_test_neg"].describe().to_string())
    print(f"\n--- Tool AUC distribution ---")
    print(valid["tool_auc_0fill_neg"].describe().to_string())

    # Display columns for tables
    name_col   = ["go_term_name"] if "go_term_name" in compare.columns else []
    depth_cols = ["depth", "n_ancestors"] if "depth" in compare.columns else []
    display = (
        ["go_term"] + name_col +
        ["n_val_proteins", "auc_val_vs_test_neg", "tool_auc_0fill_neg",
         "auc_diff_cav_minus_tool", "tool_recall_at_threshold",
         "spearman_r_cav_vs_tool"] +
        depth_cols
    )

    print(f"\n--- Top 10 GO terms by CAV AUC ---")
    print(compare.head(10)[display].to_string(index=False))

    print(f"\n--- Bottom 10 GO terms by CAV AUC ---")
    print(compare.tail(10)[display].to_string(index=False))

    print(f"\n--- GO terms where tool most outperforms CAV ---")
    print(compare.nsmallest(10, "auc_diff_cav_minus_tool")[display].to_string(index=False))


if __name__ == "__main__":
    main()
