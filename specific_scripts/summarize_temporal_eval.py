#!/usr/bin/env python3
"""
summarize_temporal_eval.py

Compute global and per-GO-term summary statistics from eval_go_temporal.py output.
Optionally annotates each GO term with its depth and ancestor count from the OBO
hierarchy, and reports Spearman correlation between specificity and AUC.

Usage
-----
python specific_scripts/summarize_temporal_eval.py \\
    --results  results/temporal_eval_mf/eval_temporal_results.tsv \\
    --out-dir  results/temporal_eval_mf/ \\
    --go-obo   /path/to/go.obo \\
    --ont      mf
"""

import argparse
import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ONT_ROOTS = {
    "mf": "GO:0003674",
    "bp": "GO:0008150",
    "cc": "GO:0005575",
}


# ---------------------------------------------------------------------------
# OBO parsing and hierarchy metrics
# ---------------------------------------------------------------------------

def parse_obo(obo_file: str) -> dict:
    """Return {term_id: {"parents": [...], "namespace": str, "name": str}} for non-obsolete terms."""
    terms = {}
    current_id = current_namespace = current_name = None
    current_parents = []
    current_is_obsolete = in_term = False

    with open(obo_file) as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if in_term and current_id and not current_is_obsolete:
                    terms[current_id] = {"parents": current_parents,
                                         "namespace": current_namespace,
                                         "name": current_name or ""}
                in_term = True
                current_id = current_namespace = current_name = None
                current_parents = []
                current_is_obsolete = False
            elif line == "[Typedef]":
                if in_term and current_id and not current_is_obsolete:
                    terms[current_id] = {"parents": current_parents,
                                         "namespace": current_namespace,
                                         "name": current_name or ""}
                in_term = False
            elif in_term:
                if line.startswith("id: "):
                    current_id = line[4:].strip()
                elif line.startswith("name: "):
                    current_name = line[6:].strip()
                elif line.startswith("namespace: "):
                    current_namespace = line[11:].strip()
                elif line.startswith("is_a: "):
                    current_parents.append(line[6:].split()[0])
                elif line.startswith("relationship: part_of "):
                    current_parents.append(line[22:].split()[0])
                elif line.startswith("is_obsolete: true"):
                    current_is_obsolete = True

    if in_term and current_id and not current_is_obsolete:
        terms[current_id] = {"parents": current_parents, "namespace": current_namespace,
                              "name": current_name or ""}
    return terms


def compute_depths(obo_terms: dict, root_id: str) -> dict:
    """BFS from root → minimum depth of each reachable term."""
    # Build parent → children index
    children = {t: [] for t in obo_terms}
    for t, info in obo_terms.items():
        for parent in info["parents"]:
            if parent in children:
                children[parent].append(t)

    depths = {root_id: 0}
    queue = deque([root_id])
    while queue:
        term = queue.popleft()
        for child in children[term]:
            if child not in depths:
                depths[child] = depths[term] + 1
                queue.append(child)
    return depths


def compute_ancestor_counts(obo_terms: dict) -> dict:
    """For each term, count all unique ancestors (walk up via parents)."""
    cache = {}

    def _ancestors(term_id):
        if term_id in cache:
            return cache[term_id]
        result = set()
        for parent in obo_terms.get(term_id, {}).get("parents", []):
            result.add(parent)
            result |= _ancestors(parent)
        cache[term_id] = result
        return result

    return {t: len(_ancestors(t)) for t in obo_terms}


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------

def compute_go_term_auc(val_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(val_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    y_true  = np.concatenate([np.ones(len(val_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([val_scores, neg_scores])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def compute_go_term_aupr(val_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(val_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    y_true  = np.concatenate([np.ones(len(val_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([val_scores, neg_scores])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(average_precision_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", required=True,
                        help="Path to eval_temporal_results.tsv.")
    parser.add_argument("--out-dir", required=True,
                        help="Directory containing {go_id}_test_neg_scores.tsv files.")
    parser.add_argument("--go-obo", default=None,
                        help="Path to go.obo. When provided, adds depth and ancestor "
                             "count per GO term and reports Spearman correlation with AUC.")
    parser.add_argument("--ont", choices=["mf", "bp", "cc"], default="mf",
                        help="Ontology namespace for depth calculation (default: mf).")
    parser.add_argument("--figure-data-dir", default=None,
                        help="If provided, write figure-ready CSVs to this directory.")
    parser.add_argument("--label", default=None,
                        help="Short label for this run (e.g. 'mf', 'bp', 'cc') appended "
                             "to figure-data CSV filenames so multiple ontology runs "
                             "coexist in the same directory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = pd.read_csv(args.results, sep="\t")
    logger.info(f"Loaded {len(results)} protein-term pairs across "
                f"{results['go_term'].nunique()} GO terms")

    # ------------------------------------------------------------------
    # OBO hierarchy metrics (optional)
    # ------------------------------------------------------------------
    depths = ancestor_counts = term_names = None
    if args.go_obo:
        logger.info(f"Parsing OBO: {args.go_obo}")
        obo_terms    = parse_obo(args.go_obo)
        root_id      = ONT_ROOTS[args.ont]
        depths       = compute_depths(obo_terms, root_id)
        ancestor_counts = compute_ancestor_counts(obo_terms)
        term_names   = {t: info["name"] for t, info in obo_terms.items()}
        logger.info(f"Computed depth and ancestor counts for {len(depths)} terms")

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
        auc        = compute_go_term_auc(val_scores, neg_scores)
        aupr       = compute_go_term_aupr(val_scores, neg_scores)

        row = {
            "go_term":                go_id,
            "n_val_proteins":         len(grp),
            "n_test_neg":             len(neg_scores),
            "auc_val_vs_test_neg":    auc,
            "aupr_val_vs_test_neg":   aupr,
            "pct_llr_positive":       float((grp["llr"] > 0).mean() * 100),
            "median_val_cav_score":   float(grp["val_cav_score"].median()),
            "median_test_pos_pct":    float(grp["test_pos_percentile"].median()),
            "median_test_neg_pct":    float(grp["test_neg_percentile"].median()),
            "median_llr":             float(grp["llr"].median()),
            "median_test_pos_zscore": float(grp["test_pos_zscore"].median()),
        }

        if term_names is not None:
            row["go_term_name"]    = term_names.get(go_id, "")
        if depths is not None:
            row["depth"]           = depths.get(go_id, np.nan)
            row["n_ancestors"]     = ancestor_counts.get(go_id, np.nan)

        per_term_rows.append(row)

    per_term = pd.DataFrame(per_term_rows).sort_values("auc_val_vs_test_neg", ascending=False)

    per_term_file = out_dir / "eval_temporal_per_term_summary.tsv"
    per_term.to_csv(per_term_file, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Per-term summary saved to {per_term_file}")

    if args.figure_data_dir:
        fig_dir = Path(args.figure_data_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        label_suffix = f"_{args.label}" if args.label else ""
        out_path = fig_dir / f"temporal_per_term_summary{label_suffix}.csv"
        per_term.to_csv(out_path, index=False, float_format="%.4f")
        logger.info(f"Figure data written to {out_path}")

    # ------------------------------------------------------------------
    # Global stats
    # ------------------------------------------------------------------
    valid_aucs  = per_term["auc_val_vs_test_neg"].dropna()
    macro_auc   = float(valid_aucs.mean())
    pct_llr_pos = float((results["llr"] > 0).mean() * 100)
    median_llr  = float(results["llr"].median())

    print(f"\n{'='*60}")
    print(f"Temporal holdout evaluation — global summary")
    print(f"{'='*60}")
    print(f"  GO terms evaluated          : {results['go_term'].nunique()}")
    print(f"  Protein-term pairs          : {len(results)}")
    print(f"  Unique validation proteins  : {results['protein_id'].nunique()}")
    print(f"\n  Macro-averaged AUC          : {macro_auc:.3f}  "
          f"(n={len(valid_aucs)} GO terms)")
    print(f"  % pairs with LLR > 0        : {pct_llr_pos:.1f}%")
    print(f"  Median LLR                  : {median_llr:.2f}")
    print(f"  Median test_pos_percentile  : {results['test_pos_percentile'].median():.1f}")
    print(f"  Median test_neg_percentile  : {results['test_neg_percentile'].median():.1f}")

    print(f"\n--- AUC distribution across GO terms ---")
    print(valid_aucs.describe().to_string())

    # ------------------------------------------------------------------
    # Specificity vs AUC correlation
    # ------------------------------------------------------------------
    if depths is not None:
        valid = per_term.dropna(subset=["auc_val_vs_test_neg", "depth", "n_ancestors"])

        r_depth, p_depth = spstats.spearmanr(valid["depth"], valid["auc_val_vs_test_neg"])
        r_anc,   p_anc   = spstats.spearmanr(valid["n_ancestors"], valid["auc_val_vs_test_neg"])

        print(f"\n--- GO term specificity vs AUC (Spearman, n={len(valid)}) ---")
        print(f"  Depth        r={r_depth:+.3f}  p={p_depth:.3e}")
        print(f"  N ancestors  r={r_anc:+.3f}  p={p_anc:.3e}")

        print(f"\n--- Depth distribution of evaluated GO terms ---")
        print(valid["depth"].describe().to_string())

    # ------------------------------------------------------------------
    # Top / bottom tables
    # ------------------------------------------------------------------
    cols = ["go_term", "n_val_proteins", "auc_val_vs_test_neg",
            "pct_llr_positive", "median_test_pos_pct", "median_test_neg_pct"]
    if term_names is not None:
        cols += ["go_term_name"]
    if depths is not None:
        cols += ["depth", "n_ancestors"]

    print(f"\n--- Top 10 GO terms by AUC ---")
    print(per_term.head(10)[cols].to_string(index=False))

    print(f"\n--- Bottom 10 GO terms by AUC ---")
    print(per_term.tail(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
