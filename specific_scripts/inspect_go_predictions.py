#!/usr/bin/env python3
"""
inspect_go_predictions.py

Pull the raw per-protein predictions from both CAV and an external tool
(e.g. DeepGOSE), across the validation positives and the proFAB test
negatives.  Useful for understanding why a GO term has a poor tool AUC
(~0.5) in compare_tool_temporal.py output.

Run for ONE GO term (--go-term) for an inspectable printout, or for ALL
tested GO terms (--all-terms) to dump a combined table with a go_term column.

Each row is one (GO term, protein) pair with:
  - go_term    : the GO term
  - split      : val_pos (validation positive) | test_neg (proFAB test negative)
  - cav_score  : CAV projection score
  - llr        : CAV log-likelihood ratio (val positives only)
  - tool_score : external tool score (NaN if the tool made no prediction)

Data sources (same files compare_tool_temporal.py reads):
  - CAV val positives   : --results            (eval_temporal_results.tsv)
  - CAV test negatives  : --out-dir/{GO}_test_neg_scores.tsv
  - tool predictions    : --tool-predictions   (wide_csv or long_tsv)

Usage
-----
# one term, printed
python specific_scripts/inspect_go_predictions.py \\
    --go-term          GO:0070403 \\
    --results          results/temporal_eval_mf/eval_temporal_results.tsv \\
    --out-dir          results/temporal_eval_mf/ \\
    --tool-predictions deepgose_val_testneg.tsv

# all tested terms, written to TSV
python specific_scripts/inspect_go_predictions.py \\
    --all-terms \\
    --results          results/temporal_eval_mf/eval_temporal_results.tsv \\
    --out-dir          results/temporal_eval_mf/ \\
    --tool-predictions deepgose_val_testneg.tsv \\
    --output           results/temporal_eval_mf/all_term_predictions.tsv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_SKIP_PARTS = {"sp", "tr", "sw", "ref"}


def normalise_pid(pid: str) -> str:
    """'sp|Q15185|TEBP_HUMAN' → 'Q15185'  |  'Q15185' → 'Q15185'"""
    parts = [p for p in str(pid).split("|") if p and p not in _SKIP_PARTS]
    return parts[0] if parts else str(pid)


def compute_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUC of positives vs negatives; NaN if a class is empty or degenerate."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    y_true  = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def load_tool_long(path: str, fmt: str, go_terms_needed: set) -> dict:
    """Return {go_term: {protein_id: tool_score}} for the needed GO terms (raw scores)."""
    if fmt == "long_tsv":
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["protein_id", "go_term", "tool_score"])
        df = df[df["go_term"].isin(go_terms_needed)].copy()
        df["protein_id"] = df["protein_id"].astype(str).apply(normalise_pid)
        df["tool_score"] = pd.to_numeric(df["tool_score"], errors="coerce")
        df = df.dropna(subset=["tool_score"])
        df = df.groupby(["go_term", "protein_id"], as_index=False)["tool_score"].max()
    else:
        wide = pd.read_csv(path)
        pid_col = wide.columns[0]
        go_cols = [c for c in wide.columns[1:] if c in go_terms_needed]
        wide = wide[[pid_col] + go_cols].copy()
        wide[pid_col] = wide[pid_col].astype(str).apply(normalise_pid)
        df = wide.melt(id_vars=pid_col, var_name="go_term", value_name="tool_score")
        df = df.rename(columns={pid_col: "protein_id"})
        df["tool_score"] = pd.to_numeric(df["tool_score"], errors="coerce")
        df = df.dropna(subset=["tool_score"])
        df = df.groupby(["go_term", "protein_id"], as_index=False)["tool_score"].max()

    out: dict = {}
    for go_term, grp in df.groupby("go_term"):
        out[go_term] = grp.set_index("protein_id")["tool_score"].to_dict()
    return out


def build_term_table(go_term: str, results: pd.DataFrame, out_dir: Path,
                     tool_scores: dict, include_test_pos: bool) -> pd.DataFrame:
    """Assemble the per-protein table for one GO term (without tool scores attached)."""
    pieces = []

    val = results[results["go_term"] == go_term]
    if len(val):
        pieces.append(pd.DataFrame({
            "go_term":    go_term,
            "protein_id": val["protein_id"].apply(normalise_pid).values,
            "split":      "val_pos",
            "cav_score":  val["val_cav_score"].values.astype(float),
            "llr":        val["llr"].values.astype(float),
        }))

    splits = [("test_neg", out_dir / f"{go_term}_test_neg_scores.tsv")]
    if include_test_pos:
        splits.append(("test_pos", out_dir / f"{go_term}_test_pos_scores.tsv"))

    for split_name, fpath in splits:
        if not fpath.exists():
            logger.warning(f"Missing {fpath}")
            continue
        d = pd.read_csv(fpath, sep="\t")
        pieces.append(pd.DataFrame({
            "go_term":    go_term,
            "protein_id": d["protein_id"].apply(normalise_pid).values,
            "split":      split_name,
            "cav_score":  d["cav_score"].values.astype(float),
            "llr":        np.nan,
        }))

    if not pieces:
        return pd.DataFrame(columns=["go_term", "protein_id", "split", "cav_score", "llr"])

    table = pd.concat(pieces, ignore_index=True)
    term_tool = tool_scores.get(go_term, {})
    table["tool_score"]     = table["protein_id"].map(term_tool)
    table["tool_predicted"] = table["tool_score"].notna()

    # Per-term AUC: val positives vs test negatives.
    # CAV uses cav_score directly; tool fills missing scores with 0 (matches
    # compare_tool_temporal.py: unpredicted proteins score 0).
    pos = table[table["split"] == "val_pos"]
    neg = table[table["split"] == "test_neg"]
    cav_auc  = compute_auc(pos["cav_score"].values, neg["cav_score"].values)
    tool_auc = compute_auc(pos["tool_score"].fillna(0.0).values,
                           neg["tool_score"].fillna(0.0).values)
    table["cav_auc"]  = cav_auc
    table["tool_auc"] = tool_auc
    table["auc_diff_cav_minus_tool"] = cav_auc - tool_auc
    return table


def print_term(table: pd.DataFrame, go_term: str) -> None:
    print(f"\n{'='*70}")
    print(f"GO term {go_term} — prediction inspection")
    cav_auc  = table["cav_auc"].iloc[0]
    tool_auc = table["tool_auc"].iloc[0]
    print(f"  CAV AUC={cav_auc:.4f}  Tool AUC={tool_auc:.4f}  "
          f"diff(CAV-tool)={cav_auc - tool_auc:+.4f}")
    print(f"{'='*70}")
    for split_name, grp in table.groupby("split"):
        n         = len(grp)
        n_tool    = int(grp["tool_predicted"].sum())
        tool_vals = grp.loc[grp["tool_predicted"], "tool_score"]
        print(f"\n[{split_name}]  n={n}")
        print(f"  CAV score   : mean={grp['cav_score'].mean():.3f}  "
              f"min={grp['cav_score'].min():.3f}  max={grp['cav_score'].max():.3f}")
        print(f"  Tool scored : {n_tool}/{n} ({100*n_tool/n:.1f}%)")
        if n_tool:
            print(f"  Tool score  : mean={tool_vals.mean():.3f}  "
                  f"min={tool_vals.min():.3f}  max={tool_vals.max():.3f}")

    print(f"\n{'-'*70}\nFull table:\n")
    show = table.sort_values(["split", "cav_score"], ascending=[True, False]).copy()
    show["llr"]        = show["llr"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    show["tool_score"] = show["tool_score"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    show["cav_score"]  = show["cav_score"].map(lambda x: f"{x:.3f}")
    print(show[["protein_id", "split", "cav_score", "llr", "tool_score"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--go-term", default=None, help="e.g. GO:0070403 (single-term mode).")
    parser.add_argument("--all-terms", action="store_true",
                        help="Process every GO term in --results (writes a combined table).")
    parser.add_argument("--results", required=True,
                        help="eval_temporal_results.tsv (CAV val positives).")
    parser.add_argument("--out-dir", required=True,
                        help="Directory with {GO}_test_neg_scores.tsv (and _test_pos_scores.tsv).")
    parser.add_argument("--tool-predictions", required=True,
                        help="External tool predictions file.")
    parser.add_argument("--tool-format", choices=["wide_csv", "long_tsv"], default="long_tsv")
    parser.add_argument("--output", default=None,
                        help="TSV path to write the assembled table (required with --all-terms).")
    parser.add_argument("--include-test-pos", action="store_true",
                        help="Also include proFAB test positives ({GO}_test_pos_scores.tsv).")
    args = parser.parse_args()

    if not args.all_terms and not args.go_term:
        parser.error("provide --go-term GO:XXXXXXX or --all-terms")
    if args.all_terms and not args.output:
        parser.error("--all-terms requires --output")

    out_dir = Path(args.out_dir)
    results = pd.read_csv(args.results, sep="\t")

    go_terms = (sorted(results["go_term"].unique())
                if args.all_terms else [args.go_term])
    tool_scores = load_tool_long(args.tool_predictions, args.tool_format, set(go_terms))
    logger.info(f"Loaded tool predictions for {len(tool_scores)} of {len(go_terms)} GO terms")

    tables = []
    for gt in go_terms:
        t = build_term_table(gt, results, out_dir, tool_scores, args.include_test_pos)
        if len(t):
            tables.append(t)

    if not tables:
        raise SystemExit("No data found — check inputs.")

    combined = pd.concat(tables, ignore_index=True)

    if not args.all_terms:
        print_term(combined, args.go_term)
    else:
        n_terms = combined["go_term"].nunique()
        print(f"\nAssembled {len(combined)} rows across {n_terms} GO terms")
        print(combined.groupby("split").size().to_string())

    if args.output:
        # Sort GO terms by AUC difference (largest CAV advantage first), then
        # by split and descending CAV score within each term.
        combined["_split_order"] = combined["split"].map(
            {"val_pos": 0, "test_pos": 1, "test_neg": 2}
        ).fillna(3)
        combined = combined.sort_values(
            ["auc_diff_cav_minus_tool", "go_term", "_split_order", "cav_score"],
            ascending=[False, True, True, False],
        ).drop(columns="_split_order").reset_index(drop=True)
        combined.to_csv(args.output, sep="\t", index=False, float_format="%.6f")
        logger.info(f"Wrote {len(combined)} rows to {args.output}")


if __name__ == "__main__":
    main()
