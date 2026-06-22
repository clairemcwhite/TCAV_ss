#!/usr/bin/env python3
"""
inspect_go_predictions.py

Pull the raw per-protein predictions for ONE GO term from both CAV and an
external tool (e.g. DeepGOSE), across the validation positives and the proFAB
test negatives.  Useful for understanding why a GO term has a poor tool AUC
(~0.5) in compare_tool_temporal.py output.

For the chosen GO term it assembles one row per protein with:
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
python specific_scripts/inspect_go_predictions.py \\
    --go-term          GO:0070403 \\
    --results          results/temporal_eval_mf/eval_temporal_results.tsv \\
    --out-dir          results/temporal_eval_mf/ \\
    --tool-predictions deepgose_val_testneg.tsv \\
    --tool-format      long_tsv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_SKIP_PARTS = {"sp", "tr", "sw", "ref"}


def normalise_pid(pid: str) -> str:
    """'sp|Q15185|TEBP_HUMAN' → 'Q15185'  |  'Q15185' → 'Q15185'"""
    parts = [p for p in str(pid).split("|") if p and p not in _SKIP_PARTS]
    return parts[0] if parts else str(pid)


def load_tool_scores(path: str, fmt: str, go_term: str) -> dict:
    """Return {normalised protein_id: tool_score} for one GO term (raw, incl. low scores)."""
    if fmt == "long_tsv":
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["protein_id", "go_term", "tool_score"])
        df = df[df["go_term"] == go_term].copy()
        df["protein_id"] = df["protein_id"].astype(str).apply(normalise_pid)
        df["tool_score"] = pd.to_numeric(df["tool_score"], errors="coerce")
        df = df.dropna(subset=["tool_score"])
        df = df.groupby("protein_id", as_index=False)["tool_score"].max()
        return df.set_index("protein_id")["tool_score"].to_dict()

    # wide_csv: protein column + one column per GO term
    wide = pd.read_csv(path)
    pid_col = wide.columns[0]
    if go_term not in wide.columns:
        logger.warning(f"GO term {go_term} not a column in {path}")
        return {}
    sub = wide[[pid_col, go_term]].copy()
    sub[pid_col] = sub[pid_col].astype(str).apply(normalise_pid)
    sub[go_term] = pd.to_numeric(sub[go_term], errors="coerce")
    sub = sub.dropna(subset=[go_term])
    return sub.groupby(pid_col)[go_term].max().to_dict()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--go-term", required=True, help="e.g. GO:0070403")
    parser.add_argument("--results", required=True,
                        help="eval_temporal_results.tsv (CAV val positives).")
    parser.add_argument("--out-dir", required=True,
                        help="Directory with {GO}_test_neg_scores.tsv (and _test_pos_scores.tsv).")
    parser.add_argument("--tool-predictions", required=True,
                        help="External tool predictions file.")
    parser.add_argument("--tool-format", choices=["wide_csv", "long_tsv"], default="long_tsv")
    parser.add_argument("--output", default=None,
                        help="Optional TSV path to write the assembled table.")
    parser.add_argument("--include-test-pos", action="store_true",
                        help="Also include proFAB test positives ({GO}_test_pos_scores.tsv).")
    args = parser.parse_args()

    go_term = args.go_term
    out_dir = Path(args.out_dir)

    # ------------------------------------------------------------------
    # Tool scores for this GO term (raw)
    # ------------------------------------------------------------------
    tool_scores = load_tool_scores(args.tool_predictions, args.tool_format, go_term)
    logger.info(f"Tool made {len(tool_scores)} predictions for {go_term}")

    pieces = []

    # ------------------------------------------------------------------
    # CAV val positives (from results TSV)
    # ------------------------------------------------------------------
    results = pd.read_csv(args.results, sep="\t")
    val = results[results["go_term"] == go_term].copy()
    if len(val):
        val["protein_id_norm"] = val["protein_id"].apply(normalise_pid)
        pieces.append(pd.DataFrame({
            "protein_id": val["protein_id_norm"].values,
            "split":      "val_pos",
            "cav_score":  val["val_cav_score"].values.astype(float),
            "llr":        val["llr"].values.astype(float),
        }))
    else:
        logger.warning(f"No val positives for {go_term} in {args.results}")

    # ------------------------------------------------------------------
    # CAV test negatives (+ optional test positives) from cached score files
    # ------------------------------------------------------------------
    splits = [("test_neg", out_dir / f"{go_term}_test_neg_scores.tsv")]
    if args.include_test_pos:
        splits.append(("test_pos", out_dir / f"{go_term}_test_pos_scores.tsv"))

    for split_name, fpath in splits:
        if not fpath.exists():
            logger.warning(f"Missing {fpath}")
            continue
        d = pd.read_csv(fpath, sep="\t")
        d["protein_id"] = d["protein_id"].apply(normalise_pid)
        pieces.append(pd.DataFrame({
            "protein_id": d["protein_id"].values,
            "split":      split_name,
            "cav_score":  d["cav_score"].values.astype(float),
            "llr":        np.nan,
        }))

    if not pieces:
        raise SystemExit(f"No data found for {go_term} — check inputs.")

    table = pd.concat(pieces, ignore_index=True)

    # ------------------------------------------------------------------
    # Attach tool scores; mark proteins the tool never scored
    # ------------------------------------------------------------------
    table["tool_score"]     = table["protein_id"].map(tool_scores)
    table["tool_predicted"] = table["tool_score"].notna()
    table = table.sort_values(["split", "cav_score"], ascending=[True, False]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"GO term {go_term} — prediction inspection")
    print(f"{'='*70}")
    for split_name, grp in table.groupby("split"):
        n          = len(grp)
        n_tool     = int(grp["tool_predicted"].sum())
        tool_vals  = grp.loc[grp["tool_predicted"], "tool_score"]
        cav_mean   = grp["cav_score"].mean()
        print(f"\n[{split_name}]  n={n}")
        print(f"  CAV score   : mean={cav_mean:.3f}  "
              f"min={grp['cav_score'].min():.3f}  max={grp['cav_score'].max():.3f}")
        print(f"  Tool scored : {n_tool}/{n} ({100*n_tool/n:.1f}%)")
        if n_tool:
            print(f"  Tool score  : mean={tool_vals.mean():.3f}  "
                  f"min={tool_vals.min():.3f}  max={tool_vals.max():.3f}")

    print(f"\n{'-'*70}\nFull table:\n")
    show = table.copy()
    show["llr"]        = show["llr"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    show["tool_score"] = show["tool_score"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    show["cav_score"]  = show["cav_score"].map(lambda x: f"{x:.3f}")
    print(show[["protein_id", "split", "cav_score", "llr", "tool_score"]].to_string(index=False))

    if args.output:
        table.to_csv(args.output, sep="\t", index=False, float_format="%.6f")
        logger.info(f"Wrote table to {args.output}")


if __name__ == "__main__":
    main()
