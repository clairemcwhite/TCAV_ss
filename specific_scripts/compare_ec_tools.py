#!/usr/bin/env python3
"""
compare_ec_tools.py

Compare multiple EC-number prediction tools against each other and against
CAV eval results (from eval_ec.py).

The tool prediction file has:
  - id         : protein identifier
  - ec_number  : true EC number(s), comma-separated (gold standard)
  - one column per tool: predicted EC number(s), comma-separated; "-" = no prediction

Matching is assessed at two levels:
  - exact  : all 4 EC levels match  (4.2.3.158 == 4.2.3.158)
  - level3 : first 3 levels match   (4.2.3.158 vs 4.2.3.x)

Wildcard predictions (4.2.3.-) are normalised to 3 levels for matching.
Wildcard true labels (4.2.3.-) are also normalised; level3 is the max
comparison level for those.

If --cav-results is supplied, CAV recall is added at LLR > 0 and at
configurable percentile thresholds.

Usage
-----
python specific_scripts/compare_ec_tools.py \\
    --tool-predictions  data/price149_predictions.csv \\
    --out-dir           results/ec_eval/ \\
    --cav-results       results/ec_eval/eval_ec_results.tsv
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Columns in the tool CSV that are NOT tool predictions
NON_TOOL_COLS = {"id", "ec_number"}


# ---------------------------------------------------------------------------
# EC matching helpers
# ---------------------------------------------------------------------------

def strip_wildcards(ec: str) -> str:
    """'4.2.3.-' → '4.2.3'   |   '4.2.3.158' → '4.2.3.158'"""
    parts = ec.strip().split(".")
    while parts and parts[-1].strip() in ("-", ""):
        parts.pop()
    return ".".join(parts)


def ec_to_level(ec: str, level: int) -> str | None:
    """Return EC truncated to `level` significant levels, or None if too shallow.

    '4.2.3.158' at level 3 → '4.2.3'
    '4.2.3'     at level 4 → None  (not specific enough)
    """
    norm = strip_wildcards(ec)
    parts = norm.split(".")
    if len(parts) >= level:
        return ".".join(parts[:level])
    return None


def parse_ec_field(field: str) -> list[str]:
    """Split a comma-separated EC field into individual EC strings, drop '-'."""
    if not field or str(field).strip() in ("-", "nan", ""):
        return []
    return [s.strip() for s in str(field).split(",") if s.strip() not in ("-", "")]


def matches_at_level(predicted_field: str, true_field: str, level: int) -> bool:
    """True if any predicted EC matches any true EC at the given level."""
    preds = parse_ec_field(predicted_field)
    trues = parse_ec_field(true_field)
    if not preds or not trues:
        return False
    pred_norms = {ec_to_level(p, level) for p in preds} - {None}
    true_norms = {ec_to_level(t, level) for t in trues} - {None}
    return bool(pred_norms & true_norms)


def matches_exact(predicted_field: str, true_field: str) -> bool:
    """Match at the truth's own specificity level.

    If the truth is '4.2.3' (3-level), a prediction of '4.2.3.158' gets
    credit because its first 3 levels match — the tool is being rewarded
    for predicting a valid specific sub-entry of an underspecified truth.
    If the truth is '4.2.3.158' (4-level), the prediction must match all 4.
    """
    preds = parse_ec_field(predicted_field)
    trues = parse_ec_field(true_field)
    if not preds or not trues:
        return False
    for true_ec in trues:
        true_norm  = strip_wildcards(true_ec)
        true_level = len(true_norm.split("."))
        true_at    = true_norm                           # already at its own level
        for pred in preds:
            pred_at = ec_to_level(pred, true_level)      # truncate prediction to same level
            if pred_at == true_at:
                return True
    return False


def predicted_anything(field: str) -> bool:
    return len(parse_ec_field(str(field))) > 0


# ---------------------------------------------------------------------------
# Load and explode tool predictions
# ---------------------------------------------------------------------------

def load_and_explode(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Read tool CSV, explode multi-EC true labels to one row per (protein, EC).

    Returns (long_df, tool_col_names).
    long_df has columns: protein_id, true_ec, true_ec_norm, <tool cols...>
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("-")
    df = df.rename(columns={"id": "protein_id"})

    tool_cols = [c for c in df.columns if c not in NON_TOOL_COLS and c != "protein_id"]

    # Explode multi-EC true labels
    df["ec_number"] = df["ec_number"].astype(str).str.strip().str.strip('"')
    df["ec_number"] = df["ec_number"].str.split(",")
    df = df.explode("ec_number")
    df["ec_number"] = df["ec_number"].str.strip()
    df = df[df["ec_number"].apply(lambda x: bool(parse_ec_field(x)))].reset_index(drop=True)

    # Normalised true EC (wildcards stripped)
    df["true_ec_norm"] = df["ec_number"].apply(strip_wildcards)

    logger.info(
        f"Loaded {df['protein_id'].nunique()} proteins, "
        f"{df['true_ec_norm'].nunique()} unique EC terms, "
        f"{len(df)} (protein, EC) pairs"
    )
    logger.info(f"Tools found: {tool_cols}")
    return df, tool_cols


# ---------------------------------------------------------------------------
# Per-pair match computation
# ---------------------------------------------------------------------------

def compute_matches(df: pd.DataFrame, tool_cols: list[str]) -> pd.DataFrame:
    """Add boolean match columns for each tool at exact and level-3."""
    for col in tool_cols:
        safe = col.replace("-", "_").replace(" ", "_")
        df[f"{safe}__exact"]  = df.apply(
            lambda r: matches_exact(r[col], r["ec_number"]), axis=1
        )
        df[f"{safe}__level3"] = df.apply(
            lambda r: matches_at_level(r[col], r["ec_number"], 3), axis=1
        )
        df[f"{safe}__predicted"] = df[col].apply(predicted_anything)
    return df


# ---------------------------------------------------------------------------
# Per-tool summary
# ---------------------------------------------------------------------------

def tool_summary(df: pd.DataFrame, tool_cols: list[str]) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    for col in tool_cols:
        safe = col.replace("-", "_").replace(" ", "_")
        n_pred   = df[f"{safe}__predicted"].sum()
        n_exact  = df[f"{safe}__exact"].sum()
        n_level3 = df[f"{safe}__level3"].sum()
        rows.append({
            "tool":                  col,
            "n_pairs":               n_total,
            "n_predicted":           int(n_pred),
            "coverage":              round(n_pred / n_total, 4),
            "n_exact_match":         int(n_exact),
            "recall_exact":          round(n_exact / n_total, 4),
            "n_level3_match":        int(n_level3),
            "recall_level3":         round(n_level3 / n_total, 4),
            # Precision among predictions made
            "precision_exact":       round(n_exact / n_pred, 4) if n_pred else float("nan"),
            "precision_level3":      round(n_level3 / n_pred, 4) if n_pred else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("recall_exact", ascending=False)


# ---------------------------------------------------------------------------
# CAV integration
# ---------------------------------------------------------------------------

def add_cav_metrics(
    df: pd.DataFrame,
    cav_results_path: str,
    llr_threshold: float = 0.0,
    percentile_threshold: float = 50.0,
) -> tuple[pd.DataFrame, dict]:
    """Join CAV eval results onto the per-pair DataFrame.

    Returns (merged_df, cav_summary_dict).
    """
    cav = pd.read_csv(cav_results_path, sep="\t")
    logger.info(f"Loaded {len(cav)} CAV result rows from {cav_results_path}")

    # Normalise CAV ec_number for join
    cav["true_ec_norm"] = cav["ec_number"].apply(strip_wildcards)

    merged = df.merge(
        cav[["protein_id", "true_ec_norm", "llr",
             "val_cav_score", "test_pos_percentile", "test_neg_percentile"]],
        on=["protein_id", "true_ec_norm"],
        how="left",
    )
    n_matched = merged["llr"].notna().sum()
    logger.info(f"CAV scores joined for {n_matched} / {len(merged)} pairs")

    # Binary CAV predictions at thresholds
    merged["cav__llr_positive"]     = merged["llr"] > llr_threshold
    merged["cav__pct50_positive"]   = merged["test_pos_percentile"] > percentile_threshold

    n_total  = len(merged)
    n_scored = int(merged["llr"].notna().sum())
    n_exact_llr  = int(merged["cav__llr_positive"].sum())
    n_exact_pct  = int(merged["cav__pct50_positive"].sum())

    cav_summary = {
        "tool":              "CAV",
        "n_pairs":           n_total,
        "n_predicted":       n_scored,
        "coverage":          round(n_scored / n_total, 4),
        # "Exact" for CAV = LLR > threshold (positively aligned with pos distribution)
        "n_exact_match":     n_exact_llr,
        "recall_exact":      round(n_exact_llr / n_total, 4),
        "n_level3_match":    n_exact_pct,
        "recall_level3":     round(n_exact_pct / n_total, 4),
        "precision_exact":   round(n_exact_llr / n_scored, 4) if n_scored else float("nan"),
        "precision_level3":  round(n_exact_pct / n_scored, 4) if n_scored else float("nan"),
    }

    return merged, cav_summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

RCPARAMS = {"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10}

def make_figures(summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    # Sort by recall_exact for display
    summary = summary.sort_values("recall_exact", ascending=True)
    tools   = summary["tool"].tolist()
    y       = np.arange(len(tools))

    # ------------------------------------------------------------------
    # 1. Horizontal bar: recall at exact and level-3
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, max(4, len(tools) * 0.4)))
    ax.barh(y - 0.2, summary["recall_level3"], height=0.35,
            color="#aec6e8", label="3-level match")
    ax.barh(y + 0.2, summary["recall_exact"],  height=0.35,
            color="#2166ac", label="Exact match (4-level)")
    ax.set_yticks(y)
    ax.set_yticklabels(tools, fontsize=8)
    ax.set_xlabel("Recall  (fraction of gold-standard pairs)")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="0.6", lw=0.8, ls="--")
    ax.legend(frameon=False)
    ax.set_title("EC prediction recall across tools")
    fig.tight_layout()
    p = out_dir / "fig_ec_tool_recall.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 2. Scatter: coverage vs recall_exact
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    is_cav = summary["tool"] == "CAV"
    ax.scatter(
        summary.loc[~is_cav, "coverage"],
        summary.loc[~is_cav, "recall_exact"],
        s=40, color="#2166ac", alpha=0.8, linewidths=0.4, edgecolors="k",
        label="Other tools", zorder=2,
    )
    if is_cav.any():
        ax.scatter(
            summary.loc[is_cav, "coverage"],
            summary.loc[is_cav, "recall_exact"],
            s=80, color="#d6604d", marker="*", linewidths=0.5, edgecolors="k",
            label="CAV", zorder=3,
        )
    for _, row in summary.iterrows():
        ax.annotate(row["tool"], (row["coverage"], row["recall_exact"]),
                    fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, zorder=0)
    ax.set_xlabel("Coverage (fraction of pairs with any prediction)")
    ax.set_ylabel("Recall (exact 4-level match)")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False)
    ax.set_title("EC prediction tools: coverage vs recall")
    fig.tight_layout()
    p = out_dir / "fig_ec_coverage_vs_recall.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tool-predictions", required=True,
                        help="Wide-format CSV with tool predictions and ec_number column.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for results and figures.")
    parser.add_argument("--cav-results", default=None,
                        help="Path to eval_ec_results.tsv (from eval_ec.py). Optional.")
    parser.add_argument("--llr-threshold", type=float, default=0.0,
                        help="LLR threshold for CAV binary prediction (default: 0).")
    parser.add_argument("--percentile-threshold", type=float, default=50.0,
                        help="test_pos_percentile threshold for CAV (default: 50).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and compute matches
    # ------------------------------------------------------------------
    df, tool_cols = load_and_explode(args.tool_predictions)

    # ------------------------------------------------------------------
    # Restrict to CAV-covered pairs (must happen before computing matches)
    # ------------------------------------------------------------------
    if args.cav_results:
        cav_pre = pd.read_csv(args.cav_results, sep="\t")
        cav_pre["true_ec_norm"] = cav_pre["ec_number"].apply(strip_wildcards)
        cav_pairs = set(zip(cav_pre["protein_id"], cav_pre["true_ec_norm"]))
        n_before = len(df)
        df = df[df.apply(
            lambda r: (r["protein_id"], r["true_ec_norm"]) in cav_pairs, axis=1
        )].reset_index(drop=True)
        logger.info(
            f"Restricted from {n_before} to {len(df)} pairs "
            f"with CAV coverage ({cav_pre['ec_number'].nunique()} EC terms)"
        )

    df = compute_matches(df, tool_cols)

    # ------------------------------------------------------------------
    # Per-tool summary
    # ------------------------------------------------------------------
    summary = tool_summary(df, tool_cols)

    # ------------------------------------------------------------------
    # CAV integration (optional)
    # ------------------------------------------------------------------
    if args.cav_results:
        df, cav_row = add_cav_metrics(
            df, args.cav_results,
            llr_threshold=args.llr_threshold,
            percentile_threshold=args.percentile_threshold,
        )
        # Note column labels for CAV in summary
        cav_df  = pd.DataFrame([cav_row])
        cav_df["note"] = f"LLR>{args.llr_threshold} / pct>{args.percentile_threshold}"
        summary = pd.concat([summary, cav_df], ignore_index=True)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    summary_file = out_dir / "ec_tool_comparison_summary.tsv"
    summary.to_csv(summary_file, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Summary saved to {summary_file}")

    detail_cols = ["protein_id", "ec_number", "true_ec_norm"]
    for col in tool_cols:
        safe = col.replace("-", "_").replace(" ", "_")
        detail_cols += [f"{safe}__exact", f"{safe}__level3"]
    if args.cav_results:
        detail_cols += ["llr", "val_cav_score", "test_pos_percentile",
                        "cav__llr_positive", "cav__pct50_positive"]

    detail_file = out_dir / "ec_tool_comparison_detail.tsv"
    df[[c for c in detail_cols if c in df.columns]].to_csv(
        detail_file, sep="\t", index=False, float_format="%.4f"
    )
    logger.info(f"Per-pair detail saved to {detail_file}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"EC tool comparison — {len(df)} (protein, EC) pairs")
    print(f"{'='*70}")
    display_cols = ["tool", "n_predicted", "coverage",
                    "recall_exact", "recall_level3",
                    "precision_exact", "precision_level3"]
    if "note" in summary.columns:
        display_cols.append("note")
    print(summary[display_cols].to_string(index=False))

    make_figures(summary, out_dir)


if __name__ == "__main__":
    main()
