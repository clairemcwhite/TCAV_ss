#!/usr/bin/env python3
"""
de_pseudobulk.py — Standard pseudobulk differential expression for each
tissue × cell-type × normal-vs-cancer pair, using the same pair enumeration
as cav_gene_correlation.py (from library_structure.json).

Two methods available:
  --method deseq2    Pseudobulk DESeq2 via pydeseq2 (preferred).
                     Aggregates raw counts per donor, runs DESeq2.
                     Requires: pip install pydeseq2
  --method wilcoxon  Per-cell Wilcoxon rank-sum via scanpy (fallback).
                     No donor column required.
                     log2FC = log2((mean_cancer + 1e-9) / (mean_normal + 1e-9))
                     on log1p-normalised counts.

Output mirrors cav_gene_correlation.py for direct comparison:
  <out-dir>/
    <pair_name>.tsv         — gene, gene_name, log2fc, pval, padj (sorted by |log2fc|)
    summary_heatmap.png     — top-N genes × pairs heatmap of log2FC
    summary_top_genes.tsv   — wide log2FC table used for heatmap

Usage
-----
python specific_scripts/de_pseudobulk.py \\
    --h5ad          cav_library/.../data/cells.h5ad \\
    --lib-dir       cav_library/.../ \\
    --group-col     cell_type \\
    --context-col   tissue \\
    --condition-col disease \\
    --donor-col     donor_id \\
    --out-dir       cav_library/.../results/de_pseudobulk/

# Compare with CAV gene correlations:
python specific_scripts/de_pseudobulk.py \\
    ... \\
    --cav-dir  cav_library/.../results/gene_correlation/ \\
    --out-dir  cav_library/.../results/de_vs_cav/
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with cav_gene_correlation.py)
# ---------------------------------------------------------------------------

def load_library_structure(lib_dir: Path) -> Optional[dict]:
    path = lib_dir / "library_structure.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def norm(v: str) -> str:
    return v.lower().replace("_", " ").strip()


def safe_labels(series) -> np.ndarray:
    return series.astype(str).replace("nan", "unknown").values


def parse_pair_parts(name: str, sep: str, group_level: int,
                     cond_level: int) -> tuple:
    parts = name.split(sep)
    group_val   = parts[group_level]
    cond_val    = parts[cond_level]
    mid         = parts[group_level + 1 : cond_level]
    context_val = sep.join(mid) if mid else ""
    return group_val, context_val, cond_val


def cells_matching(obs: pd.DataFrame,
                   group_col: str, context_col: str, condition_col: str,
                   group_val: str, context_val: str, cond_val: str) -> np.ndarray:
    def match(col, val):
        if not col or col not in obs.columns:
            return np.ones(len(obs), bool)
        labels = safe_labels(obs[col])
        nval = norm(val)
        return np.array([norm(l) == nval for l in labels])
    return (match(group_col, group_val) &
            match(context_col, context_val) &
            match(condition_col, cond_val))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adata(h5ad_path: str, gene_name_col: str = ""):
    """Load AnnData, return adata + gene_name_map."""
    import anndata as ad
    logger.info(f"Loading h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    logger.info(f"  obs columns: {list(adata.obs.columns)}")
    logger.info(f"  var columns: {list(adata.var.columns)}")

    gene_ids = list(adata.var_names)
    gene_name_map = {}
    col = gene_name_col
    if not col:
        for candidate in ["feature_name", "gene_name", "gene_symbol",
                          "symbol", "name", "Gene", "gene"]:
            if candidate in adata.var.columns:
                col = candidate
                break
    if col and col in adata.var.columns:
        gene_name_map = dict(zip(gene_ids, adata.var[col].astype(str).values))
        logger.info(f"  Gene name column: '{col}'")

    return adata, gene_name_map


# ---------------------------------------------------------------------------
# Pseudobulk DESeq2
# ---------------------------------------------------------------------------

def run_deseq2_pair(adata_pair, donor_col: str,
                    condition_col: str, baseline_value: str,
                    min_donors: int = 3) -> Optional[pd.DataFrame]:
    """
    Aggregate counts per donor, run DESeq2 (pydeseq2).
    Returns DataFrame with gene, log2fc, pval, padj or None if insufficient donors.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError("pydeseq2 not installed. Run: pip install pydeseq2")

    import scipy.sparse as sp

    obs = adata_pair.obs.copy()
    obs["_condition"] = obs[condition_col].astype(str)
    obs["_donor"]     = obs[donor_col].astype(str) if donor_col else "single_donor"

    # Aggregate raw counts per donor
    donors     = obs["_donor"].unique()
    conditions = obs["_condition"].unique()

    n_cond    = {c: obs[obs["_condition"] == c]["_donor"].nunique() for c in conditions}
    min_n     = min(n_cond.values()) if n_cond else 0
    if min_n < min_donors:
        logger.warning(f"  Insufficient donors per condition {dict(n_cond)} "
                       f"(need ≥ {min_donors}) — skipping DESeq2")
        return None

    rows, meta_rows = [], []
    for donor in sorted(donors):
        mask = (obs["_donor"] == donor).values
        X    = adata_pair.X[mask]
        if sp.issparse(X):
            X = X.toarray()
        rows.append(X.sum(axis=0))
        cond = obs.loc[obs["_donor"] == donor, "_condition"].iloc[0]
        meta_rows.append({"sample": donor, "condition": cond})

    counts_df = pd.DataFrame(
        np.vstack(rows).astype(int),
        index=[r["sample"] for r in meta_rows],
        columns=adata_pair.var_names,
    )
    meta_df = pd.DataFrame(meta_rows).set_index("sample")

    # Filter low-count genes
    keep = counts_df.sum(axis=0) >= len(donors)
    counts_df = counts_df.loc[:, keep]

    logger.info(f"  DESeq2: {len(donors)} donors, {counts_df.shape[1]} genes after filtering")

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=meta_df,
        design="~condition",
        quiet=True,
    )
    dds.deseq2()

    # Determine the condition value that is NOT the baseline
    # Use normalised comparison to handle case/underscore/space mismatches
    unique_conds = meta_df["condition"].unique()
    logger.info(f"    condition values in pseudobulk: {list(unique_conds)}")
    cond_values = [v for v in unique_conds
                   if norm(str(v)) != norm(str(baseline_value))]
    # Also find the exact baseline string as it appears in the data
    baseline_actual = next(
        (v for v in unique_conds if norm(str(v)) == norm(str(baseline_value))),
        baseline_value,
    )
    if not cond_values:
        logger.warning(f"  Could not determine condition value for contrast. "
                       f"Unique values: {list(unique_conds)}, baseline: '{baseline_value}'")
        return None
    condition_value = cond_values[0]

    logger.info(f"    contrast: '{condition_value}' vs '{baseline_actual}'")
    ds = DeseqStats(
        dds,
        contrast=["condition", condition_value, baseline_actual],
        quiet=True,
    )
    ds.summary()

    res = ds.results_df.reset_index()
    res.columns = [c if c != "index" else "gene" for c in res.columns]
    if "gene" not in res.columns:
        res = res.rename(columns={res.columns[0]: "gene"})

    res = res.rename(columns={
        "log2FoldChange": "log2fc",
        "pvalue":         "pval",
        "padj":           "padj",
    })
    return res[["gene", "log2fc", "pval", "padj"]].dropna(subset=["pval"])


# ---------------------------------------------------------------------------
# Wilcoxon fallback
# ---------------------------------------------------------------------------

def run_wilcoxon_pair(adata_pair, condition_col: str,
                      baseline_value: str) -> pd.DataFrame:
    """
    Per-gene Wilcoxon rank-sum on log1p-normalised counts.
    Returns DataFrame with gene, log2fc, pval, padj.
    """
    import scipy.sparse as sp
    import scanpy as sc

    ad_tmp = adata_pair.copy()
    sc.pp.normalize_total(ad_tmp, target_sum=1e4)
    sc.pp.log1p(ad_tmp)

    obs = ad_tmp.obs[condition_col].astype(str).values
    is_baseline = np.array([norm(v) == norm(baseline_value) for v in obs])
    is_condition = ~is_baseline

    X = ad_tmp.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    n_genes = X.shape[1]
    log2fc  = np.zeros(n_genes, dtype=np.float32)
    pvals   = np.ones(n_genes, dtype=np.float64)

    mean_b = X[is_baseline].mean(axis=0)
    mean_c = X[is_condition].mean(axis=0)
    log2fc  = np.log2((mean_c + 1e-9) / (mean_b + 1e-9))

    logger.info(f"  Wilcoxon: {is_baseline.sum()} baseline, "
                f"{is_condition.sum()} condition cells, {n_genes} genes")

    # Vectorize per-gene Wilcoxon is slow for 60k genes; use scipy
    for i in range(n_genes):
        b_vals = X[is_baseline, i]
        c_vals = X[is_condition, i]
        if b_vals.std() == 0 and c_vals.std() == 0:
            continue
        _, pvals[i] = stats.mannwhitneyu(c_vals, b_vals, alternative="two-sided")

    # BH correction
    from statsmodels.stats.multitest import multipletests
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame({
        "gene":   list(adata_pair.var_names),
        "log2fc": log2fc,
        "pval":   pvals,
        "padj":   padj,
    })


# ---------------------------------------------------------------------------
# Summary heatmap
# ---------------------------------------------------------------------------

def make_summary_heatmap(lfc_matrix: pd.DataFrame, out_path: str,
                          top_n: int = 50,
                          figsize_w: float = 14, figsize_h: float = 10):
    if lfc_matrix.empty:
        return
    max_abs = lfc_matrix.abs().max(axis=1)
    top_genes = max_abs.nlargest(top_n).index
    sub = lfc_matrix.loc[top_genes].copy()
    sub.columns = [c.replace("__", " | ").replace("_", " ") for c in sub.columns]
    sub.index   = [g.replace("_", " ") for g in sub.index]

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
    vmax = np.nanpercentile(lfc_matrix.abs().values, 95)
    sns.heatmap(sub, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                ax=ax, xticklabels=True, yticklabels=True,
                linewidths=0.2, linecolor="#dddddd")
    ax.set_title(f"Top {top_n} DE genes by |log2FC| (cancer vs. normal)\n"
                 f"red = higher in cancer, blue = higher in normal", fontsize=11)
    ax.set_xlabel("Pair", fontsize=10)
    ax.set_ylabel("Gene", fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved summary heatmap: {out_path}")


# ---------------------------------------------------------------------------
# CAV comparison
# ---------------------------------------------------------------------------

def compare_with_cav(de_dir: Path, cav_dir: Path, out_path: str,
                     top_n: int = 200):
    """
    For each pair present in both de_dir and cav_dir, compute Spearman r
    between log2FC and CAV Pearson r across the top_n genes (by |log2FC|).
    Saves a scatter summary and a correlation table.
    """
    de_files  = {f.stem: f for f in de_dir.glob("*.tsv")}
    cav_files = {f.stem: f for f in cav_dir.glob("*.tsv")}
    shared    = sorted(set(de_files) & set(cav_files))

    if not shared:
        logger.warning("No matching pair TSVs found between DE and CAV dirs.")
        return

    logger.info(f"Comparing {len(shared)} pairs (DE vs. CAV)")

    records = []
    fig, axes = plt.subplots(
        max(1, (len(shared) + 2) // 3), min(3, len(shared)),
        figsize=(5 * min(3, len(shared)),
                 4 * max(1, (len(shared) + 2) // 3)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, pair in enumerate(shared):
        try:
            de  = pd.read_csv(de_files[pair],  sep="\t")
            cav = pd.read_csv(cav_files[pair], sep="\t")
        except Exception as e:
            logger.warning(f"  {pair}: could not read file — {e}, skipping")
            continue
        if de.empty or cav.empty:
            logger.warning(f"  {pair}: empty file — skipping")
            continue
        if "log2fc" not in de.columns:
            logger.warning(f"  {pair}: DE file missing 'log2fc' column "
                           f"(has: {list(de.columns)}) — skipping")
            continue
        if "r" not in cav.columns:
            logger.warning(f"  {pair}: CAV file missing 'r' column "
                           f"(has: {list(cav.columns)}) — skipping")
            continue

        # Align on gene
        de  = de.rename(columns={"log2fc": "lfc"})
        cav = cav.rename(columns={"r": "cav_r"})
        merged = de[["gene", "lfc"]].merge(
            cav[["gene", "cav_r"]], on="gene", how="inner"
        ).dropna()

        if len(merged) < 10:
            logger.warning(f"  {pair}: only {len(merged)} shared genes — skipping")
            continue

        # Restrict to top_n by |log2FC| to focus on informative genes
        merged = merged.reindex(
            merged["lfc"].abs().nlargest(top_n).index
        )

        rho, pval = stats.spearmanr(merged["lfc"], merged["cav_r"])
        records.append({"pair": pair, "spearman_r": rho, "pval": pval,
                        "n_genes": len(merged)})
        logger.info(f"  {pair}: Spearman r={rho:.3f} (p={pval:.2e}, n={len(merged)})")

        ax = axes_flat[idx]
        ax.scatter(merged["lfc"], merged["cav_r"], s=4, alpha=0.4, color="#2277bb")
        ax.axhline(0, color="#aaaaaa", lw=0.5)
        ax.axvline(0, color="#aaaaaa", lw=0.5)
        ax.set_xlabel("log2FC (DE)", fontsize=8)
        ax.set_ylabel("Pearson r (CAV)", fontsize=8)
        ax.set_title(f"{pair.replace('__', ' | ')}\nSpearman r={rho:.2f}",
                     fontsize=7)

    # Hide unused subplots
    for ax in axes_flat[len(shared):]:
        ax.set_visible(False)

    plt.suptitle("DE log2FC vs. CAV gene correlation (top genes by |log2FC|)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison scatter: {out_path}")

    summary = pd.DataFrame(records).sort_values("spearman_r", ascending=False)
    summary_path = Path(out_path).parent / "de_vs_cav_spearman.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Saved Spearman summary: {summary_path}")
    print("\nDE vs. CAV Spearman correlations:")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pseudobulk DE for each tissue×cell_type pair, "
                    "comparable to cav_gene_correlation.py output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h5ad",     required=True)
    parser.add_argument("--lib-dir",  required=True,
                        help="CAV library dir containing library_structure.json")
    parser.add_argument("--group-col",     default="cell_type")
    parser.add_argument("--context-col",   default="tissue")
    parser.add_argument("--condition-col", default="disease")
    parser.add_argument("--donor-col",     default="",
                        help="obs column for donor/sample ID (required for DESeq2).")
    parser.add_argument("--baseline-value", default="normal",
                        help="Value in condition-col that is baseline (default: normal)")
    parser.add_argument("--method",
                        choices=["deseq2", "wilcoxon"], default="wilcoxon",
                        help="DE method: deseq2 (pseudobulk, needs --donor-col) "
                             "or wilcoxon (per-cell, default).")
    parser.add_argument("--out-dir",  required=True)
    parser.add_argument("--gene-name-col", default="",
                        help="adata.var column for human-readable gene names.")
    parser.add_argument("--min-cells",  type=int, default=10)
    parser.add_argument("--min-donors", type=int, default=3,
                        help="Minimum donors per condition for DESeq2 (default: 3).")
    parser.add_argument("--top-n",      type=int, default=50,
                        help="Top N genes for summary heatmap (default: 50).")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Run only these baseline CAV names.")
    parser.add_argument("--cav-dir", default=None,
                        help="Directory of cav_gene_correlation.py output TSVs. "
                             "If provided, computes Spearman r between log2FC and CAV r.")
    parser.add_argument("--heatmap-width",  type=float, default=14.0)
    parser.add_argument("--heatmap-height", type=float, default=10.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Library structure
    structure = load_library_structure(Path(args.lib_dir))
    if structure is None:
        logger.error("No library_structure.json — run analyze_cav_library.py first.")
        sys.exit(1)

    all_pairs   = structure.get("pairs", [])
    sep         = structure.get("separator", "__")
    group_lvl   = structure.get("group_level", 0)
    cond_lvl    = structure.get("condition_level", 2)
    baseline_vals = set(structure.get("baseline_values", ["normal"]))

    if args.pairs:
        filter_set = set(args.pairs)
        all_pairs  = [p for p in all_pairs if p["baseline"] in filter_set]
    logger.info(f"Processing {len(all_pairs)} pairs via {args.method}")

    # Load h5ad
    adata, gene_name_map = load_adata(args.h5ad, args.gene_name_col)
    adata.obs.index = adata.obs.index.astype(str)

    # Normalise obs index for matching
    lfc_collection: Dict[str, pd.Series] = {}

    for pair in all_pairs:
        bname = pair["baseline"]
        cname = pair["condition"]

        g_val, ctx_val, b_val = parse_pair_parts(bname, sep, group_lvl, cond_lvl)
        _,     _,       c_val = parse_pair_parts(cname, sep, group_lvl, cond_lvl)

        pair_label = f"{g_val}__{ctx_val}__{b_val}_vs_{c_val}"

        base_mask = cells_matching(adata.obs, args.group_col, args.context_col,
                                   args.condition_col, g_val, ctx_val, b_val)
        cond_mask = cells_matching(adata.obs, args.group_col, args.context_col,
                                   args.condition_col, g_val, ctx_val, c_val)
        pair_mask = base_mask | cond_mask

        n_base = base_mask.sum()
        n_cond = cond_mask.sum()
        n_pair = pair_mask.sum()

        if n_pair < args.min_cells:
            logger.warning(f"  {pair_label}: only {n_pair} cells — skipping")
            continue

        logger.info(f"  {g_val} | {ctx_val} | {b_val} ({n_base}) vs {c_val} ({n_cond})")

        adata_pair = adata[pair_mask].copy()

        try:
            if args.method == "deseq2":
                if not args.donor_col:
                    logger.error("--donor-col required for DESeq2")
                    sys.exit(1)
                result_df = run_deseq2_pair(
                    adata_pair, args.donor_col, args.condition_col,
                    args.baseline_value, min_donors=args.min_donors,
                )
            else:
                result_df = run_wilcoxon_pair(
                    adata_pair, args.condition_col, args.baseline_value,
                )
        except Exception as e:
            logger.warning(f"  {pair_label}: DE failed — {e}")
            continue

        if result_df is None or result_df.empty:
            logger.warning(f"  {pair_label}: no results")
            continue

        # Sort by |log2fc| descending
        result_df = (result_df
                     .assign(abs_lfc=result_df["log2fc"].abs())
                     .sort_values("abs_lfc", ascending=False)
                     .drop(columns="abs_lfc")
                     .reset_index(drop=True))

        if gene_name_map:
            result_df.insert(1, "gene_name",
                             result_df["gene"].map(gene_name_map).fillna(""))

        tsv_path = out_dir / f"{pair_label}.tsv"
        result_df.to_csv(tsv_path, sep="\t", index=False)
        top = result_df.iloc[0]
        logger.info(f"    → {tsv_path.name}  "
                    f"top: {top.get('gene_name', top['gene'])}  "
                    f"log2fc={top['log2fc']:.3f}")

        lfc_series = result_df.set_index("gene")["log2fc"]
        lfc_collection[pair_label] = lfc_series

    # Summary heatmap
    if lfc_collection:
        lfc_matrix = pd.DataFrame(lfc_collection).fillna(0)
        make_summary_heatmap(
            lfc_matrix, str(out_dir / "summary_heatmap.png"),
            top_n=args.top_n,
            figsize_w=args.heatmap_width, figsize_h=args.heatmap_height,
        )
        top_n_genes = lfc_matrix.abs().max(axis=1).nlargest(args.top_n).index
        lfc_matrix.loc[top_n_genes].to_csv(
            out_dir / "summary_top_genes.tsv", sep="\t")
        logger.info("Saved summary files.")
    else:
        logger.warning("No pairs produced results.")

    # CAV comparison
    if args.cav_dir:
        compare_with_cav(
            de_dir=out_dir,
            cav_dir=Path(args.cav_dir),
            out_path=str(out_dir / "de_vs_cav_scatter.png"),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
