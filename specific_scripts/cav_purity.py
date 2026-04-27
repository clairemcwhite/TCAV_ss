#!/usr/bin/env python3
"""
cav_purity.py — Identify potentially mislabeled cells using L0 CAV cross-projections.

Every cell already has projections onto ALL L0 (cell-type) axes in
cell_coordinates.tsv (from cav_hierarchy.py).  Since all CAVs share the same
PCA background, scores are directly comparable across cell types.

A cell is "pure" when its annotated cell type's L0 score is the highest among
all L0 axes.  A "confused" cell scores higher on a different cell type's L0
axis, suggesting contamination or mislabeling.

Purity score (per cell)
-----------------------
    purity = L0_score(annotated_type) - L0_score(best_other_type)

    > 0  → cell looks more like its label than anything else  (pure)
    = 0  → tie
    < 0  → cell looks MORE like a different type              (confused)

Usage
-----
    python $GIT/specific_scripts/cav_purity.py \\
        --coords    results/hierarchy/cell_coordinates.tsv \\
        --h5ad      data/tumor_cells.h5ad \\
        --out       results/purity/ \\
        --cell-type-col cell_type \\
        --tissue-col    tissue \\
        --condition-col condition

Run on normal cells only (more trusted annotations):
    ... --normal-only

Outputs
-------
    purity_scores.tsv         per-cell: annotated type, best L0 type, purity score
    purity_summary.tsv        per (cell_type, tissue, condition): confusion rate
    confusion_matrix.pdf      heatmap — annotated type vs best L0 type
    purity_by_condition.pdf   violin plots per cell type, split by condition
    confused_markers.pdf      top DE genes in confused vs pure cells of same type
"""

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import seaborn as sns

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def norm_label(s: str) -> str:
    """Normalise a cell-type label for matching: lowercase, spaces→underscores."""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def load_coords(coords_path: Path) -> pd.DataFrame:
    df = pd.read_csv(coords_path, sep="\t")
    df["cell_id"] = df["cell_id"].astype(str)
    logger.info(f"Loaded cell_coordinates: {df.shape[0]} cells, {df.shape[1]} columns")
    return df


def extract_l0_cols(coords_df: pd.DataFrame) -> dict:
    """Return {norm_label: column_name} for all L0__ columns."""
    l0_cols = [c for c in coords_df.columns if c.startswith("L0__")]
    if not l0_cols:
        raise ValueError("No L0__ columns found in cell_coordinates.tsv. "
                         "Re-run cav_hierarchy.py first.")
    mapping = {norm_label(c.replace("L0__", "")): c for c in l0_cols}
    logger.info(f"Found {len(l0_cols)} L0 axes: {sorted(mapping)}")
    return mapping


def load_metadata(h5ad_path: Path,
                  cell_type_col: str,
                  tissue_col: str,
                  condition_col: str) -> pd.DataFrame:
    """Load obs metadata from h5ad; return DataFrame indexed by cell_id."""
    import anndata as ad
    logger.info(f"Loading h5ad metadata from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs[[cell_type_col, tissue_col, condition_col]].copy()
    obs.index = obs.index.astype(str)
    obs = obs.rename(columns={
        cell_type_col: "annotated_type",
        tissue_col:    "tissue",
        condition_col: "condition",
    })
    return obs


# ============================================================
# Core purity computation
# ============================================================

def compute_purity(coords_df: pd.DataFrame,
                   meta_df: pd.DataFrame,
                   l0_map: dict,
                   normal_only: bool = False,
                   baseline_values: set = None) -> pd.DataFrame:
    """
    Merge coordinates with metadata and compute per-cell purity scores.

    Returns a DataFrame with columns:
        cell_id, annotated_type, tissue, condition,
        annotated_type_norm,           — normalised label for L0 matching
        l0_annotated,                  — L0 score on own cell type
        best_l0_type,                  — cell type with highest L0 score
        best_l0_score,
        purity,                        — l0_annotated - best_other_score
        is_confused,                   — bool: best type != annotated type
        annotated_in_library,          — bool: annotated type has an L0 axis
        <L0__{type}> cols              — all raw L0 scores
    """
    if baseline_values is None:
        baseline_values = {"normal"}

    # Merge
    df = coords_df.set_index("cell_id").join(meta_df, how="inner")
    n_lost = len(coords_df) - len(df)
    if n_lost:
        logger.warning(f"{n_lost} cells dropped (not in h5ad metadata)")
    logger.info(f"Working with {len(df)} cells after merge")

    if normal_only:
        mask = df["condition"].apply(lambda v: norm_label(str(v)) in
                                     {norm_label(b) for b in baseline_values})
        df = df[mask]
        logger.info(f"--normal-only: keeping {len(df)} normal cells")

    # All L0 columns present in coords
    l0_cols_present = [c for c in df.columns if c.startswith("L0__")]
    l0_matrix = df[l0_cols_present].values  # (n_cells, n_types)
    l0_labels = [c.replace("L0__", "") for c in l0_cols_present]  # original casing

    # Normalised label → index in l0_labels
    norm_to_idx = {norm_label(lbl): i for i, lbl in enumerate(l0_labels)}

    # For each cell: find index of its annotated type in L0
    ann_norms = df["annotated_type"].apply(norm_label).values
    ann_in_lib = np.array([n in norm_to_idx for n in ann_norms])

    l0_annotated = np.full(len(df), np.nan)
    for i, (n, in_lib) in enumerate(zip(ann_norms, ann_in_lib)):
        if in_lib:
            l0_annotated[i] = l0_matrix[i, norm_to_idx[n]]

    # Best L0 overall
    best_idx   = np.argmax(l0_matrix, axis=1)
    best_score = l0_matrix[np.arange(len(df)), best_idx]
    best_type  = np.array([l0_labels[i] for i in best_idx])

    # Second-best for purity when annotated == best
    # purity = annotated_score - best_other_score
    purity = np.full(len(df), np.nan)
    for i in range(len(df)):
        if not ann_in_lib[i]:
            continue
        ann_idx   = norm_to_idx[ann_norms[i]]
        ann_score = l0_matrix[i, ann_idx]
        # best score among OTHER types
        others = np.delete(l0_matrix[i], ann_idx)
        best_other = others.max() if len(others) > 0 else 0.0
        purity[i] = ann_score - best_other

    is_confused = np.array([
        ann_in_lib[i] and (norm_label(best_type[i]) != ann_norms[i])
        for i in range(len(df))
    ])

    result = df[["annotated_type", "tissue", "condition"] + l0_cols_present].copy()
    result.index.name = "cell_id"
    result = result.reset_index()
    result["annotated_type_norm"] = ann_norms
    result["annotated_in_library"] = ann_in_lib
    result["l0_annotated"]         = l0_annotated
    result["best_l0_type"]         = best_type
    result["best_l0_score"]        = best_score
    result["purity"]               = purity
    result["is_confused"]          = is_confused

    n_lib   = ann_in_lib.sum()
    n_conf  = is_confused.sum()
    logger.info(f"Cells with annotated type in L0 library: {n_lib} / {len(result)}")
    logger.info(f"Confused cells (best L0 ≠ annotation): {n_conf} / {n_lib} "
                f"({100*n_conf/max(n_lib,1):.1f}%)")
    return result


def build_summary(purity_df: pd.DataFrame) -> pd.DataFrame:
    """Per (annotated_type, tissue, condition) confusion statistics."""
    rows = []
    grp_cols = ["annotated_type", "tissue", "condition"]
    for key, grp in purity_df[purity_df["annotated_in_library"]].groupby(grp_cols):
        ann_type, tissue, cond = key
        n_cells    = len(grp)
        n_confused = grp["is_confused"].sum()
        conf_rate  = n_confused / n_cells if n_cells > 0 else np.nan
        top_target = (grp[grp["is_confused"]]["best_l0_type"]
                      .value_counts().idxmax()
                      if n_confused > 0 else "")
        med_purity = grp["purity"].median()
        rows.append({
            "annotated_type": ann_type,
            "tissue":         tissue,
            "condition":      cond,
            "n_cells":        n_cells,
            "n_confused":     n_confused,
            "confusion_rate": round(conf_rate, 4),
            "top_confusion_target": top_target,
            "median_purity":  round(med_purity, 4),
        })
    summary = pd.DataFrame(rows).sort_values("confusion_rate", ascending=False)
    return summary


# ============================================================
# Plots
# ============================================================

def plot_confusion_matrix(purity_df: pd.DataFrame,
                          out_path: Path,
                          min_cells: int = 20) -> None:
    """
    Heatmap: annotated type (rows) × best L0 type (cols).
    Values = fraction of cells in that row assigned to that column.
    Only shows cell types with >= min_cells cells.
    """
    sub = purity_df[purity_df["annotated_in_library"]].copy()
    # filter to types with enough cells
    counts_per_type = sub["annotated_type"].value_counts()
    keep = counts_per_type[counts_per_type >= min_cells].index
    sub = sub[sub["annotated_type"].isin(keep)]

    if sub.empty:
        logger.warning("confusion_matrix: no cell types with enough cells — skipping")
        return

    ct = pd.crosstab(sub["annotated_type"], sub["best_l0_type"], normalize="index")
    # Sort so diagonal is as large as possible
    types = sorted(ct.index)
    ct = ct.reindex(index=types)
    for col in types:
        if col not in ct.columns:
            ct[col] = 0.0
    ct = ct[types]

    fig, ax = plt.subplots(figsize=(max(6, len(types) * 0.8),
                                    max(5, len(types) * 0.7)))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues",
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Fraction of cells"})
    ax.set_title("CAV purity: fraction of cells by best L0 match", fontsize=12)
    ax.set_xlabel("Best L0 cell type")
    ax.set_ylabel("Annotated cell type")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix: {out_path}")


def plot_purity_by_condition(purity_df: pd.DataFrame,
                             out_path: Path,
                             min_cells: int = 20) -> None:
    """
    One panel per cell type; violin of purity scores split by condition.
    Confused cells (purity < 0) are highlighted with a dashed zero line.
    """
    sub = purity_df[purity_df["annotated_in_library"] & purity_df["purity"].notna()].copy()
    cell_types = [ct for ct, cnt in sub["annotated_type"].value_counts().items()
                  if cnt >= min_cells]
    if not cell_types:
        logger.warning("purity_by_condition: no cell types with enough cells — skipping")
        return

    conditions = sorted(sub["condition"].unique())
    palette    = sns.color_palette("Set2", len(conditions))
    cond_color = dict(zip(conditions, palette))

    ncols = min(4, len(cell_types))
    nrows = (len(cell_types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.5),
                             sharey=False)
    axes = np.array(axes).flatten()

    for ax, ct in zip(axes, cell_types):
        grp = sub[sub["annotated_type"] == ct]
        data_by_cond = [grp[grp["condition"] == c]["purity"].values
                        for c in conditions]
        parts = ax.violinplot(data_by_cond,
                              positions=range(len(conditions)),
                              showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(palette[i])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="purity=0 (confused)")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
        ax.set_title(ct, fontsize=9)
        ax.set_ylabel("Purity score", fontsize=8)

        # Annotate confusion rate per condition
        for i, c in enumerate(conditions):
            cv = grp[grp["condition"] == c]
            rate = (cv["is_confused"].sum() / len(cv) * 100) if len(cv) else 0
            ax.text(i, ax.get_ylim()[0], f"{rate:.0f}%",
                    ha="center", va="bottom", fontsize=7, color="darkred")

    for ax in axes[len(cell_types):]:
        ax.set_visible(False)

    fig.suptitle("Cell purity scores by condition\n(red % = confused cells)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved purity-by-condition plot: {out_path}")


def plot_confused_markers(purity_df: pd.DataFrame,
                          h5ad_path: Path,
                          out_path: Path,
                          purity_threshold: float = 0.0,
                          top_n_genes: int = 15,
                          min_confused: int = 10,
                          gene_name_col: str = None) -> None:
    """
    For each (annotated_type → best_l0_type) confusion pair with enough cells:
      - Compare confused cells vs pure cells of the same annotated type
      - Show top genes upregulated in confused cells
      - These should be markers of the best_l0_type (the "true" type)

    Uses a simple Wilcoxon rank-sum test.
    """
    import anndata as ad
    from scipy import stats

    logger.info(f"Loading h5ad for marker analysis: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Gene name mapping
    gene_ids = list(adata.var_names)
    if gene_name_col and gene_name_col in adata.var.columns:
        gene_names = list(adata.var[gene_name_col])
    else:
        for candidate in ["gene_name", "gene_symbol", "feature_name",
                          "gene_short_name", "symbol"]:
            if candidate in adata.var.columns:
                gene_names = list(adata.var[candidate])
                logger.info(f"Using gene name column: {candidate}")
                break
        else:
            gene_names = gene_ids

    X = (adata.X.toarray() if hasattr(adata.X, "toarray")
         else np.asarray(adata.X)).astype(np.float32)
    cell_id_to_idx = {str(c): i for i, c in enumerate(adata.obs_names)}

    # Identify confusion pairs
    confused = purity_df[purity_df["purity"] < purity_threshold].copy()
    pure     = purity_df[purity_df["purity"] >= purity_threshold].copy()

    pairs = (confused.groupby(["annotated_type", "best_l0_type"])
             .size().reset_index(name="n_confused"))
    pairs = pairs[pairs["n_confused"] >= min_confused].sort_values(
        "n_confused", ascending=False)

    if pairs.empty:
        logger.info("No confusion pairs with enough cells — skipping marker plot")
        return

    panels = []
    for _, row in pairs.iterrows():
        ann_type  = row["annotated_type"]
        best_type = row["best_l0_type"]

        conf_ids = confused[
            (confused["annotated_type"] == ann_type) &
            (confused["best_l0_type"]   == best_type)
        ]["cell_id"].values
        pure_ids = pure[pure["annotated_type"] == ann_type]["cell_id"].values

        conf_idx = [cell_id_to_idx[c] for c in conf_ids if c in cell_id_to_idx]
        pure_idx = [cell_id_to_idx[c] for c in pure_ids if c in cell_id_to_idx]

        if len(conf_idx) < min_confused or len(pure_idx) < 5:
            continue

        X_conf = X[conf_idx]
        X_pure = X[pure_idx]

        # Wilcoxon for each gene
        n_genes = X.shape[1]
        stats_arr = np.zeros(n_genes)
        for g in range(n_genes):
            a = X_conf[:, g]
            b = X_pure[:, g]
            if a.max() == 0 and b.max() == 0:
                continue
            try:
                res = stats.mannwhitneyu(a, b, alternative="greater")
                stats_arr[g] = -np.log10(res.pvalue + 1e-300)
            except Exception:
                pass

        top_idx = np.argsort(stats_arr)[::-1][:top_n_genes]
        mean_conf = X_conf[:, top_idx].mean(axis=0)
        mean_pure = X_pure[:, top_idx].mean(axis=0)
        log2fc    = np.log2(mean_conf + 1) - np.log2(mean_pure + 1)
        top_names = [gene_names[i] for i in top_idx]

        panels.append({
            "ann_type":  ann_type,
            "best_type": best_type,
            "n_conf":    len(conf_idx),
            "n_pure":    len(pure_idx),
            "genes":     top_names,
            "log2fc":    log2fc,
            "neg_logp":  stats_arr[top_idx],
        })

    if not panels:
        logger.info("No panels to plot for confused markers")
        return

    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 5),
                             squeeze=False)
    axes = axes.flatten()

    for ax, panel in zip(axes, panels):
        genes   = panel["genes"]
        log2fc  = panel["log2fc"]
        neg_lp  = panel["neg_logp"]

        colors = ["#d62728" if v > 0 else "#1f77b4" for v in log2fc]
        y_pos  = range(len(genes))
        ax.barh(y_pos, log2fc, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("log2FC (confused vs pure)", fontsize=8)
        ax.set_title(
            f"{panel['ann_type']} → {panel['best_type']}\n"
            f"n_confused={panel['n_conf']}, n_pure={panel['n_pure']}",
            fontsize=9,
        )

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Top upregulated genes in confused cells (purity < {purity_threshold})\n"
        "Red = up in confused, Blue = down in confused",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confused-cell marker plot: {out_path}")


def plot_confusion_by_condition(purity_df: pd.DataFrame,
                                out_path: Path,
                                baseline_values: set = None,
                                min_cells: int = 10) -> None:
    """
    Two side-by-side confusion matrices: one for normal cells, one for cancer cells.

    Rows  = annotated cell type (ground-truth label)
    Cols  = best L0 category   (CAV prediction)
    Values = raw cell counts, with row-fraction in parentheses.

    Diagonal = correctly classified.  Off-diagonal = confused cells.
    Color scale is row-normalised so rare cell types are still readable.
    """
    if baseline_values is None:
        baseline_values = {"normal"}

    sub = purity_df[purity_df["annotated_in_library"]].copy()
    if sub.empty:
        logger.warning("confusion_by_condition: no cells with annotated type in library")
        return

    # Split into normal / disease groups
    norm_mask    = sub["condition"].apply(
        lambda v: norm_label(str(v)) in {norm_label(b) for b in baseline_values})
    normal_sub   = sub[norm_mask]
    disease_sub  = sub[~norm_mask]

    # Collect all cell types present across both splits
    all_types = sorted(set(sub["annotated_type"].unique()) |
                       set(sub["best_l0_type"].unique()))

    def make_count_matrix(df):
        if df.empty:
            return pd.DataFrame(0, index=all_types, columns=all_types)
        ct = pd.crosstab(df["annotated_type"], df["best_l0_type"])
        ct = ct.reindex(index=all_types, columns=all_types, fill_value=0)
        return ct

    count_norm    = make_count_matrix(normal_sub)
    count_disease = make_count_matrix(disease_sub)

    # Filter to rows/cols that have at least min_cells across either split
    row_totals = count_norm.sum(axis=1) + count_disease.sum(axis=1)
    keep_types = row_totals[row_totals >= min_cells].index
    if len(keep_types) == 0:
        logger.warning("confusion_by_condition: no types pass min_cells threshold")
        return
    count_norm    = count_norm.loc[keep_types, keep_types]
    count_disease = count_disease.loc[keep_types, keep_types]

    # Row-normalised fractions for colour; raw counts for annotation
    def row_frac(ct):
        totals = ct.sum(axis=1).replace(0, np.nan)
        return ct.div(totals, axis=0).fillna(0)

    frac_norm    = row_frac(count_norm)
    frac_disease = row_frac(count_disease)

    # Annotation: "count\n(frac%)"
    def annot_matrix(counts, fracs):
        arr = np.empty(counts.shape, dtype=object)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                n = int(counts.iloc[i, j])
                f = fracs.iloc[i, j]
                arr[i, j] = f"{n}\n({f:.0%})" if n > 0 else ""
        return arr

    ann_norm    = annot_matrix(count_norm,    frac_norm)
    ann_disease = annot_matrix(count_disease, frac_disease)

    n = len(keep_types)
    cell_size = max(0.65, min(1.2, 8 / n))
    fig_w = n * cell_size * 2 + 3
    fig_h = n * cell_size + 1.5

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    cond_labels = list(baseline_values)
    disease_label = "cancer / disease"

    for ax, frac, ann, title in [
        (axes[0], frac_norm,    ann_norm,    f"Normal ({len(normal_sub):,} cells)"),
        (axes[1], frac_disease, ann_disease, f"Cancer / disease ({len(disease_sub):,} cells)"),
    ]:
        sns.heatmap(
            frac, annot=ann, fmt="", cmap="Blues",
            vmin=0, vmax=1, ax=ax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Row fraction", "shrink": 0.6},
            annot_kws={"fontsize": max(5, min(9, 80 // n))},
        )
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel("Best L0 category (CAV prediction)", fontsize=8)
        ax.set_ylabel("Annotated cell type (label)", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)

        # Highlight diagonal
        for k in range(n):
            ax.add_patch(plt.Rectangle((k, k), 1, 1,
                                       fill=False, edgecolor="limegreen",
                                       lw=1.5, clip_on=False))

    fig.suptitle("CAV purity confusion matrices\n"
                 "Rows = annotated label  |  Cols = best L0 match  |  "
                 "Green border = diagonal (correct)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved stratified confusion matrices: {out_path}")


def plot_purity_scatter(purity_df: pd.DataFrame,
                        out_path: Path,
                        cell_type: str,
                        tissue: str = None,
                        condition_col: str = "condition",
                        baseline_values: set = None) -> None:
    """
    Scatter: L0_annotated (x) vs L0_best_other (y), colored by condition.
    One dot per cell; diagonal = purity boundary.
    """
    if baseline_values is None:
        baseline_values = {"normal"}

    sub = purity_df[
        purity_df["annotated_in_library"] &
        (purity_df["annotated_type"] == cell_type)
    ].copy()
    if tissue:
        sub = sub[sub["tissue"] == tissue]
    if sub.empty:
        return

    # best_other score = best_l0_score when best!=annotated, else second best
    l0_cols = [c for c in sub.columns if c.startswith("L0__")]
    ann_norm = norm_label(cell_type)
    ann_col  = next((c for c in l0_cols
                     if norm_label(c.replace("L0__", "")) == ann_norm), None)
    if ann_col is None:
        return

    other_cols = [c for c in l0_cols if c != ann_col]
    sub["l0_best_other"] = sub[other_cols].max(axis=1)
    sub["l0_ann"]        = sub[ann_col]

    conditions = sorted(sub["condition"].unique())
    palette    = sns.color_palette("Set2", len(conditions))
    cond_color = dict(zip(conditions, palette))

    fig, ax = plt.subplots(figsize=(5, 5))
    for cond in conditions:
        cv = sub[sub["condition"] == cond]
        ax.scatter(cv["l0_ann"], cv["l0_best_other"],
                   c=[cond_color[cond]], alpha=0.3, s=10, label=cond)

    lims = [
        min(sub["l0_ann"].min(), sub["l0_best_other"].min()) - 0.05,
        max(sub["l0_ann"].max(), sub["l0_best_other"].max()) + 0.05,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"L0 score — {cell_type} (annotated)", fontsize=9)
    ax.set_ylabel("L0 score — best other type", fontsize=9)
    title = f"CAV purity: {cell_type}"
    if tissue:
        title += f" / {tissue}"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved purity scatter: {out_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Identify mislabeled cells via L0 CAV cross-projections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--coords", required=True,
                        help="cell_coordinates.tsv from cav_hierarchy.py")
    parser.add_argument("--h5ad", required=True,
                        help="AnnData h5ad with cell metadata and expression")
    parser.add_argument("--out", required=True,
                        help="Output directory")
    parser.add_argument("--cell-type-col", default="cell_type",
                        help="obs column for annotated cell type (default: cell_type)")
    parser.add_argument("--tissue-col", default="tissue",
                        help="obs column for tissue (default: tissue)")
    parser.add_argument("--condition-col", default="condition",
                        help="obs column for condition (default: condition)")
    parser.add_argument("--baseline-values", nargs="+", default=["normal"],
                        help="Condition values treated as baseline / normal "
                             "(default: normal)")
    parser.add_argument("--normal-only", action="store_true",
                        help="Restrict analysis to normal-condition cells only "
                             "(useful for establishing a clean reference purity)")
    parser.add_argument("--purity-threshold", type=float, default=0.0,
                        help="Purity score below this → cell is 'confused' "
                             "(default: 0.0 = best other type beats annotated type)")
    parser.add_argument("--min-cells", type=int, default=20,
                        help="Min cells per group for plots (default: 20)")
    parser.add_argument("--min-confused", type=int, default=10,
                        help="Min confused cells for marker DE analysis (default: 10)")
    parser.add_argument("--top-n-genes", type=int, default=15,
                        help="Top DE genes to show per confusion pair (default: 15)")
    parser.add_argument("--gene-name-col", default=None,
                        help="adata.var column for gene symbols (auto-detected if omitted)")
    parser.add_argument("--scatter", nargs="+", default=None,
                        metavar="cell_type[/tissue]",
                        help="Generate purity scatter plots for specified cell "
                             "types (optionally slash-separated with tissue, e.g. "
                             "T_cell/lung B_cell)")
    parser.add_argument("--no-markers", action="store_true",
                        help="Skip the marker DE analysis (faster)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_values = set(args.baseline_values)

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    coords_df = load_coords(Path(args.coords))
    l0_map    = extract_l0_cols(coords_df)
    meta_df   = load_metadata(Path(args.h5ad),
                              args.cell_type_col,
                              args.tissue_col,
                              args.condition_col)

    # ------------------------------------------------------------------ #
    # 2. Compute purity
    # ------------------------------------------------------------------ #
    purity_df = compute_purity(
        coords_df, meta_df, l0_map,
        normal_only=args.normal_only,
        baseline_values=baseline_values,
    )

    scores_path = out_dir / "purity_scores.tsv"
    purity_df.to_csv(scores_path, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Saved purity scores: {scores_path}")

    # ------------------------------------------------------------------ #
    # 3. Summary table
    # ------------------------------------------------------------------ #
    summary_df = build_summary(purity_df)
    summary_path = out_dir / "purity_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Saved purity summary: {summary_path}")

    print("\nTop confused cell types (by confusion rate):")
    print(summary_df.head(15).to_string(index=False))

    # ------------------------------------------------------------------ #
    # 4. Plots
    # ------------------------------------------------------------------ #
    plot_confusion_matrix(
        purity_df,
        out_dir / "confusion_matrix.pdf",
        min_cells=args.min_cells,
    )

    plot_confusion_by_condition(
        purity_df,
        out_dir / "confusion_matrix_by_condition.pdf",
        baseline_values=baseline_values,
        min_cells=args.min_cells,
    )

    if not args.normal_only:
        plot_purity_by_condition(
            purity_df,
            out_dir / "purity_by_condition.pdf",
            min_cells=args.min_cells,
        )

    if not args.no_markers:
        plot_confused_markers(
            purity_df,
            Path(args.h5ad),
            out_dir / "confused_markers.pdf",
            purity_threshold=args.purity_threshold,
            top_n_genes=args.top_n_genes,
            min_confused=args.min_confused,
            gene_name_col=args.gene_name_col,
        )

    # Per-cell-type scatter plots
    if args.scatter:
        scatter_dir = out_dir / "scatter"
        scatter_dir.mkdir(exist_ok=True)
        for spec in args.scatter:
            parts = spec.split("/", 1)
            ct      = parts[0]
            tissue  = parts[1] if len(parts) > 1 else None
            fname   = f"{ct}{'__' + tissue if tissue else ''}_scatter.pdf"
            plot_purity_scatter(
                purity_df,
                scatter_dir / fname,
                cell_type=ct,
                tissue=tissue,
                baseline_values=baseline_values,
            )

    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
