#!/usr/bin/env python3
"""
cav_viz_pattern.py — Visualize CAV directions matched by a glob pattern.

Like cav_viz.py but instead of a library directory with the cavs/CONCEPT/concept_v1.npy
layout, this script accepts a shell glob pattern that directly matches .npy files,
e.g. ``cavs/PFAM/*v1.npy``.

The concept name for each file is the file stem with any trailing ``_v1``-style
version suffix stripped.

Two figures are available:

  1. Direction map  (--plot direction-map)
     All matched CAV direction vectors projected to 2D via PCA of the direction
     matrix.  Each point is one concept; points are labelled with the concept name.

  2. CAV-space UMAP  (--plot cav-umap)
     Cells embedded in CAV coordinate space (from cav_hierarchy.py
     cell_coordinates.tsv) and reduced to 2D with UMAP/t-SNE/PCA.

Usage
-----
# Direction map from a PFAM CAV glob:
python specific_scripts/cav_viz_pattern.py \\
    --cav-pattern "cavs/PFAM/*v1.npy" \\
    --plot        direction-map \\
    --out         results/figures/pfam_direction_map.png

# CAV-space UMAP (needs cell_coordinates.tsv from cav_hierarchy.py):
python specific_scripts/cav_viz_pattern.py \\
    --cav-pattern "cavs/PFAM/*v1.npy" \\
    --coords      results/hierarchy/cell_coordinates.tsv \\
    --obs         data/cells.h5ad \\
    --plot        cav-umap \\
    --out         results/figures/pfam_cav_umap.png
"""

import sys
import re
import argparse
import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CELL_TYPE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
]

_VERSION_SUFFIX = re.compile(r'_v\d+$')


def _concept_name(path: Path) -> str:
    """
    Derive a concept name from a .npy file path.
    Uses the parent directory name when the file lives in a per-concept subdirectory
    (e.g. cavs/PF01112/L25_concept_v1.npy → 'PF01112'), otherwise falls back to
    the file stem with any trailing _v1-style suffix stripped.
    """
    parent = path.parent.name
    stem   = _VERSION_SUFFIX.sub('', path.stem)
    # If the stem contains 'concept', the directory name is the real concept identity
    if 'concept' in stem:
        return parent
    return stem


def load_directions_from_pattern(pattern: str) -> Dict[str, np.ndarray]:
    """
    Load and unit-normalise CAV direction vectors from all files matching `pattern`.
    Keys are concept names derived from file stems.
    """
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    result = {}
    for p in paths:
        path = Path(p)
        name = _concept_name(path)
        v = np.load(path).astype(np.float64)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            result[name] = v / norm
        else:
            logger.warning(f"Skipping zero-norm vector: {path}")
    logger.info(f"Loaded {len(result)} CAV directions from pattern '{pattern}'")
    return result


def load_obs(h5ad_path: str, columns: List[str]) -> pd.DataFrame:
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs[columns].copy()
    adata.file.close()
    return obs


# ---------------------------------------------------------------------------
# Figure 1: Direction map
# ---------------------------------------------------------------------------

def plot_direction_map(
    cav_pattern: str,
    out_path: str,
    figsize: Tuple[int, int] = (10, 9),
    label_points: bool = True,
):
    """Project all matched CAV direction vectors to 2D via PCA and plot."""
    cav_dirs = load_directions_from_pattern(cav_pattern)
    if not cav_dirs:
        raise ValueError("No valid CAV directions loaded.")

    names  = list(cav_dirs.keys())
    matrix = np.vstack([cav_dirs[n] for n in names])

    pca2d  = PCA(n_components=2, random_state=42)
    coords = pca2d.fit_transform(matrix)
    coord_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=names)

    # Color by a simple prefix group (e.g. PF00001 → PF00001, or first token before "_")
    def _group(name: str) -> str:
        return name.split("_")[0] if "_" in name else name

    groups    = sorted({_group(n) for n in names})
    color_map = {g: CELL_TYPE_COLORS[i % len(CELL_TYPE_COLORS)]
                 for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=figsize)

    for name in names:
        x, y  = coord_df.loc[name, ["PC1", "PC2"]]
        group = _group(name)
        ax.scatter(x, y, c=color_map[group], s=70, marker="o",
                   alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)

    if label_points:
        for name in names:
            x, y = coord_df.loc[name, ["PC1", "PC2"]]
            ax.annotate(name, (x, y), fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points", color="dimgray")

    if len(groups) > 1:
        handles = [mpatches.Patch(color=color_map[g], label=g.replace("_", " "))
                   for g in groups]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=8, frameon=False)

    var1 = pca2d.explained_variance_ratio_[0] * 100
    var2 = pca2d.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=11)
    ax.set_title(f"CAV direction space\n{cav_pattern}", fontsize=12)
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    ax.set_aspect("equal")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved direction map: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: CAV-space UMAP
# ---------------------------------------------------------------------------

def _make_reducer(reducer: str, n_components: int = 2):
    if reducer == "umap":
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "umap-learn is not installed. Use --reducer tsne or --reducer pca, "
                "or install with: pip install umap-learn"
            )
        return UMAP(n_components=n_components, random_state=42,
                    n_neighbors=30, min_dist=0.1)
    elif reducer == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=n_components, random_state=42,
                    perplexity=30, max_iter=1000, init="pca", learning_rate="auto")
    elif reducer == "pca":
        return PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown reducer '{reducer}'. Choose: umap, tsne, pca")


def plot_cav_umap(
    cav_pattern: str,
    coords_path: str,
    out_path: str,
    obs_path: Optional[str] = None,
    color_col: str = "tissue",
    condition_col: Optional[str] = None,
    baseline_value: str = "normal",
    reducer: str = "umap",
    max_cells: int = 20000,
    figsize: Tuple[int, int] = (9, 8),
):
    """
    Project cells into the space of matched CAVs and plot a 2-D embedding.
    The feature matrix is cells × matched-CAV-scores derived from cell_coordinates.tsv,
    filtered to the columns that match the loaded CAV names.
    """
    cav_dirs = load_directions_from_pattern(cav_pattern)
    cav_names = set(cav_dirs.keys())

    logger.info(f"Loading coordinates: {coords_path}")
    coords = pd.read_csv(coords_path, sep="\t", index_col=0)

    # Keep only columns whose suffix matches a loaded CAV name
    feature_cols = [c for c in coords.columns
                    if any(c.endswith(f"__{name}") or c == name for name in cav_names)]
    if not feature_cols:
        # Fall back: any column whose last token matches a CAV name
        feature_cols = [c for c in coords.columns
                        if c.split("__")[-1] in cav_names]
    if not feature_cols:
        raise ValueError(
            f"No coordinate columns matched any of the {len(cav_names)} loaded CAV names. "
            f"Sample columns: {list(coords.columns[:5])}"
        )
    logger.info(f"Using {len(feature_cols)} matched CAV axes as features")

    rng = np.random.default_rng(42)
    n = min(max_cells, len(coords))
    idx = rng.choice(len(coords), n, replace=False)
    X_feat = coords[feature_cols].iloc[idx].values

    logger.info(f"Running {reducer.upper()} on {n} cells × {len(feature_cols)} axes...")
    emb = _make_reducer(reducer).fit_transform(X_feat)

    cat_colors_arr = np.array(["steelblue"] * n, dtype=object)
    handles_color: list = []

    def _safe_labels(series) -> np.ndarray:
        return series.astype(str).replace("nan", "unknown").values

    if obs_path:
        cols = list(dict.fromkeys(c for c in [color_col, condition_col] if c))
        try:
            obs = load_obs(obs_path, cols)
            aligned = obs.reindex(coords.index.astype(str)).iloc[idx]
            if color_col and color_col in aligned.columns:
                labels  = _safe_labels(aligned[color_col])
                unique  = sorted(set(labels))
                palette = plt.colormaps.get_cmap("tab20")
                lc_map  = {l: palette(i / max(len(unique), 1)) for i, l in enumerate(unique)}
                cat_colors_arr = np.array([lc_map[l] for l in labels])
                handles_color  = [
                    mpatches.Patch(color=lc_map[l], label=l.replace("_", " "))
                    for l in unique
                ]
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")

    xl, yl = {"umap": ("UMAP 1", "UMAP 2"),
               "tsne": ("t-SNE 1", "t-SNE 2"),
               "pca":  ("PC 1", "PC 2")}.get(reducer, ("Dim 1", "Dim 2"))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(emb[:, 0], emb[:, 1], c=list(cat_colors_arr),
               s=6, alpha=0.5, linewidths=0)
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel(yl, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"CAV-space {reducer.upper()}\n{cav_pattern}", fontsize=12
    )
    if handles_color:
        ax.legend(handles=handles_color, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=7, frameon=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {reducer.upper()}: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize CAV directions matched by a glob pattern.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cav-pattern", required=True,
                        help="Glob pattern matching CAV .npy files, "
                             "e.g. 'cavs/PFAM/*v1.npy'.")
    parser.add_argument("--plot", required=True,
                        choices=["direction-map", "cav-umap"],
                        help="Which figure to produce.")
    parser.add_argument("--out", required=True,
                        help="Output file path.")
    parser.add_argument("--coords",
                        help="cell_coordinates.tsv from cav_hierarchy.py (for cav-umap).")
    parser.add_argument("--obs",
                        help="h5ad file for cell metadata (obs columns).")
    parser.add_argument("--color-col", default="tissue",
                        help="obs column to color embedding by (default: tissue).")
    parser.add_argument("--condition-col", default=None,
                        help="obs column for binary normal/condition coloring.")
    parser.add_argument("--baseline-value", default="normal",
                        help="Value identifying baseline cells in --condition-col.")
    parser.add_argument("--reducer", default="umap",
                        choices=["umap", "tsne", "pca"],
                        help="Dimensionality reduction for cav-umap (default: umap).")
    parser.add_argument("--max-cells", type=int, default=20000,
                        help="Subsample cap for cav-umap (default: 20000).")
    parser.add_argument("--no-labels", action="store_true",
                        help="Suppress concept name labels on direction-map.")
    args = parser.parse_args()

    if args.plot == "direction-map":
        plot_direction_map(
            cav_pattern=args.cav_pattern,
            out_path=args.out,
            label_points=not args.no_labels,
        )

    elif args.plot == "cav-umap":
        if not args.coords:
            parser.error("--coords required for cav-umap")
        plot_cav_umap(
            cav_pattern=args.cav_pattern,
            coords_path=args.coords,
            out_path=args.out,
            obs_path=args.obs,
            color_col=args.color_col,
            condition_col=args.condition_col,
            baseline_value=args.baseline_value,
            reducer=args.reducer,
            max_cells=args.max_cells,
        )


if __name__ == "__main__":
    main()
