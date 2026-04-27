#!/usr/bin/env python3
"""
cav_viz_pattern.py — Visualize CAV directions matched by a glob pattern.

Like cav_viz.py but instead of a library directory with the cavs/CONCEPT/concept_v1.npy
layout, this script accepts a shell glob pattern that directly matches .npy files,
e.g. ``cavs/PFAM/*/L25_concept_v1.npy``.

Each matched .npy file is one CAV; its concept name is taken from the parent
directory (e.g. cavs/PF01112/L25_concept_v1.npy → 'PF01112').

The CAV direction vectors themselves are embedded in 2D — no cell coordinates needed.

Usage
-----
python specific_scripts/cav_viz_pattern.py \\
    --cav-pattern "/path/to/cavs/*/L25_concept_v1.npy" \\
    --reducer     umap \\
    --out         results/figures/pfam_direction_map.png
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


# ---------------------------------------------------------------------------
# Direction map (CAV vectors embedded in 2D)
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


def plot_direction_map(
    cav_pattern: str,
    out_path: str,
    reducer: str = "pca",
    figsize: Tuple[int, int] = (10, 9),
    label_points: bool = True,
):
    """Embed all matched CAV direction vectors in 2D and plot."""
    cav_dirs = load_directions_from_pattern(cav_pattern)
    if not cav_dirs:
        raise ValueError("No valid CAV directions loaded.")

    names  = list(cav_dirs.keys())
    matrix = np.vstack([cav_dirs[n] for n in names])

    logger.info(f"Running {reducer.upper()} on {len(names)} CAV vectors...")
    red    = _make_reducer(reducer)
    coords = red.fit_transform(matrix)
    coord_df = pd.DataFrame(coords, columns=["D1", "D2"], index=names)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coord_df["D1"], coord_df["D2"], s=70, marker="o",
               color="steelblue", alpha=0.85, edgecolors="white",
               linewidths=0.5, zorder=3)

    if label_points:
        for name in names:
            x, y = coord_df.loc[name, ["D1", "D2"]]
            ax.annotate(name, (x, y), fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points", color="dimgray")

    xl, yl = {"umap": ("UMAP 1", "UMAP 2"),
               "tsne": ("t-SNE 1", "t-SNE 2"),
               "pca":  ("PC 1", "PC 2")}.get(reducer, ("Dim 1", "Dim 2"))

    if reducer == "pca" and hasattr(red, "explained_variance_ratio_"):
        var = red.explained_variance_ratio_ * 100
        xl  = f"PC 1 ({var[0]:.1f}%)"
        yl  = f"PC 2 ({var[1]:.1f}%)"

    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(f"CAV direction space ({reducer.upper()})\n{cav_pattern}", fontsize=12)
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved direction map: {out_path}")


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
                             "e.g. '/path/to/cavs/*/L25_concept_v1.npy'.")
    parser.add_argument("--out", required=True,
                        help="Output file path.")
    parser.add_argument("--reducer", default="pca",
                        choices=["umap", "tsne", "pca"],
                        help="Dimensionality reduction algorithm (default: pca).")
    parser.add_argument("--no-labels", action="store_true",
                        help="Suppress concept name labels.")
    args = parser.parse_args()

    plot_direction_map(
        cav_pattern=args.cav_pattern,
        out_path=args.out,
        reducer=args.reducer,
        label_points=not args.no_labels,
    )


if __name__ == "__main__":
    main()
