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
# Static PNG:
python specific_scripts/cav_viz_pattern.py \\
    --cav-pattern "/path/to/cavs/*/L25_concept_v1.npy" \\
    --reducer     umap \\
    --out         results/figures/pfam_umap.png

# Interactive Plotly HTML (with PFAM hover annotations):
python specific_scripts/cav_viz_pattern.py \\
    --cav-pattern    "/path/to/cavs/*/L25_concept_v1.npy" \\
    --reducer        umap \\
    --pfam-annotations pfamA.txt \\
    --interactive \\
    --out            results/figures/pfam_umap.html
"""

import re
import argparse
import logging
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_VERSION_SUFFIX = re.compile(r'_v\d+$')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _concept_name(path: Path) -> str:
    """
    Derive a concept name from a .npy file path.
    Uses the parent directory name when the file lives in a per-concept subdirectory
    (e.g. cavs/PF01112/L25_concept_v1.npy → 'PF01112'), otherwise falls back to
    the file stem with any trailing _v1-style suffix stripped.
    """
    parent = path.parent.name
    stem   = _VERSION_SUFFIX.sub('', path.stem)
    if 'concept' in stem:
        return parent
    return stem


def load_directions_from_pattern(pattern: str) -> Dict[str, np.ndarray]:
    """Load and unit-normalise CAV direction vectors from all files matching `pattern`."""
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    result = {}
    for p in paths:
        path = Path(p)
        name = _concept_name(path)
        v    = np.load(path).astype(np.float64)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            result[name] = v / norm
        else:
            logger.warning(f"Skipping zero-norm vector: {path}")
    logger.info(f"Loaded {len(result)} CAV directions from pattern '{pattern}'")
    return result


def load_pfam_annotations(pfam_txt: str) -> pd.DataFrame:
    """
    Parse pfamA.txt (tab-separated, no header).
    Returns a DataFrame indexed by accession with columns:
      short_name, description, long_description.
    col 0: accession, col 1: short name, col 2: one-line description,
    col 5: extended paragraph description.
    """
    rows = []
    with open(pfam_txt) as fh:
        for line in fh:
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            rows.append({
                "accession":        parts[0].strip(),
                "short_name":       parts[1].strip(),
                "description":      parts[2].strip(),
                "long_description": parts[5].strip() if len(parts) > 5 else "",
            })
    df = pd.DataFrame(rows).set_index("accession")
    logger.info(f"Loaded {len(df)} PFAM annotations from {pfam_txt}")
    return df


# ---------------------------------------------------------------------------
# Shared: compute 2-D embedding
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


def _embed(cav_dirs: Dict[str, np.ndarray], reducer: str, dims: int = 2):
    names  = list(cav_dirs.keys())
    matrix = np.vstack([cav_dirs[n] for n in names])
    logger.info(f"Running {reducer.upper()} ({dims}D) on {len(names)} CAV vectors...")
    red    = _make_reducer(reducer, n_components=dims)
    coords = red.fit_transform(matrix)
    cols   = ["D1", "D2", "D3"][:dims]
    coord_df = pd.DataFrame(coords, columns=cols, index=names)
    return coord_df, red


def _axis_labels(reducer: str, red) -> Tuple[str, str]:
    xl, yl = {"umap": ("UMAP 1", "UMAP 2"),
               "tsne": ("t-SNE 1", "t-SNE 2"),
               "pca":  ("PC 1", "PC 2")}.get(reducer, ("Dim 1", "Dim 2"))
    if reducer == "pca" and hasattr(red, "explained_variance_ratio_"):
        var = red.explained_variance_ratio_ * 100
        xl, yl = f"PC 1 ({var[0]:.1f}%)", f"PC 2 ({var[1]:.1f}%)"
    return xl, yl


# ---------------------------------------------------------------------------
# Static plot (matplotlib)
# ---------------------------------------------------------------------------

def plot_direction_map(
    cav_pattern: str,
    out_path: str,
    reducer: str = "pca",
    figsize: Tuple[int, int] = (10, 9),
    label_points: bool = True,
):
    """Embed all matched CAV direction vectors in 2D and save a static PNG."""
    cav_dirs = load_directions_from_pattern(cav_pattern)
    coord_df, red = _embed(cav_dirs, reducer)
    xl, yl = _axis_labels(reducer, red)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coord_df["D1"], coord_df["D2"], s=70, marker="o",
               color="steelblue", alpha=0.85, edgecolors="white",
               linewidths=0.5, zorder=3)

    if label_points:
        for name in coord_df.index:
            x, y = coord_df.loc[name, ["D1", "D2"]]
            ax.annotate(name, (x, y), fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points", color="dimgray")

    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(f"CAV direction space ({reducer.upper()})\n{cav_pattern}", fontsize=12)
    if reducer == "pca":
        ax.axhline(0, color="lightgray", lw=0.5)
        ax.axvline(0, color="lightgray", lw=0.5)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved static plot: {out_path}")


# ---------------------------------------------------------------------------
# Interactive plot (Plotly)
# ---------------------------------------------------------------------------

def plot_direction_map_interactive(
    cav_pattern: str,
    out_path: str,
    reducer: str = "pca",
    dims: int = 2,
    pfam_annotations: Optional[str] = None,
):
    """Embed CAV direction vectors in 2D or 3D and save an interactive Plotly HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is not installed. Run: pip install plotly")

    if dims not in (2, 3):
        raise ValueError("--dims must be 2 or 3")

    cav_dirs = load_directions_from_pattern(cav_pattern)
    coord_df, red = _embed(cav_dirs, reducer, dims=dims)
    xl, yl = _axis_labels(reducer, red)
    zl = yl.replace("2", "3")  # e.g. "UMAP 3", "PC 3"

    # Build hover text
    annot = None
    if pfam_annotations:
        annot = load_pfam_annotations(pfam_annotations)

    hover_texts = []
    for acc in coord_df.index:
        if annot is not None and acc in annot.index:
            row = annot.loc[acc]
            long = row["long_description"]
            # Wrap long description at ~80 chars per line for readability
            wrapped = "<br>".join(
                long[i:i+80] for i in range(0, min(len(long), 400), 80)
            ) + ("…" if len(long) > 400 else "")
            text = (f"<b>{acc}</b> · {row['short_name']}<br>"
                    f"<i>{row['description']}</i><br><br>"
                    f"{wrapped}")
        else:
            text = f"<b>{acc}</b>"
        hover_texts.append(text)

    marker = dict(size=5, color="steelblue", opacity=0.85,
                  line=dict(width=0.5, color="white"))

    if dims == 3:
        trace = go.Scatter3d(
            x=coord_df["D1"], y=coord_df["D2"], z=coord_df["D3"],
            mode="markers",
            marker=marker,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
        layout = go.Layout(
            title=dict(text=f"CAV direction space ({reducer.upper()}, 3D)", font_size=15),
            scene=dict(
                xaxis_title=xl, yaxis_title=yl, zaxis_title=zl,
                xaxis=dict(showgrid=True, zeroline=False),
                yaxis=dict(showgrid=True, zeroline=False),
                zaxis=dict(showgrid=True, zeroline=False),
            ),
            hoverlabel=dict(bgcolor="white", font_size=12),
            paper_bgcolor="white",
            width=950, height=800,
        )
    else:
        trace = go.Scatter(
            x=coord_df["D1"], y=coord_df["D2"],
            mode="markers",
            marker=marker,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
        layout = go.Layout(
            title=dict(text=f"CAV direction space ({reducer.upper()})", font_size=15),
            xaxis=dict(title=xl, showgrid=False, zeroline=(reducer == "pca")),
            yaxis=dict(title=yl, showgrid=False, zeroline=(reducer == "pca")),
            hoverlabel=dict(bgcolor="white", font_size=12),
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=950, height=800,
        )

    fig = go.Figure(data=[trace], layout=layout)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info(f"Saved interactive {dims}D plot: {out_path}")


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
                        help="Output file path (.png for static, .html for interactive).")
    parser.add_argument("--reducer", default="pca",
                        choices=["umap", "tsne", "pca"],
                        help="Dimensionality reduction algorithm (default: pca).")
    parser.add_argument("--interactive", action="store_true",
                        help="Write an interactive Plotly HTML instead of a static PNG.")
    parser.add_argument("--dims", type=int, default=2, choices=[2, 3],
                        help="Embedding dimensions for interactive plot (default: 2).")
    parser.add_argument("--pfam-annotations",
                        help="Path to pfamA.txt for hover annotations "
                             "(accession, short name, description).")
    parser.add_argument("--no-labels", action="store_true",
                        help="Suppress concept name labels on static plot.")
    args = parser.parse_args()

    if args.interactive:
        plot_direction_map_interactive(
            cav_pattern=args.cav_pattern,
            out_path=args.out,
            reducer=args.reducer,
            dims=args.dims,
            pfam_annotations=args.pfam_annotations,
        )
    else:
        plot_direction_map(
            cav_pattern=args.cav_pattern,
            out_path=args.out,
            reducer=args.reducer,
            label_points=not args.no_labels,
        )


if __name__ == "__main__":
    main()
