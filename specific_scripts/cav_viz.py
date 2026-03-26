#!/usr/bin/env python3
"""
cav_viz.py — Visualize CAV directions and cell projections.

Works with any CAV library structure (cell types, disease, age, treatment, etc.).
Reads library_structure.json produced by analyze_cav_library.py to interpret
the naming convention automatically — run that first for best results.

Three main figures:

  1. Direction map  (--plot direction-map)
     All CAV direction vectors projected to 2D (PCA of the direction matrix).
     Each point is one concept; baseline and condition versions of the same
     group are connected by an arrow showing the contrast direction.
     Works for any contrast: normal/cancer, young/old, treated/untreated, etc.

  2. Condition scatter  (--plot condition-scatter)
     For each group with a matched baseline + condition pair, scatter cells
     on (baseline_score, condition_score) axes with the delta direction
     drawn as an arrow.

  3. CAV-space UMAP  (--plot cav-umap)
     Cells embedded in CAV coordinate space (from cav_hierarchy.py
     cell_coordinates.tsv) and reduced to 2D with UMAP.

Usage
-----
# Direction map (only needs the library — no cell embeddings required):
python specific_scripts/cav_viz.py \\
    --lib-dir  cav_library/3f7c572c/ \\
    --pca-pkl  reference_population/global_pca_v1.pkl \\
    --plot     direction-map \\
    --out      results/figures/direction_map.png

# Condition scatter (needs cell embeddings + obs metadata):
python specific_scripts/cav_viz.py \\
    --lib-dir  cav_library/3f7c572c/ \\
    --pca-pkl  reference_population/global_pca_v1.pkl \\
    --pkl      cav_library/3f7c572c/embeddings/cells.pkl \\
    --obs      cav_library/3f7c572c/data/cells.h5ad \\
    --plot     condition-scatter \\
    --out      results/figures/condition_scatter/

# CAV-space UMAP (needs cell_coordinates.tsv from cav_hierarchy.py):
python specific_scripts/cav_viz.py \\
    --coords   results/3f7c572c/hierarchy/cell_coordinates.tsv \\
    --obs      cav_library/3f7c572c/data/cells.h5ad \\
    --plot     cav-umap \\
    --out      results/figures/cav_umap.png
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "tcav"))
from src.utils.data_loader import load_sequence_embeddings
from src.utils.preprocessing import preprocess_embeddings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CELL_TYPE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
]


def load_library_structure(lib_dir: Path) -> Optional[dict]:
    """
    Load library_structure.json if it exists (produced by analyze_cav_library.py).
    Returns None if not found — callers fall back to basic heuristics.
    """
    path = lib_dir / "library_structure.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def cav_group_from_structure(name: str, structure: dict) -> str:
    """Return the group value for a CAV name using the inferred structure."""
    sep   = structure.get("separator", "__")
    level = structure.get("group_level", 0)
    parts = name.split(sep)
    return parts[level] if level < len(parts) else parts[-1]


def cav_label_from_structure(name: str, structure: dict) -> str:
    """Human-readable label: all parts except level 0, joined."""
    sep   = structure.get("separator", "__")
    parts = name.split(sep)
    return " ".join(parts[1:]).replace("_", " ")


def cav_condition_from_structure(name: str, structure: dict) -> Optional[str]:
    """Return the condition value for this CAV, or None if no condition level."""
    cl = structure.get("condition_level")
    if cl is None:
        return None
    sep   = structure.get("separator", "__")
    parts = name.split(sep)
    return parts[cl] if cl < len(parts) else None


def load_directions(lib_dir: Path, version: str = "v1") -> Dict[str, np.ndarray]:
    """Load and unit-normalise all concept vectors from lib_dir/cavs/."""
    cavs_dir = lib_dir / "cavs"
    result = {}
    for d in sorted(cavs_dir.iterdir()):
        npy = d / f"concept_{version}.npy"
        if not npy.exists():
            continue
        v = np.load(npy).astype(np.float64)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            result[d.name] = v / norm
    return result


def load_pca_artifacts(pca_pkl: Path):
    import joblib
    artifacts = joblib.load(pca_pkl)
    return artifacts["scaler"], artifacts["pca"]


def load_obs(h5ad_path: str, columns: List[str]) -> pd.DataFrame:
    """Load only obs columns from an h5ad, without reading the expression matrix."""
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs[columns].copy()
    adata.file.close()
    return obs


# ---------------------------------------------------------------------------
# Figure 1: Direction map
# ---------------------------------------------------------------------------

def plot_direction_map(
    lib_dir: Path,
    out_path: str,
    pca_pkl: Optional[Path] = None,
    version: str = "v1",
    figsize: Tuple[int, int] = (10, 9),
):
    """
    Project all CAV direction vectors to 2D via PCA of the direction matrix.
    Normal and cancer versions of the same (cell_type, tissue) are connected
    by an arrow showing the disease-transition delta.
    """
    cav_dirs = load_directions(lib_dir, version)
    if not cav_dirs:
        raise ValueError(f"No CAV directions found in {lib_dir}/cavs/")

    names  = list(cav_dirs.keys())
    matrix = np.vstack([cav_dirs[n] for n in names])

    # PCA of the direction vectors → 2D
    pca2d  = PCA(n_components=2, random_state=42)
    coords = pca2d.fit_transform(matrix)
    coord_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=names)

    # Load inferred structure if available, else fall back to heuristics
    structure   = load_library_structure(lib_dir)
    has_condition = structure is not None and structure.get("condition_level") is not None
    baseline_vals = set(structure["baseline_values"]) if has_condition else set()

    if structure is not None:
        group_fn     = lambda n: cav_group_from_structure(n, structure)
        label_fn     = lambda n: cav_label_from_structure(n, structure)
        condition_fn = lambda n: cav_condition_from_structure(n, structure)
        pairs        = {(p["baseline"], p["condition"]) for p in structure.get("pairs", [])}
        logger.info(f"Using library_structure.json: {structure['summary']}")
    else:
        logger.info("No library_structure.json found — run analyze_cav_library.py "
                    "first for accurate structure inference. Using heuristics.")
        sep = "__"
        group_fn   = lambda n: n.split(sep)[1] if len(n.split(sep)) >= 2 else n
        label_fn   = lambda n: " ".join(n.split(sep)[1:]).replace("_", " ")
        condition_fn = lambda n: n.split(sep)[-1] if len(n.split(sep)) == 3 else None
        pairs        = set()

    groups    = sorted({group_fn(n) for n in names})
    color_map = {g: CELL_TYPE_COLORS[i % len(CELL_TYPE_COLORS)]
                 for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=figsize)

    # Draw baseline → condition arrows for matched pairs
    if has_condition and structure is not None:
        for pair in structure.get("pairs", []):
            normal_name  = pair["baseline"]
            disease_name = pair["condition"]
            if normal_name not in coord_df.index or disease_name not in coord_df.index:
                continue
            x0, y0 = coord_df.loc[normal_name, ["PC1", "PC2"]]
            x1, y1 = coord_df.loc[disease_name, ["PC1", "PC2"]]
            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color_map[group_fn(normal_name)],
                    lw=1.2, alpha=0.6,
                ),
            )

    # Plot points
    for name in names:
        x, y    = coord_df.loc[name, ["PC1", "PC2"]]
        group     = group_fn(name)
        condition = condition_fn(name)
        is_baseline = (condition in baseline_vals) if (condition and baseline_vals) else True
        marker = "o" if is_baseline else "X"
        size   = 80  if is_baseline else 70
        alpha  = 0.9 if is_baseline else 0.75
        ax.scatter(x, y, c=color_map[group], s=size, marker=marker,
                   alpha=alpha, edgecolors="white", linewidths=0.5, zorder=3)

    # Labels — show on normal/unpaired points only
    for name in names:
        condition   = condition_fn(name)
        is_baseline = (condition in baseline_vals) if (condition and baseline_vals) else True
        if has_condition and not is_baseline:
            continue
        x, y = coord_df.loc[name, ["PC1", "PC2"]]
        ax.annotate(label_fn(name), (x, y), fontsize=6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", color="dimgray")

    # Legend
    handles = [mpatches.Patch(color=color_map[g], label=g.replace("_", " "))
               for g in groups]
    if has_condition:
        baseline_label   = (structure["baseline_values"][0]
                            if structure and structure["baseline_values"] else "baseline")
        handles += [
            plt.scatter([], [], marker="o", c="gray", s=60, label=baseline_label),
            plt.scatter([], [], marker="X", c="gray", s=50, label="condition"),
        ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, frameon=False)

    var1 = pca2d.explained_variance_ratio_[0] * 100
    var2 = pca2d.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=11)
    summary = structure["summary"] if structure else lib_dir.name
    title = f"CAV direction space\n{summary}"
    ax.set_title(title, fontsize=12)
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    ax.set_aspect("equal")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved direction map: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Disease scatter
# ---------------------------------------------------------------------------

def plot_disease_scatter(
    lib_dir: Path,
    pkl_path: str,
    pca_pkl: Path,
    out_dir: str,
    obs_path: Optional[str] = None,
    disease_col: str = "disease",
    cell_type_col: str = "cell_type",
    tissue_col: str = "tissue",
    version: str = "v1",
    max_cells: int = 5000,
):
    """
    For each (cell_type, tissue) pair with matched normal + cancer, scatter
    cells on (L2_normal_score, L2_cancer_score) axes and draw the delta
    direction as an arrow.
    """
    from cav_hierarchy import (build_hierarchy, discover_cavs,
                                load_all_directions, orthogonalize)

    scaler, pca = load_pca_artifacts(pca_pkl)

    logger.info(f"Loading embeddings: {pkl_path}")
    embs_raw, cell_ids = load_sequence_embeddings(pkl_path)
    X = preprocess_embeddings(embs_raw, scaler, pca)

    cav_names = discover_cavs(lib_dir, version)
    cav_dirs  = load_all_directions(lib_dir, cav_names, version)
    hierarchy = build_hierarchy(cav_names, cav_dirs)

    # Load obs metadata if available
    obs = None
    if obs_path:
        try:
            cols = [c for c in [disease_col, cell_type_col, tissue_col]
                    if c is not None]
            obs = load_obs(obs_path, cols)
            obs = obs.reindex(cell_ids)
        except Exception as e:
            logger.warning(f"Could not load obs metadata: {e}")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Find matched normal/cancer pairs at L2
    L2 = hierarchy["L2"]
    pairs = set()
    for (ct, tissue, disease) in L2:
        if disease != "normal" and (ct, tissue, "normal") in L2:
            pairs.add((ct, tissue))

    logger.info(f"Plotting {len(pairs)} matched normal/cancer pairs")

    for (ct, tissue) in sorted(pairs):
        v_normal = L2[(ct, tissue, "normal")]

        # Find the cancer disease key for this pair
        cancer_keys = [(ct2, t2, d2) for (ct2, t2, d2) in L2
                       if ct2 == ct and t2 == tissue and d2 != "normal"]
        if not cancer_keys:
            continue
        ct2, t2, cancer_disease = cancer_keys[0]
        v_cancer = L2[(ct2, t2, cancer_disease)]

        # Project all cells onto both axes
        scores_normal = X @ v_normal
        scores_cancer = X @ v_cancer

        # Subsample for plotting
        rng = np.random.default_rng(42)
        n = min(max_cells, len(cell_ids))
        idx = rng.choice(len(cell_ids), n, replace=False)

        xvals = scores_normal[idx]
        yvals = scores_cancer[idx]

        # Color by known disease label if available
        if obs is not None and disease_col in obs.columns:
            disease_labels = obs[disease_col].iloc[idx].fillna("unknown")
            unique_labels  = disease_labels.unique()
            palette = cm.get_cmap("Set2", len(unique_labels))
            label_color = {l: palette(i) for i, l in enumerate(unique_labels)}
            colors = [label_color[l] for l in disease_labels]
        else:
            colors = "steelblue"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(xvals, yvals, c=colors, s=8, alpha=0.4, linewidths=0)

        # Draw delta direction as arrow from center of mass
        cx, cy = xvals.mean(), yvals.mean()
        # Delta in 2D score space: direction of (v_cancer - v_normal)
        delta = v_cancer - v_normal
        # Project delta onto the two axes to get its 2D representation
        dx = np.dot(delta, v_normal)
        dy = np.dot(delta, v_cancer)
        scale = 0.3 * max(xvals.std(), yvals.std()) / (np.hypot(dx, dy) + 1e-10)
        ax.annotate(
            "disease\ndirection",
            xy=(cx + dx * scale, cy + dy * scale),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2),
            fontsize=9, color="crimson", ha="center",
        )

        ax.set_xlabel(f"Score on {ct}__{tissue}__normal (L2 residual)", fontsize=9)
        ax.set_ylabel(f"Score on {ct}__{tissue}__{cancer_disease} (L2 residual)", fontsize=9)
        ax.set_title(f"{ct.replace('_',' ')} — {tissue.replace('_',' ')}", fontsize=11)

        if obs is not None and disease_col in obs.columns:
            handles = [mpatches.Patch(color=label_color[l], label=l)
                       for l in unique_labels]
            ax.legend(handles=handles, fontsize=8, frameon=False)

        fname = f"{ct}__{tissue}.png"
        plt.tight_layout()
        plt.savefig(out_dir_path / fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Figure 3: CAV-space UMAP
# ---------------------------------------------------------------------------

def _make_reducer(reducer: str, n_components: int = 2):
    """Return a fitted-reducer-compatible object for the given method name."""
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
                    perplexity=30, n_iter=1000, init="pca", learning_rate="auto")
    elif reducer == "pca":
        return PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown reducer '{reducer}'. Choose: umap, tsne, pca")


def plot_cav_umap(
    coords_path: str,
    out_path: str,
    obs_path: Optional[str] = None,
    color_col: str = "cell_type",
    level_prefix: str = "L0",
    reducer: str = "umap",
    max_cells: int = 20000,
    figsize: Tuple[int, int] = (9, 8),
):
    """
    2-D embedding of cells in CAV coordinate space. `level_prefix` selects
    which axes to use as features (e.g. 'L0', 'L1', 'L2', or 'delta').
    `reducer` controls the algorithm: 'umap' (default), 'tsne', or 'pca'.
    """
    logger.info(f"Loading coordinates: {coords_path}")
    coords = pd.read_csv(coords_path, sep="\t", index_col=0)

    # Select axes for the chosen level
    feature_cols = [c for c in coords.columns if c.startswith(level_prefix + "__")]
    if not feature_cols:
        raise ValueError(f"No columns with prefix '{level_prefix}__' in {coords_path}")
    logger.info(f"Using {len(feature_cols)} '{level_prefix}' axes as features")

    # Subsample
    rng = np.random.default_rng(42)
    n = min(max_cells, len(coords))
    idx = rng.choice(len(coords), n, replace=False)
    X_feat = coords[feature_cols].iloc[idx].values

    logger.info(f"Running {reducer.upper()} on {n} cells × {len(feature_cols)} axes...")
    emb = _make_reducer(reducer).fit_transform(X_feat)

    # Color by metadata
    colors = "steelblue"
    handles = None
    if obs_path and color_col:
        try:
            obs = load_obs(obs_path, [color_col])
            labels = obs[color_col].reindex(coords.index.astype(str)).iloc[idx].fillna("unknown")
            unique = sorted(labels.unique())
            palette = cm.get_cmap("tab20", len(unique))
            lc_map  = {l: palette(i) for i, l in enumerate(unique)}
            colors  = [lc_map[l] for l in labels]
            handles = [mpatches.Patch(color=lc_map[l],
                                       label=l.replace("_", " "))
                       for l in unique]
        except Exception as e:
            logger.warning(f"Could not load color column '{color_col}': {e}")

    axis_labels = {
        "umap": ("UMAP 1", "UMAP 2"),
        "tsne": ("t-SNE 1", "t-SNE 2"),
        "pca":  ("PC 1", "PC 2"),
    }
    xl, yl = axis_labels.get(reducer, ("Dim 1", "Dim 2"))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4, alpha=0.5, linewidths=0)
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(f"CAV-space {reducer.upper()} ({level_prefix} axes)\ncolored by {color_col}",
                 fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=7, frameon=False, markerscale=2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved UMAP: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize CAV directions and disease transitions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--plot", required=True,
                        choices=["direction-map", "condition-scatter", "cav-umap"],
                        help="Which figure to produce.")
    parser.add_argument("--lib-dir",
                        help="CAV library directory (cavs/ subdirectory).")
    parser.add_argument("--pca-pkl",
                        help="Global PCA pkl (scaler + PCA).")
    parser.add_argument("--pkl",
                        help="Cell embedding pkl for condition-scatter.")
    parser.add_argument("--obs",
                        help="h5ad file for cell metadata (obs columns).")
    parser.add_argument("--coords",
                        help="cell_coordinates.tsv from cav_hierarchy.py (for cav-umap).")
    parser.add_argument("--out", required=True,
                        help="Output path (file for direction-map/cav-umap, "
                             "directory for condition-scatter).")
    parser.add_argument("--color-col", default="cell_type",
                        help="obs column to color UMAP by (default: cell_type).")
    parser.add_argument("--level", default="L0",
                        choices=["L0", "L1", "L2", "delta"],
                        help="Hierarchy level to use as embedding features (default: L0).")
    parser.add_argument("--reducer", default="umap",
                        choices=["umap", "tsne", "pca"],
                        help="Dimensionality reduction for cav-umap "
                             "(default: umap; tsne/pca need no extra install).")
    parser.add_argument("--max-cells", type=int, default=20000,
                        help="Subsample cells for scatter/UMAP (default: 20000).")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    if args.plot == "direction-map":
        if not args.lib_dir:
            parser.error("--lib-dir required for direction-map")
        plot_direction_map(
            lib_dir=Path(args.lib_dir),
            out_path=args.out,
            pca_pkl=Path(args.pca_pkl) if args.pca_pkl else None,
            version=args.version,
        )

    elif args.plot == "condition-scatter":
        for req in ["lib_dir", "pkl", "pca_pkl"]:
            if not getattr(args, req):
                parser.error(f"--{req.replace('_','-')} required for condition-scatter")
        plot_disease_scatter(
            lib_dir=Path(args.lib_dir),
            pkl_path=args.pkl,
            pca_pkl=Path(args.pca_pkl),
            out_dir=args.out,
            obs_path=args.obs,
            version=args.version,
            max_cells=args.max_cells,
        )

    elif args.plot == "cav-umap":
        if not args.coords:
            parser.error("--coords required for cav-umap")
        plot_cav_umap(
            coords_path=args.coords,
            out_path=args.out,
            obs_path=args.obs,
            color_col=args.color_col,
            level_prefix=args.level,
            reducer=args.reducer,
            max_cells=args.max_cells,
        )


if __name__ == "__main__":
    main()
