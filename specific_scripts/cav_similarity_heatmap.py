#!/usr/bin/env python3
"""
Cell-type similarity heatmap via CAV cross-scoring.

For each cell type, scores its training cells against ALL CAVs, producing
a (cell_type x CAV) mean-score matrix. Correlation between rows gives
biologically meaningful cell-type similarity.

Also produces the raw CAV-direction cosine similarity for comparison.

Usage
-----
python specific_scripts/cav_similarity_heatmap.py \
    --library cav_library/9d8e5dca-... \
    --pkl     cav_library/9d8e5dca-.../embeddings/cells.pkl \
    --out     cav_similarity.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "tcav"))
from src.evaluate import load_cav_artifacts, compute_projections
from src.utils.data_loader import load_sequence_embeddings


def read_ids(path):
    p = Path(path)
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def load_all_cavs(library_dir):
    """Load all CAV artifacts. Returns {concept_name: artifacts}."""
    cavs = {}
    for d in sorted((Path(library_dir) / "cavs").iterdir()):
        if (d / "concept_v1.npy").exists():
            cavs[d.name] = load_cav_artifacts(str(d), version="v1")
    return cavs


def short_label(name):
    return name.replace("cell_type__", "").replace("_", " ")


def build_cross_score_matrix(library_dir, pkl_path):
    """
    For each concept, score its training-positive cells against every CAV.
    Returns a DataFrame: rows = cell types, columns = CAVs, values = mean score.
    """
    print("Loading embeddings...")
    seq_embs, cell_ids = load_sequence_embeddings(str(pkl_path))
    id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    print(f"  {len(cell_ids)} cells, dim={seq_embs.shape[1]}")

    print("Loading CAVs...")
    cavs = load_all_cavs(library_dir)
    print(f"  {len(cavs)} CAVs")

    spans_dir = Path(library_dir) / "spans"
    concept_names = sorted(cavs.keys())

    rows = {}
    for concept_name in concept_names:
        # Get training positive cells for this concept
        pos_ids = [i for i in read_ids(spans_dir / concept_name / "pos.txt")
                   if i in id_to_idx]
        if not pos_ids:
            print(f"  WARNING: no embeddings found for {concept_name}, skipping")
            continue
        pos_embs = seq_embs[[id_to_idx[i] for i in pos_ids]]

        # Score against every CAV
        scores = {}
        for cav_name, artifacts in cavs.items():
            s = compute_projections(
                pos_embs,
                artifacts["concept_cav"],
                artifacts["scaler"],
                artifacts["pca"],
            )
            scores[cav_name] = s.mean()

        rows[concept_name] = scores

    df = pd.DataFrame(rows).T   # rows = cell types, cols = CAVs
    df.index   = [short_label(n) for n in df.index]
    df.columns = [short_label(n) for n in df.columns]
    return df


def plot_cross_score(df, out_path, figsize=12):
    """Clustermap of raw cross-scores (mean CAV score per cell type)."""
    g = sns.clustermap(
        df,
        cmap="RdBu_r",
        center=0,
        figsize=(figsize, figsize),
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=0.15,
        linewidths=0.3,
        annot=len(df) <= 20,
        fmt=".2f",
    )
    g.fig.suptitle("Mean CAV score per cell type\n(rows = cell types, cols = CAVs)",
                   y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_celltype_similarity(df, out_path, figsize=10):
    """Clustermap of cell-type similarity via row correlation of cross-scores."""
    corr = df.T.corr()   # correlate rows → (cell_type x cell_type)

    g = sns.clustermap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        figsize=(figsize, figsize),
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=0.2,
        linewidths=0.5,
        annot=len(corr) <= 20,
        fmt=".2f",
    )
    g.fig.suptitle("Cell-type similarity (correlation of CAV score profiles)",
                   y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    # Print top pairs
    names = corr.columns.tolist()
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((corr.iloc[i, j], names[i], names[j]))

    print("\nMost similar cell types:")
    for s, a, b in sorted(pairs, reverse=True)[:10]:
        print(f"  {s:+.3f}  {a}  ↔  {b}")
    print("\nMost dissimilar cell types:")
    for s, a, b in sorted(pairs)[:10]:
        print(f"  {s:+.3f}  {a}  ↔  {b}")

    return corr


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--library", required=True,
                        help="Path to cav_library/<dataset_id>/")
    parser.add_argument("--pkl", required=True,
                        help="Path to embeddings pkl.")
    parser.add_argument("--out", default="cav_similarity",
                        help="Output prefix (default: cav_similarity). "
                             "Produces <out>_scores.png and <out>_celltypes.png")
    parser.add_argument("--figsize", type=int, default=12)
    args = parser.parse_args()

    df = build_cross_score_matrix(args.library, args.pkl)

    plot_cross_score(df,             f"{args.out}_scores.png",    args.figsize)
    plot_celltype_similarity(df,     f"{args.out}_celltypes.png", args.figsize - 2)


if __name__ == "__main__":
    main()
