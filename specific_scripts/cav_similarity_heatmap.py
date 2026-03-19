#!/usr/bin/env python3
"""
Pairwise cosine similarity heatmap across all CAVs in a library.

Usage
-----
python specific_scripts/cav_similarity_heatmap.py \
    --library cav_library/9d8e5dca-... \
    --out     cav_similarity.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_cavs(library_dir):
    cavs_dir = Path(library_dir) / "cavs"
    vectors, names = [], []
    for concept_dir in sorted(cavs_dir.iterdir()):
        cav_path = concept_dir / "concept_v1.npy"
        if not cav_path.exists():
            continue
        vec = np.load(cav_path)
        vectors.append(vec / np.linalg.norm(vec))   # unit norm
        # strip "cell_type__" prefix for cleaner labels
        label = concept_dir.name.replace("cell_type__", "").replace("_", " ")
        names.append(label)
    return np.vstack(vectors), names


def cosine_similarity_matrix(vecs):
    # vecs are already unit-normed, so dot product = cosine similarity
    return vecs @ vecs.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True,
                        help="Path to cav_library/<dataset_id>/")
    parser.add_argument("--out", default="cav_similarity.png",
                        help="Output image path (default: cav_similarity.png)")
    parser.add_argument("--figsize", type=int, default=12,
                        help="Figure size in inches (default: 12)")
    args = parser.parse_args()

    vecs, names = load_cavs(args.library)
    print(f"Loaded {len(names)} CAVs")

    sim = cosine_similarity_matrix(vecs)
    df  = pd.DataFrame(sim, index=names, columns=names)

    g = sns.clustermap(
        df,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        figsize=(args.figsize, args.figsize),
        annot=len(names) <= 30,   # annotate values only if small enough
        fmt=".2f",
        linewidths=0.5,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=0.15,
    )
    g.ax_heatmap.set_title("CAV cosine similarity", pad=12)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")

    # Print top similar pairs
    print("\nMost similar pairs:")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((sim[i, j], names[i], names[j]))
    for s, a, b in sorted(pairs, reverse=True)[:10]:
        print(f"  {s:+.3f}  {a}  ↔  {b}")

    print("\nMost dissimilar pairs:")
    for s, a, b in sorted(pairs)[:10]:
        print(f"  {s:+.3f}  {a}  ↔  {b}")


if __name__ == "__main__":
    main()
