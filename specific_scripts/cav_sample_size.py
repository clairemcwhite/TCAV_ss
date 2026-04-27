#!/usr/bin/env python3
"""
cav_sample_size.py — Test CAV direction stability across increasing sample sizes.

For a specified (tissue, cell_type, condition) group, trains CAVs at a range of
cell counts (from very few up to --max-n), repeating each size --n-repeats times
with different random subsamples. Plots the resulting CAV vectors projected into
2D, colored by sample size, to show that directions converge as n increases.

Usage
-----
python specific_scripts/cav_sample_size.py \\
    --h5ad        data/cells.h5ad \\
    --pkl         embeddings/cells.pkl \\
    --pca-pkl     reference_population/global_pca_v1.pkl \\
    --tissue      lung \\
    --cell-type   T_cell \\
    --condition   cancer \\
    --baseline    normal \\
    --tissue-col  tissue \\
    --celltype-col cell_type \\
    --condition-col disease \\
    --max-n       500 \\
    --n-repeats   10 \\
    --out         results/sample_size/

Outputs
-------
    cav_directions.png   — 2D PCA of CAV vectors, colored by n
    cav_directions.tsv   — raw data (n, repeat, PC1, PC2, cosine_sim_to_full)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_label(s):
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def load_embeddings_pkl(pkl_path, cell_ids):
    """Load embeddings from a .pkl file, returning array aligned to cell_ids."""
    import pickle
    logger.info(f"Loading embeddings from {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Support dict {cell_id: embedding} or (ids, matrix) tuple
    if isinstance(data, dict):
        id_to_emb = {str(k): v for k, v in data.items()}
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        ids, mat = data
        id_to_emb = {str(i): mat[j] for j, i in enumerate(ids)}
    else:
        raise ValueError("Unrecognised pkl format — expected dict or (ids, matrix) tuple")

    found = [cid for cid in cell_ids if cid in id_to_emb]
    missing = len(cell_ids) - len(found)
    if missing:
        logger.warning(f"{missing} cell IDs not found in pkl — dropped")
    mat = np.array([id_to_emb[cid] for cid in found])
    return mat, found


def load_obs(h5ad_path, tissue_col, celltype_col, condition_col):
    import anndata as ad
    logger.info(f"Loading obs metadata from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs[[tissue_col, celltype_col, condition_col]].copy()
    obs.index = obs.index.astype(str)
    adata.file.close()
    return obs


def preprocess(X, scaler, pca):
    X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return X


def train_cav_direction(X_pos, X_neg, scaler, pca):
    """Train logistic regression and return unit-normalised weight vector."""
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    X_pre = preprocess(X, scaler, pca)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                              class_weight="balanced", random_state=42)
    clf.fit(X_pre, y)
    v = clf.coef_[0]
    return v / (np.linalg.norm(v) + 1e-10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CAV direction stability vs. sample size.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h5ad",          required=True)
    parser.add_argument("--pkl",           required=True,
                        help="Embeddings pkl: dict {cell_id: vec} or (ids, matrix)")
    parser.add_argument("--pca-pkl",       required=True,
                        help="Global PCA pkl (joblib) with scaler and pca attributes, "
                             "or a tuple (scaler, pca)")
    parser.add_argument("--tissue",        required=True)
    parser.add_argument("--cell-type",     required=True)
    parser.add_argument("--condition",     required=True,
                        help="Disease condition to use as positive class")
    parser.add_argument("--baseline",      default="normal",
                        help="Baseline condition for negative class (default: normal)")
    parser.add_argument("--tissue-col",    default="tissue")
    parser.add_argument("--celltype-col",  default="cell_type")
    parser.add_argument("--condition-col", default="disease")
    parser.add_argument("--max-n",         type=int, default=500,
                        help="Maximum cells per class (default: 500)")
    parser.add_argument("--n-repeats",     type=int, default=10,
                        help="Repeats per sample size (default: 10)")
    parser.add_argument("--out",           required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load scaler + PCA
    # ------------------------------------------------------------------ #
    import joblib
    logger.info(f"Loading PCA pkl: {args.pca_pkl}")
    pca_obj = joblib.load(args.pca_pkl)
    if isinstance(pca_obj, (list, tuple)) and len(pca_obj) == 2:
        scaler, pca = pca_obj
    elif hasattr(pca_obj, "scaler") and hasattr(pca_obj, "pca"):
        scaler, pca = pca_obj.scaler, pca_obj.pca
    else:
        raise ValueError("--pca-pkl must be a (scaler, pca) tuple or object with .scaler/.pca")

    # ------------------------------------------------------------------ #
    # Load metadata and filter to target groups
    # ------------------------------------------------------------------ #
    obs = load_obs(args.h5ad, args.tissue_col, args.celltype_col, args.condition_col)

    t  = norm_label(args.tissue)
    ct = norm_label(args.cell_type)
    cd = norm_label(args.condition)
    bl = norm_label(args.baseline)

    def mask(tissue_val, ct_val, cond_val):
        return (
            obs[args.tissue_col].apply(norm_label).eq(tissue_val) &
            obs[args.celltype_col].apply(norm_label).eq(ct_val) &
            obs[args.condition_col].apply(norm_label).eq(cond_val)
        )

    pos_ids = obs[mask(t, ct, cd)].index.tolist()
    neg_ids = obs[mask(t, ct, bl)].index.tolist()

    logger.info(f"Positive cells ({args.condition}): {len(pos_ids)}")
    logger.info(f"Negative cells ({args.baseline}):  {len(neg_ids)}")

    if len(pos_ids) < 5 or len(neg_ids) < 5:
        logger.error("Too few cells in one class — check tissue/cell_type/condition values.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Load embeddings
    # ------------------------------------------------------------------ #
    all_ids = pos_ids + neg_ids
    all_embs, found_ids = load_embeddings_pkl(args.pkl, all_ids)
    found_set = set(found_ids)

    pos_embs = all_embs[[i for i, cid in enumerate(found_ids) if cid in set(pos_ids)]]
    neg_embs = all_embs[[i for i, cid in enumerate(found_ids) if cid in set(neg_ids)]]

    logger.info(f"Embeddings loaded — pos: {len(pos_embs)}, neg: {len(neg_embs)}")

    max_n = min(args.max_n, len(pos_embs), len(neg_embs))
    if max_n < 10:
        logger.error(f"Only {max_n} cells available per class after embedding lookup — too few.")
        sys.exit(1)

    # Sample sizes: log-spaced from 5 up to max_n
    ns = np.unique(np.round(
        np.geomspace(5, max_n, num=15)
    ).astype(int))
    ns = ns[ns <= max_n]
    logger.info(f"Sample sizes to test: {ns.tolist()}")

    # ------------------------------------------------------------------ #
    # Train CAV at each n, multiple repeats
    # ------------------------------------------------------------------ #
    # First train on full data to use as reference direction
    logger.info("Training reference CAV on full data...")
    ref_vec = train_cav_direction(pos_embs, neg_embs, scaler, pca)

    rng = np.random.default_rng(42)
    records = []
    vectors = []

    for n in ns:
        for rep in range(args.n_repeats):
            pos_idx = rng.choice(len(pos_embs), size=n, replace=False)
            neg_idx = rng.choice(len(neg_embs), size=n, replace=False)
            v = train_cav_direction(pos_embs[pos_idx], neg_embs[neg_idx], scaler, pca)
            cos_sim = float(np.dot(v, ref_vec))
            records.append({"n": int(n), "repeat": rep, "cosine_sim": cos_sim})
            vectors.append(v)
        logger.info(f"  n={n:4d}  mean cosine_sim={np.mean([r['cosine_sim'] for r in records if r['n']==n]):.3f}")

    vectors = np.array(vectors)   # (n_experiments, cav_dim)

    # ------------------------------------------------------------------ #
    # Project CAV vectors into 2D via PCA for visualisation
    # ------------------------------------------------------------------ #
    logger.info("Projecting CAV vectors to 2D for visualisation...")
    all_vecs = np.vstack([vectors, ref_vec[np.newaxis]])
    pca2 = PCA(n_components=2, random_state=42)
    coords2d = pca2.fit_transform(all_vecs)

    exp_var = pca2.explained_variance_ratio_
    coords_exp = coords2d[:-1]   # experiments
    ref_coord  = coords2d[-1]    # reference

    df = pd.DataFrame(records)
    df["pc1"] = coords_exp[:, 0]
    df["pc2"] = coords_exp[:, 1]
    df.to_csv(out_dir / "cav_directions.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------ #
    # Plot 1: 2D vector directions colored by n
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # — scatter of CAV directions —
    ax = axes[0]
    unique_ns = sorted(df["n"].unique())
    cmap  = cm.plasma
    norm  = matplotlib.colors.LogNorm(vmin=unique_ns[0], vmax=unique_ns[-1])

    for n in unique_ns:
        sub = df[df["n"] == n]
        ax.scatter(sub["pc1"], sub["pc2"],
                   c=[cmap(norm(n))] * len(sub),
                   s=28, alpha=0.75, linewidths=0)

    # Reference point
    ax.scatter(*ref_coord, c="black", s=80, marker="*", zorder=5, label=f"Full N ({max_n})")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("n cells per class (log scale)", fontsize=9)

    ax.set_xlabel(f"CAV direction PC1 ({exp_var[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"CAV direction PC2 ({exp_var[1]:.1%} var)", fontsize=10)
    ax.set_title(f"CAV direction stability\n"
                 f"{args.tissue} · {args.cell_type} · {args.condition} vs {args.baseline}",
                 fontsize=10)
    ax.legend(fontsize=8)

    # — cosine similarity vs n —
    ax2 = axes[1]
    summary = df.groupby("n")["cosine_sim"].agg(["mean", "std"]).reset_index()

    ax2.fill_between(summary["n"],
                     summary["mean"] - summary["std"],
                     summary["mean"] + summary["std"],
                     alpha=0.2, color="#8a4fc8")
    ax2.plot(summary["n"], summary["mean"],
             color="#8a4fc8", linewidth=2, marker="o", markersize=4)
    ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("n cells per class (log scale)", fontsize=10)
    ax2.set_ylabel("Cosine similarity to full-N CAV", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Direction convergence with sample size", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"CAV sample size analysis: {args.tissue} · {args.cell_type} · {args.condition}",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out_path = out_dir / "cav_directions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
