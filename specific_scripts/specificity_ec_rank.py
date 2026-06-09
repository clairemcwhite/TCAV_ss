#!/usr/bin/env python3
"""
specificity_ec_rank.py

Test CAV specificity: score each val protein against ALL available EC CAVs
and find the rank of its true EC(s) among all scores.

If CAVs are specific, the true EC should consistently rank at or near 1
out of ~1000 EC CAVs — demonstrating that high recall is not just from
predicting everything positive.

Usage
-----
python specific_scripts/specificity_ec_rank.py \\
    --val-pkl        results/ec_eval/ecbenchtest/eval_embeddings/val_proteins.span.fasta.pkl \\
    --gold-standard  data/ec_gold_standard_long.tsv \\
    --ec-base-dirs   /xdisk/.../ec_dataset_part* \\
    --scaler-pkl     reference_population/scaler_v1.pkl \\
    --out-dir        results/ec_eval/ecbenchtest/
"""

import argparse
import glob
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CAV_SUBDIR  = "random_positive_train_max1000_cav"
CAV_VERSION = "v1"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--val-pkl", required=True,
                        help="PKL of val protein embeddings.")
    parser.add_argument("--gold-standard", required=True,
                        help="Long-format TSV with protein_id, ec_number, ec_cav_id columns.")
    parser.add_argument("--ec-base-dirs", nargs="+", required=True,
                        help="Base directories containing ecNo_* subdirs (glob patterns ok).")
    parser.add_argument("--scaler-pkl", required=True,
                        help="Shared reference population scaler pkl.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for results and figures.")
    parser.add_argument("--version", default="v1",
                        help="CAV version suffix (default: v1).")
    parser.add_argument("--exclude", default=None,
                        help="Optional TSV with ec_number and protein_id to exclude "
                             "(train/val overlap pairs).")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "tcav"))
    from src.utils.data_loader import load_sequence_embeddings
    import joblib

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load val embeddings
    # ------------------------------------------------------------------
    logger.info(f"Loading val embeddings from {args.val_pkl}")
    val_embs, val_ids = load_sequence_embeddings(args.val_pkl)
    logger.info(f"  {len(val_ids)} proteins, embedding dim {val_embs.shape[1]}")

    id_to_idx = {sid: i for i, sid in enumerate(val_ids)}
    # Also index by bare accession (handle sp|ACC|GENE format)
    _SKIP = {"sp", "tr", "sw", "ref"}
    for i, sid in enumerate(val_ids):
        for part in sid.split("|"):
            if part and part not in _SKIP:
                id_to_idx.setdefault(part, i)

    # ------------------------------------------------------------------
    # Load gold standard
    # ------------------------------------------------------------------
    gold = pd.read_csv(args.gold_standard, sep="\t", dtype=str)
    gold = gold[["protein_id", "ec_number", "ec_cav_id"]].drop_duplicates()

    if args.exclude:
        excl = pd.read_csv(args.exclude, sep="\t")[["ec_number", "protein_id"]]
        excl["_drop"] = True
        before = len(gold)
        gold = gold.merge(excl, on=["ec_number", "protein_id"], how="left")
        gold = gold[gold["_drop"].isna()].drop(columns="_drop").reset_index(drop=True)
        logger.info(f"Excluded {before - len(gold)} pairs via --exclude")

    # Keep only val proteins present in the embeddings pkl
    gold = gold[gold["protein_id"].isin(id_to_idx)].reset_index(drop=True)
    logger.info(f"Gold standard: {len(gold)} (protein, EC) pairs for "
                f"{gold['protein_id'].nunique()} proteins")

    # ------------------------------------------------------------------
    # Find and load all EC CAVs
    # ------------------------------------------------------------------
    ec_base_dirs = []
    for pattern in args.ec_base_dirs:
        ec_base_dirs.extend(Path(d) for d in sorted(glob.glob(pattern)))
    logger.info(f"Searching {len(ec_base_dirs)} base dir(s) for EC CAVs...")

    cav_vectors  = []   # list of np.array shape (dim,)
    cav_ids      = []   # ec_cav_id string for each CAV

    for base in ec_base_dirs:
        for ec_dir in sorted(base.glob("ecNo_*")):
            cav_file = ec_dir / CAV_SUBDIR / f"concept_{args.version}.npy"
            if cav_file.exists():
                cav_vectors.append(np.load(cav_file))
                cav_ids.append(ec_dir.name)

    n_cavs = len(cav_vectors)
    logger.info(f"Loaded {n_cavs} EC CAVs")
    if n_cavs == 0:
        raise SystemExit("No CAVs found — check --ec-base-dirs and --version.")

    # Stack into matrix: (n_cavs, dim)
    cav_matrix = np.stack(cav_vectors, axis=0)

    # ------------------------------------------------------------------
    # Preprocess val embeddings once, then score against all CAVs
    # ------------------------------------------------------------------
    logger.info(f"Loading scaler from {args.scaler_pkl}")
    scaler = joblib.load(args.scaler_pkl)

    logger.info("Preprocessing val embeddings...")
    val_preprocessed = scaler.transform(val_embs)  # (n_val, dim)

    logger.info(f"Scoring {len(val_ids)} proteins × {n_cavs} CAVs...")
    score_matrix = val_preprocessed @ cav_matrix.T  # (n_val, n_cavs)

    # Build lookup: ec_cav_id → column index
    cav_id_to_col = {cid: i for i, cid in enumerate(cav_ids)}

    # Build per-protein true EC set for fast lookup
    protein_true_cavs = gold.groupby("protein_id")["ec_cav_id"].apply(set).to_dict()

    # ------------------------------------------------------------------
    # Compute per-(protein, EC) rank  +  top-20 table
    # ------------------------------------------------------------------
    rows      = []
    top20_rows = []

    for _, pair in gold.iterrows():
        protein_id = pair["protein_id"]
        ec_number  = pair["ec_number"]
        ec_cav_id  = pair["ec_cav_id"]

        idx = id_to_idx.get(protein_id)
        col = cav_id_to_col.get(ec_cav_id)

        if idx is None or col is None:
            if col is None:
                logger.warning(f"  {ec_cav_id} not among loaded CAVs — skipping")
            continue

        scores     = score_matrix[idx]          # (n_cavs,)
        true_score = scores[col]
        # Rank 1 = highest scoring CAV
        rank = int((scores > true_score).sum()) + 1
        pct_rank = rank / n_cavs * 100

        rows.append({
            "protein_id":  protein_id,
            "ec_number":   ec_number,
            "ec_cav_id":   ec_cav_id,
            "true_score":  float(true_score),
            "rank":        rank,
            "n_cavs":      n_cavs,
            "pct_rank":    pct_rank,
        })

    results = pd.DataFrame(rows)
    results_file = out_dir / "ec_specificity_ranks.tsv"
    results.to_csv(results_file, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Saved rank results to {results_file}")

    # ------------------------------------------------------------------
    # Top-20 table: for each val protein, top 20 EC CAVs by score
    # ------------------------------------------------------------------
    top20_rows = []
    seen_proteins = set()
    for protein_id in gold["protein_id"].unique():
        idx = id_to_idx.get(protein_id)
        if idx is None:
            continue
        scores       = score_matrix[idx]
        top20_cols   = np.argsort(scores)[::-1][:20]
        true_cavs    = protein_true_cavs.get(protein_id, set())
        for rank_i, col in enumerate(top20_cols, start=1):
            ec_cav_id  = cav_ids[col]
            ec_number  = ec_cav_id.replace("ecNo_", "").replace("-", ".")
            top20_rows.append({
                "protein_id":    protein_id,
                "ec_cav_id":     ec_cav_id,
                "ec_number":     ec_number,
                "score":         float(scores[col]),
                "true_positive": ec_cav_id in true_cavs,
                "rank":          rank_i,
            })

    top20 = pd.DataFrame(top20_rows)
    top20_file = out_dir / "ec_specificity_top20.tsv"
    top20.to_csv(top20_file, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Saved top-20 table to {top20_file}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"EC CAV specificity — {len(results)} (protein, EC) pairs")
    print(f"  Total EC CAVs scored against : {n_cavs}")
    print(f"  Median rank of true EC       : {results['rank'].median():.0f} / {n_cavs}")
    print(f"  % pairs with rank = 1        : {(results['rank'] == 1).mean()*100:.1f}%")
    print(f"  % pairs with rank ≤ 5        : {(results['rank'] <= 5).mean()*100:.1f}%")
    print(f"  % pairs with rank ≤ 10       : {(results['rank'] <= 10).mean()*100:.1f}%")
    print(f"  % pairs with rank ≤ top 1%   : {(results['pct_rank'] <= 1).mean()*100:.1f}%")
    print(f"{'='*55}")
    print(results[["protein_id", "ec_number", "rank", "pct_rank"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Figure: rank distribution
    # ------------------------------------------------------------------
    RCPARAMS = {"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10}
    plt.rcParams.update(RCPARAMS)

    ranks = results["rank"].values

    # Cumulative recall vs rank threshold
    max_rank_show = min(50, n_cavs)
    thresholds    = np.arange(1, max_rank_show + 1)
    cum_recall    = [(ranks <= t).mean() for t in thresholds]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, cum_recall, color="#2166ac", lw=2)
    ax.axhline(1.0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel(f"Rank threshold (out of {n_cavs} EC CAVs)")
    ax.set_ylabel("Fraction of val pairs correctly ranked")
    ax.set_xlim(1, max_rank_show)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    p = out_dir / "fig_ec_specificity_rank.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")

    # Histogram of ranks (clipped to first 50 for readability)
    fig, ax = plt.subplots(figsize=(6, 4))
    clip = ranks[ranks <= max_rank_show]
    ax.hist(clip, bins=range(1, max_rank_show + 2), color="#2166ac",
            edgecolor="white", lw=0.4, align="left")
    ax.set_xlabel(f"Rank of true EC (out of {n_cavs})")
    ax.set_ylabel("Number of val proteins")
    ax.set_xlim(0.5, max_rank_show + 0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if len(ranks) > len(clip):
        ax.set_title(f"(showing ranks ≤ {max_rank_show}; "
                     f"{len(ranks)-len(clip)} pairs beyond range)")
    fig.tight_layout()
    p = out_dir / "fig_ec_specificity_hist.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")


if __name__ == "__main__":
    main()
