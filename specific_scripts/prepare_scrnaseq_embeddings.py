#!/usr/bin/env python3
"""
Prepare single-cell RNA-seq embeddings using a local Geneformer model.

Each cell becomes one "sample".  Output matches the .pkl + .info format
used by prepare_embeddings.py and train_cav_from_embeddings.py.

Expects:
  - AnnData (.h5ad) in cell × gene orientation
  - Gene names in var_names are already Ensembl IDs (ENSG...)

Preprocessing applied automatically:
  1. normalize_total (target 10 000 counts per cell) then log1p.
  2. Rank genes by expression per cell and tokenize using the model's
     token_dictionary.pkl.

Output
------
  <out>       — pickle dict with keys:
      "sequence_embeddings"  np.ndarray (n_cells, hidden_dim)
                             mean-pooled over expressed gene tokens
      "aa_embeddings"        np.ndarray (n_cells, max_input_size, hidden_dim)
                             per-gene token embeddings (only if --save-gene-embs)
  <out>.info  — plain text, one cell ID per line (matches row order in pkl)

Examples
--------
# Embed all cells:
python specific_scripts/prepare_scrnaseq_embeddings.py \\
    --input  data/endometrium_c1.h5ad \\
    --model  /path/to/geneformer \\
    --out    embeddings/scrnaseq/c1.pkl

# Restrict to annotated cells (unnamed first column of CSV = cell IDs):
python specific_scripts/prepare_scrnaseq_embeddings.py \\
    --input    data/endometrium_c1.h5ad \\
    --metadata data/GSE111976_summary_C1_day_donor_ctype.csv \\
    --model    /path/to/geneformer \\
    --out      embeddings/scrnaseq/c1.pkl
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scanpy as sc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def load_h5ad(input_path: Path) -> sc.AnnData:
    adata = sc.read_h5ad(input_path)
    logger.info(f"Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def filter_to_metadata(adata: sc.AnnData, metadata_path: Path, cell_col: str | None) -> sc.AnnData:
    """
    Restrict AnnData to cells listed in the metadata file.

    By default uses the row index of the metadata CSV as cell IDs — this
    matches the GEO convention used by GSE111976 where the unnamed first
    column holds cell barcodes/names.
    """
    meta = pd.read_csv(metadata_path, index_col=0)

    if cell_col:
        if cell_col not in meta.columns:
            raise ValueError(
                f"--cell-col '{cell_col}' not found in {metadata_path}. "
                f"Available: {list(meta.columns)}"
            )
        meta_cells = set(meta[cell_col].astype(str))
        logger.info(f"Using metadata column '{cell_col}' as cell IDs")
    else:
        meta_cells = set(meta.index.astype(str))
        logger.info("Using metadata row index as cell IDs")

    mask = adata.obs_names.isin(meta_cells)
    n_before = adata.n_obs
    adata = adata[mask].copy()
    logger.info(f"Filtered: {n_before} → {adata.n_obs} cells")

    if adata.n_obs == 0:
        raise ValueError(
            "No cells remain after filtering. Check that obs_names in the h5ad "
            "match the cell IDs in the metadata file."
        )
    return adata


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(adata: sc.AnnData, target_sum: float = 1e4) -> np.ndarray:
    """
    Normalize counts per cell to target_sum then log1p.
    Returns a dense (n_cells, n_genes) float32 array.
    Does not modify adata in place.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Geneformer tokenization
# ---------------------------------------------------------------------------

def load_token_dict(model_dir: Path) -> dict[str, int]:
    """
    Load a Geneformer token dictionary from the model directory.

    Searches the model directory and one level up for any file matching
    token_dictionary*.pkl, so it handles naming variants such as:
      token_dictionary.pkl
      token_dictionary_gc104M.pkl
      token_dictionary_gc95M.pkl
    """
    search_dirs = [model_dir, model_dir.parent]
    for search_dir in search_dirs:
        matches = sorted(search_dir.glob('token_dictionary*.pkl'))
        if matches:
            chosen = matches[0]
            if len(matches) > 1:
                logger.warning(
                    f"Multiple token dictionaries found: {[m.name for m in matches]}. "
                    f"Using {chosen.name}. Pass --token-dict to override."
                )
            with open(chosen, 'rb') as f:
                d = pickle.load(f)
            logger.info(f"Loaded token dictionary: {len(d)} genes from {chosen}")
            return d
    raise FileNotFoundError(
        f"No token_dictionary*.pkl found in {model_dir} or {model_dir.parent}. "
        "Check your model directory."
    )


def tokenize_cell(
    expression: np.ndarray,
    gene_ids: list[str],
    token_dict: dict[str, int],
    max_len: int,
) -> list[int]:
    """
    Rank genes by normalized expression (descending) and return token IDs.
    Genes with zero expression or absent from the token dictionary are skipped.
    """
    nonzero = expression > 0
    if not nonzero.any():
        return []

    expr_nz  = expression[nonzero]
    genes_nz = [gene_ids[i] for i, keep in enumerate(nonzero) if keep]
    order    = np.argsort(-expr_nz)

    token_ids: list[int] = []
    for idx in order:
        tok = token_dict.get(genes_nz[idx])
        if tok is not None:
            token_ids.append(tok)
            if len(token_ids) >= max_len:
                break
    return token_ids


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def embed_cells(
    X: np.ndarray,
    gene_ids: list[str],
    token_dict: dict[str, int],
    model_path: Path,
    batch_size: int,
    max_input_size: int,
    device: str,
    save_gene_embs: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Run all cells through the model and return embeddings.

    Returns
    -------
    sequence_embeddings : (n_cells, hidden_dim)
        Mean-pooled over expressed gene tokens.
    aa_embeddings : (n_cells, max_input_size, hidden_dim) or None
        Per-gene token hidden states (only when save_gene_embs=True).
    """
    from transformers import AutoModel

    dev = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    if str(dev) != device:
        logger.info(f"Requested {device} but using {dev}")

    logger.info(f"Loading model from {model_path}")
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval()
    model.to(dev)

    hidden_dim = model.config.hidden_size
    n_cells    = X.shape[0]
    logger.info(f"hidden_dim={hidden_dim}, embedding {n_cells} cells")

    seq_embs = np.zeros((n_cells, hidden_dim), dtype=np.float32)
    aa_embs  = (
        np.zeros((n_cells, max_input_size, hidden_dim), dtype=np.float32)
        if save_gene_embs else None
    )

    for batch_start in range(0, n_cells, batch_size):
        batch_end  = min(batch_start + batch_size, n_cells)
        batch_size_ = batch_end - batch_start

        # Tokenize
        token_lists = [
            tokenize_cell(X[ci], gene_ids, token_dict, max_input_size)
            for ci in range(batch_start, batch_end)
        ]

        if all(len(t) == 0 for t in token_lists):
            logger.warning(
                f"Cells {batch_start}–{batch_end}: no tokens found. "
                "Check that gene IDs match the token dictionary."
            )
            continue

        pad_len   = max(len(t) for t in token_lists) or 1
        input_ids = torch.zeros(batch_size_, pad_len, dtype=torch.long)
        attn_mask = torch.zeros(batch_size_, pad_len, dtype=torch.long)
        for i, toks in enumerate(token_lists):
            if toks:
                input_ids[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
                attn_mask[i, :len(toks)] = 1

        input_ids = input_ids.to(dev)
        attn_mask = attn_mask.to(dev)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)

        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Mean-pool over non-padding positions
        mask_exp = attn_mask.unsqueeze(-1).float()
        pooled   = ((hidden * mask_exp).sum(dim=1) /
                    mask_exp.sum(dim=1).clamp(min=1e-9))
        seq_embs[batch_start:batch_end] = pooled.cpu().numpy()

        if aa_embs is not None:
            hs  = hidden.cpu().numpy()
            col = min(hs.shape[1], max_input_size)
            aa_embs[batch_start:batch_end, :col, :] = hs[:, :col, :]

        logger.info(f"  Embedded cells {batch_start + 1}–{batch_end} / {n_cells}")

    return seq_embs, aa_embs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Embed scRNA-seq cells with a local Geneformer model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--input', required=True,
                        help='AnnData file (.h5ad), cell × gene, gene names as Ensembl IDs.')
    parser.add_argument('--model', required=True,
                        help='Path to local Geneformer model directory (HuggingFace format).')
    parser.add_argument('--out', required=True,
                        help='Output .pkl path (e.g. embeddings/scrnaseq/c1.pkl).')
    parser.add_argument('--metadata',
                        help='CSV with cell annotations. When supplied, only cells '
                             'listed here are embedded. Cell IDs are taken from the '
                             'row index (unnamed first column) by default — correct '
                             'for GSE111976 summary CSVs.')
    parser.add_argument('--cell-col',
                        help='Named column in --metadata that holds cell IDs. '
                             'Defaults to the row index (unnamed first column).')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Cells per GPU forward pass (default: 16).')
    parser.add_argument('--max-input-size', type=int, default=2048,
                        help='Maximum number of gene tokens per cell (default: 2048).')
    parser.add_argument('--device', default='cuda',
                        help='Compute device: cuda or cpu (default: cuda, '
                             'falls back to cpu if CUDA is unavailable).')
    parser.add_argument('--save-gene-embs', action='store_true',
                        help='Also save per-gene token embeddings as aa_embeddings. '
                             'Shape: (n_cells, max_input_size, hidden_dim). '
                             'WARNING: memory-intensive for large datasets.')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load
    adata = load_h5ad(Path(args.input))

    # 2. Filter to metadata (optional)
    if args.metadata:
        adata = filter_to_metadata(adata, Path(args.metadata), args.cell_col)

    cell_ids = list(adata.obs_names)
    gene_ids = list(adata.var_names)

    # 3. Normalize
    logger.info("Normalizing: normalize_total (10 000) + log1p")
    X = normalize(adata)

    # 4. Token dictionary
    token_dict = load_token_dict(Path(args.model))

    # Quick sanity check: how many genes overlap with token dictionary
    n_overlap = sum(1 for g in gene_ids if g in token_dict)
    logger.info(f"Gene → token overlap: {n_overlap} / {len(gene_ids)} genes in token dictionary")
    if n_overlap == 0:
        raise ValueError(
            "No genes match the token dictionary. "
            "Verify that var_names are Ensembl IDs matching those in token_dictionary.pkl."
        )

    # 5. Embed
    if args.save_gene_embs:
        logger.warning(
            f"--save-gene-embs: aa_embeddings will be "
            f"({len(cell_ids)}, {args.max_input_size}, hidden_dim) — ensure sufficient RAM."
        )

    seq_embs, aa_embs = embed_cells(
        X=X,
        gene_ids=gene_ids,
        token_dict=token_dict,
        model_path=Path(args.model),
        batch_size=args.batch_size,
        max_input_size=args.max_input_size,
        device=args.device,
        save_gene_embs=args.save_gene_embs,
    )

    # 6. Save .pkl + .pkl.info
    embeddings: dict = {'sequence_embeddings': seq_embs}
    if aa_embs is not None:
        embeddings['aa_embeddings'] = aa_embs

    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved pkl: {out_path}")

    info_path = Path(str(out_path) + '.info')
    with open(info_path, 'w') as f:
        f.write('\n'.join(cell_ids) + '\n')
    logger.info(f"Saved {len(cell_ids)} cell IDs: {info_path}")

    summary = f"sequence_embeddings: {seq_embs.shape}"
    if aa_embs is not None:
        summary += f", aa_embeddings: {aa_embs.shape}"
    logger.info(f"Done. {summary}")


if __name__ == '__main__':
    main()
