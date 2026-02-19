"""
Data loading utilities for TCAV.

Embedding files
---------------
.pkl  — dict with keys:
    "sequence_embeddings": np.ndarray, shape (n_samples, hidden_dim)
    "aa_embeddings":       np.ndarray, shape (n_samples, max_seq_len, hidden_dim)
                           (padded; use seq_len from spans or the array's dim 1)

.pkl.info — plain text, one sample ID per line (no header).
    Row index in the arrays corresponds to line index in this file.

Spans files
-----------
Tab-separated, one entry per line, no header.
Each line is one of:
    accession                       → whole-sequence mode (no span)
    accession  TAB  pos             → single-position mode (PTM etc.)
    accession  TAB  start  TAB  end → window mode (half-open: [start, end))

The same accession may appear on multiple lines — each line produces a
separate embedding vector. This allows multiple negative windows from one
protein, or multiple PTM sites from one sequence.

Lines beginning with '#' are treated as comments and ignored.
"""

import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding pkl loading
# ---------------------------------------------------------------------------

def load_embeddings_pkl(
    pkl_path: str,
    info_path: str
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load embedding pkl and companion info file.

    Args:
        pkl_path:  Path to .pkl file containing embedding arrays.
        info_path: Path to .pkl.info file with one sample ID per line.

    Returns:
        Tuple of (embeddings_dict, sample_ids)
        embeddings_dict has keys "sequence_embeddings" and/or "aa_embeddings".
    """
    with open(pkl_path, 'rb') as f:
        embeddings = pickle.load(f)

    sample_ids = []
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sample_ids.append(line)

    # Validate row counts
    for key, arr in embeddings.items():
        if len(arr) != len(sample_ids):
            raise ValueError(
                f"{key}: array has {len(arr)} rows but info file has "
                f"{len(sample_ids)} IDs"
            )

    logger.info(
        f"Loaded {len(sample_ids)} samples from {pkl_path} "
        f"(keys: {list(embeddings.keys())})"
    )
    return embeddings, sample_ids


# ---------------------------------------------------------------------------
# Spans file loading
# ---------------------------------------------------------------------------

SpanEntry = Optional[Tuple[int, ...]]  # None | (pos,) | (start, end)

# A spans list is a sequence of (accession, span) pairs.
# Multiple entries with the same accession are allowed and each produces
# a separate embedding vector (e.g. multiple negative windows per protein).
SpansList = List[Tuple[str, SpanEntry]]


def load_spans(spans_path: str) -> SpansList:
    """
    Parse a spans file into an ordered list of (accession, span) pairs.

    Multiple lines with the same accession are each kept as separate entries,
    allowing multiple windows or positions per protein.

    Each span is one of:
        None          — whole-sequence mode (no span columns)
        (pos,)        — single-position mode (one span column)
        (start, end)  — window mode, half-open (two span columns)
    """
    spans: SpansList = []

    with open(spans_path, 'r') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            accession = parts[0]

            if len(parts) == 1:
                spans.append((accession, None))
            elif len(parts) == 2:
                spans.append((accession, (int(parts[1]),)))
            elif len(parts) == 3:
                spans.append((accession, (int(parts[1]), int(parts[2]))))
            else:
                raise ValueError(
                    f"{spans_path}:{lineno}: expected 1–3 tab-separated columns, "
                    f"got {len(parts)}"
                )

    logger.info(f"Loaded {len(spans)} span entries from {spans_path}")
    return spans


# ---------------------------------------------------------------------------
# Span pooling
# ---------------------------------------------------------------------------

def pool_span(
    aa_embeddings_row: np.ndarray,
    span: SpanEntry,
    sequence_embedding_row: Optional[np.ndarray] = None,
    seq_len: Optional[int] = None
) -> np.ndarray:
    """
    Extract a fixed-size embedding vector from a single sample's token embeddings.

    Args:
        aa_embeddings_row:      Shape (max_seq_len, hidden_dim). May be padded.
        span:                   None | (pos,) | (start, end) from load_spans().
        sequence_embedding_row: Shape (hidden_dim,). Used for whole-sequence mode
                                if present; falls back to mean-pooling aa_embeddings.
        seq_len:                Actual (unpadded) sequence length. If None, uses
                                aa_embeddings_row.shape[0].

    Returns:
        1-D embedding vector of shape (hidden_dim,).
    """
    if seq_len is None:
        seq_len = aa_embeddings_row.shape[0]

    if span is None:
        # Whole-sequence mode
        if sequence_embedding_row is not None:
            return sequence_embedding_row
        return aa_embeddings_row[:seq_len].mean(axis=0)

    if len(span) == 1:
        # Single-position mode
        pos = span[0]
        if pos < 0 or pos >= seq_len:
            raise ValueError(
                f"Position {pos} out of range for sequence of length {seq_len}"
            )
        return aa_embeddings_row[pos]

    # Window mode
    start, end = span
    if start < 0 or end > seq_len or start >= end:
        raise ValueError(
            f"Span [{start}, {end}) invalid for sequence of length {seq_len}"
        )
    return aa_embeddings_row[start:end].mean(axis=0)


# ---------------------------------------------------------------------------
# Build pooled embedding matrix for CAV training
# ---------------------------------------------------------------------------

def build_embedding_matrix(
    embeddings: Dict[str, np.ndarray],
    sample_ids: List[str],
    spans: Optional[SpansList] = None
) -> np.ndarray:
    """
    Build a (n_entries, hidden_dim) matrix by pooling each span entry.

    Each entry in spans produces one row in the output matrix. The same
    accession may appear multiple times (e.g. multiple negative windows from
    one protein), yielding one row per entry.

    If spans is None, each sample in sample_ids is used once in
    whole-sequence mode.

    Args:
        embeddings:  Dict from load_embeddings_pkl().
        sample_ids:  Ordered list of IDs (row index = position in arrays).
        spans:       List of (accession, span) pairs from load_spans(),
                     or None for whole-sequence mode on all samples.

    Returns:
        2-D array of shape (n_entries, hidden_dim).
    """
    aa_emb = embeddings.get('aa_embeddings')
    seq_emb = embeddings.get('sequence_embeddings')

    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    vectors = []
    skipped = 0

    entries = spans if spans is not None else [(sid, None) for sid in sample_ids]

    for accession, span in entries:
        if accession not in id_to_idx:
            logger.warning(f"'{accession}' in spans not found in pkl — skipping")
            skipped += 1
            continue

        idx = id_to_idx[accession]
        row_aa = aa_emb[idx] if aa_emb is not None else None
        row_seq = seq_emb[idx] if seq_emb is not None else None

        if row_aa is None and span is not None:
            raise ValueError(
                f"Span specified for '{accession}' but pkl has no aa_embeddings"
            )
        if row_aa is None:
            if row_seq is None:
                raise ValueError(
                    "pkl has neither aa_embeddings nor sequence_embeddings"
                )
            vectors.append(row_seq)
        else:
            vectors.append(pool_span(row_aa, span, sequence_embedding_row=row_seq))

    if skipped:
        logger.warning(f"Skipped {skipped} accessions not found in pkl")

    if not vectors:
        raise ValueError("No valid samples found after matching spans to pkl")

    result = np.vstack(vectors)
    logger.info(f"Built embedding matrix: {result.shape}")
    return result
