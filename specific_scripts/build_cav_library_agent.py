#!/usr/bin/env python3
"""
Census-backed Gemini agent that builds a CAV library from any CellxGene dataset.

Queries obs metadata from the CellxGene Census (no h5ad download needed),
then uses an LLM agent to decide which concepts to train and generate spans
files (soma_joinid lists) for pos/neg/val splits.

Usage
-----
# All cell types, one-vs-rest
python specific_scripts/build_cav_library_agent.py \
    --dataset-id 53d208b0-2cfd-4366-9866-c3c6114081bc \
    --prompt "Create one-vs-rest CAVs for all cell types. \
              Skip any cell type with fewer than 200 cells. \
              Use max 1000 cells per group." \
    --api-key $GEMINI_API_KEY

# Free-text concept
python specific_scripts/build_cav_library_agent.py \
    --dataset-id 53d208b0-2cfd-4366-9866-c3c6114081bc \
    --prompt "Create a CAV for young (20-30 year old) vs old (70-80 year old) cells." \
    --api-key $GEMINI_API_KEY

After this runs:
  cav_library/<dataset_id>/
    spans/
      cell_type__T_cell/
        pos.txt        <- soma_joinids for training positives
        neg.txt        <- soma_joinids for training negatives
        val_pos.txt    <- held-out positives for validation
        val_neg.txt    <- held-out negatives for validation
      cell_type__stromal_cell/
        ...
    cav_plan.json      <- full plan with concept descriptions and rationale
"""

import argparse
import json
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_OBS = None       # pd.DataFrame — obs indexed by soma_joinid (string)
_OUT_DIR = None   # Path


def _load_obs(dataset_id, census_version="stable"):
    global _OBS
    if _OBS is None:
        import cellxgene_census
        logger.info(f"Querying CellxGene Census (version={census_version}) "
                    f"for dataset_id={dataset_id} ...")
        census = cellxgene_census.open_soma(census_version=census_version)
        value_filter = f"dataset_id == '{dataset_id}'"
        logger.info(f"  Census query: {value_filter}")
        _OBS = (
            census["census_data"]["homo_sapiens"]
            .obs.read(value_filter=value_filter)
            .concat()
            .to_pandas()
            .set_index("soma_joinid")
        )
        _OBS.index = _OBS.index.astype(str)
        census.close()
        logger.info(f"  {len(_OBS)} cells, {len(_OBS.columns)} obs columns")
    return _OBS


# ===========================================================================
# Tools the agent can call
# ===========================================================================

def get_dataset_summary() -> str:
    """
    Return a high-level summary of the dataset: total cell count and a list
    of all obs column names with their data types and number of unique values.
    Call this first to understand what metadata is available.
    """
    obs = _OBS
    rows = []
    for col in obs.columns:
        n_unique = obs[col].nunique()
        dtype = str(obs[col].dtype)
        rows.append({"column": col, "dtype": dtype, "n_unique": n_unique})
    summary = {
        "n_cells": len(obs),
        "columns": rows
    }
    return json.dumps(summary, indent=2)


def get_column_details(column: str, max_values: int = 200) -> str:
    """
    Return the value counts for a specific obs column, sorted by cell count
    descending.  Use this to see which categories exist and how many cells
    belong to each — essential for deciding whether a category meets the
    minimum-cell threshold and for constructing positive/negative splits.

    Args:
        column:     Name of the obs column to inspect.
        max_values: Maximum number of unique values to return (default 200).
                    The top max_values by cell count are returned when there
                    are more unique values than this limit.
    """
    obs = _OBS
    if column not in obs.columns:
        return json.dumps({"error": f"Column '{column}' not found. "
                           f"Available: {obs.columns.tolist()}"})
    counts = obs[column].value_counts().head(max_values)
    result = {
        "column": column,
        "n_unique": int(obs[column].nunique()),
        "n_cells_total": len(obs),
        "value_counts": {str(k): int(v) for k, v in counts.items()}
    }
    return json.dumps(result, indent=2)


def create_cav_spans(
    concept_name: str,
    positive_column: str = None,
    positive_values: list = None,
    negative_column: str = None,
    negative_values: list = None,
    positive_filters: dict = None,
    negative_filters: dict = None,
    n_per_group: int = 1000,
    min_cells: int = 200,
    seed: int = 42
) -> str:
    """
    Sample cells for a CAV concept and write pos/neg/val spans files.
    Each spans file contains one soma_joinid per line.

    Writes four files under <out_dir>/spans/<concept_name>/:
        pos.txt      — training positives
        neg.txt      — training negatives
        val_pos.txt  — held-out positives  (~20% of pos, min 1)
        val_neg.txt  — held-out negatives  (~20% of neg, min 1)

    Returns an error (does NOT write files) if either the positive or
    negative group has fewer cells than min_cells before sampling.

    Args:
        concept_name:      Short identifier, e.g. "cell_type__T_cell" or
                           "macrophage__colorectal__tumor".
                           Use double underscores to separate fields/values.
        positive_filters:  Dict mapping obs column names to lists of acceptable
                           values.  ALL conditions must be met (AND logic).
                           Use this for multi-column criteria, e.g.:
                             {"cell_type": ["macrophage"],
                              "tissue":    ["colorectal"],
                              "disease":   ["tumor"]}
                           Takes precedence over positive_column/positive_values
                           when provided.
        positive_column:   obs column for single-column positive selection
                           (legacy; ignored when positive_filters is given).
        positive_values:   Values for single-column positive selection
                           (legacy; ignored when positive_filters is given).
        negative_filters:  Dict mapping obs column names to lists of acceptable
                           values for negatives (AND logic).  Takes precedence
                           over negative_column/negative_values when provided.
        negative_column:   obs column for single-column negative selection
                           (legacy; ignored when negative_filters is given).
        negative_values:   Values for single-column negative selection.
                           Pass ["__all_others__"] to use all cells not
                           matching the positive criteria as negatives.
        n_per_group:       Max cells to sample per group for training
                           (default 1000).  Actual count may be lower.
        min_cells:         Minimum cells required in both pos and neg groups
                           before sampling.  Returns an error and skips file
                           creation if not met (default 200).
        seed:              Random seed for reproducibility (default 42).
    """
    obs = _OBS
    rng = np.random.default_rng(seed)

    # ---- Positive cells ----
    if positive_filters is not None:
        pos_mask = pd.Series(True, index=obs.index)
        for col, vals in positive_filters.items():
            if col not in obs.columns:
                return json.dumps({"error": f"Column '{col}' not found in obs."})
            pos_mask &= obs[col].isin(vals)
    elif positive_column is not None and positive_values is not None:
        pos_mask = obs[positive_column].isin(positive_values)
    else:
        return json.dumps({"error": "Provide either positive_filters or "
                                    "both positive_column and positive_values."})
    pos_idx = obs.index[pos_mask].tolist()

    # ---- Negative cells ----
    all_others = (negative_values == ["__all_others__"])
    if negative_filters is not None:
        neg_mask = pd.Series(True, index=obs.index)
        for col, vals in negative_filters.items():
            if col not in obs.columns:
                return json.dumps({"error": f"Column '{col}' not found in obs."})
            neg_mask &= obs[col].isin(vals)
        neg_idx = obs.index[neg_mask].tolist()
    elif all_others:
        neg_mask = ~pos_mask
        neg_idx = obs.index[neg_mask].tolist()
    elif negative_column is not None and negative_values is not None:
        neg_mask = obs[negative_column].isin(negative_values)
        neg_idx = obs.index[neg_mask].tolist()
    else:
        return json.dumps({"error": "Provide either negative_filters, "
                                    "negative_values=['__all_others__'], or "
                                    "both negative_column and negative_values."})

    # ---- Minimum-cell guard ----
    if len(pos_idx) < min_cells:
        return json.dumps({
            "error": f"Skipping '{concept_name}': only {len(pos_idx)} positive "
                     f"cells found (min_cells={min_cells})."
        })
    if len(neg_idx) < min_cells:
        return json.dumps({
            "error": f"Skipping '{concept_name}': only {len(neg_idx)} negative "
                     f"cells found (min_cells={min_cells})."
        })

    # ---- Stratified negative sampling for __all_others__ (single-column only) ----
    if all_others and positive_column is not None:
        neg_obs   = obs.loc[neg_idx]
        groups    = neg_obs.groupby(positive_column, observed=True)
        n_groups  = len(groups)
        n_total   = min(int(n_per_group * 1.25) + 1, len(neg_idx))
        per_group = max(1, n_total // n_groups)
        stratified = []
        for _, grp in groups:
            k = min(per_group, len(grp))
            stratified.extend(
                rng.choice(grp.index.tolist(), size=k, replace=False).tolist()
            )
        if len(stratified) > n_total:
            stratified = rng.choice(
                stratified, size=n_total, replace=False
            ).tolist()
        rng.shuffle(stratified)
        neg_idx = stratified
    else:
        rng.shuffle(neg_idx)

    rng.shuffle(pos_idx)

    # ---- Train / val split (80 / 20) ----
    # Reserve val cells FIRST, then take up to n_per_group for training
    # from what remains — avoids the case where n_per_group >= len(ids)
    # leaving nothing for validation.
    def split_train_val(ids, n_train):
        total = len(ids)
        n_val   = max(1, total // 5)          # 20% of what's available
        n_train = min(n_train, total - n_val)  # train gets the rest, up to n_train
        return ids[:n_train], ids[n_train: n_train + n_val]

    pos_train, pos_val = split_train_val(pos_idx, n_per_group)
    neg_train, neg_val = split_train_val(neg_idx, n_per_group)

    # ---- Write files ----
    spans_dir = _OUT_DIR / "spans" / concept_name
    spans_dir.mkdir(parents=True, exist_ok=True)

    def write_ids(path, ids):
        with open(path, 'w') as f:
            f.write('\n'.join(map(str, ids)) + '\n')

    write_ids(spans_dir / "pos.txt",     pos_train)
    write_ids(spans_dir / "neg.txt",     neg_train)
    write_ids(spans_dir / "val_pos.txt", pos_val)
    write_ids(spans_dir / "val_neg.txt", neg_val)

    result = {
        "concept_name":     concept_name,
        "n_pos_train":      len(pos_train),
        "n_neg_train":      len(neg_train),
        "n_pos_val":        len(pos_val),
        "n_neg_val":        len(neg_val),
        "pos_file":         str(spans_dir / "pos.txt"),
        "neg_file":         str(spans_dir / "neg.txt"),
        "val_pos_file":     str(spans_dir / "val_pos.txt"),
        "val_neg_file":     str(spans_dir / "val_neg.txt"),
        "positive_filters": positive_filters,
        "positive_column":  positive_column,
        "positive_values":  positive_values,
        "negative_filters": negative_filters,
        "negative_column":  negative_column,
        "negative_values":  negative_values,
    }
    logger.info(
        f"  Spans '{concept_name}': "
        f"train {len(pos_train)} pos / {len(neg_train)} neg  |  "
        f"val {len(pos_val)} pos / {len(neg_val)} neg"
    )
    return json.dumps(result, indent=2)


def record_cav_plan(plan_entries: list) -> str:
    """
    Save the full CAV plan as a JSON file in the output directory.
    Call this once at the end after all create_cav_spans calls, passing
    the complete list of every concept you created or attempted.

    Args:
        plan_entries: List of dicts, one per concept.  Each should include at
                      minimum: concept_name, rationale, positive_values,
                      negative_values, field_name.  Skipped concepts should
                      be noted with a 'skipped' key and reason.
    """
    plan_path = _OUT_DIR / "cav_plan.json"
    with open(plan_path, 'w') as f:
        json.dump(plan_entries, f, indent=2)
    logger.info(f"Saved CAV plan: {plan_path} ({len(plan_entries)} entries)")
    return json.dumps({"saved": str(plan_path), "n_entries": len(plan_entries)})


# ===========================================================================
# System prompt
# ===========================================================================

SYSTEM_PROMPT = """You are a computational biology expert building a CAV
(Concept Activation Vector) library for single-cell RNA-seq data.

The dataset obs is indexed by soma_joinid (string). All spans files will
contain soma_joinid values, one per line, split into train (pos.txt, neg.txt)
and held-out validation sets (val_pos.txt, val_neg.txt).

Workflow:
1. Call get_dataset_summary to see all available obs columns.
2. Call get_column_details on EVERY column relevant to the user's request
   to get exact category names and cell counts BEFORE deciding what to create.
   Do not assume column names or values — always inspect first.
3. For each qualifying concept, call create_cav_spans.
4. Call record_cav_plan once at the end with an entry for every concept
   you created AND every concept you skipped (with a reason).

IMPORTANT: You must NOT stop or return until you have:
  (a) inspected all relevant columns,
  (b) attempted create_cav_spans for every qualifying combination, AND
  (c) called record_cav_plan.
Do not return early. Do not say you are done before calling record_cav_plan.

Rules:
- Use the ACTUAL column names and values found in the data. If the user's
  prompt mentions "macrophage" but the data has "mononuclear phagocyte", use
  "mononuclear phagocyte". Adapt all names to what the data actually contains.
- Always call get_column_details before using any column — never guess values.
- For one-vs-rest: pass ["__all_others__"] as negative_values; the tool
  handles stratified sampling across all remaining categories automatically.
- For multi-column positive criteria (e.g. cell_type AND tissue AND disease),
  use the positive_filters parameter: a dict mapping each column name to lists
  of acceptable values. ALL conditions are ANDed together.
  Example: positive_filters={"cell_type": ["mononuclear phagocyte"],
                              "tissue":    ["colorectal"],
                              "disease":   ["colorectal adenocarcinoma"]}
  You MUST call get_column_details to confirm exact column names and values.
- Concept names: use double underscores to join field values, no spaces,
  replace spaces with underscores
  (e.g. "mononuclear_phagocyte__colorectal__tumor").
- If a concept has an error returned by create_cav_spans (e.g. too few cells),
  record it as skipped in the plan — do not retry with the same values.
- Be systematic: iterate through EVERY qualifying combination, not just the
  first few. Do not stop partway through a set of combinations."""


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Census-backed Gemini agent to build a CAV library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--dataset-id', required=True,
        help='CellxGene Census dataset_id (e.g. 53d208b0-2cfd-4366-9866-c3c6114081bc).'
    )
    parser.add_argument(
        '--prompt', required=True,
        help=(
            'Natural language instruction for the agent.  Include any '
            'constraints you want enforced, e.g.: '
            '"Create one-vs-rest CAVs for all cell types. '
            'Skip any cell type with fewer than 200 cells. '
            'Use max 1000 cells per group."'
        )
    )
    parser.add_argument(
        '--out', default=None,
        help='Output directory (default: cav_library/<dataset_id>/).'
    )
    parser.add_argument(
        '--api-key', default=None,
        help='Gemini API key (or set GEMINI_API_KEY env var).'
    )
    parser.add_argument(
        '--model', default='gemini-2.5-flash',
        help='Gemini model to use (default: gemini-2.5-flash).'
    )
    parser.add_argument(
        '--census-version', default='stable',
        help='CellxGene Census version (default: stable).'
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Provide --api-key or set GEMINI_API_KEY env var.")

    global _OBS, _OUT_DIR
    _OUT_DIR = Path(args.out or f"cav_library/{args.dataset_id}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    _load_obs(args.dataset_id, census_version=args.census_version)

    from chatlas import ChatGoogle

    chat = ChatGoogle(
        api_key=api_key,
        model=args.model,
        system_prompt=SYSTEM_PROMPT,
    )
    chat.register_tool(get_dataset_summary)
    chat.register_tool(get_column_details)
    chat.register_tool(create_cav_spans)
    chat.register_tool(record_cav_plan)

    print("\n" + "=" * 60)
    print(f"Dataset : {args.dataset_id}")
    print(f"Output  : {_OUT_DIR}")
    print(f"Prompt  : {args.prompt}")
    print("=" * 60 + "\n")

    response = chat.chat(args.prompt)

    print("\n" + "=" * 60)
    print("Agent complete.")
    print("=" * 60)
    print(response)

    # ---- Summary ----
    plan_path = _OUT_DIR / "cav_plan.json"
    if plan_path.exists():
        with open(plan_path) as f:
            plan = json.load(f)
        created = [e for e in plan if not e.get("skipped")]
        skipped = [e for e in plan if e.get("skipped")]
        print(f"\n{len(created)} CAV concepts created, {len(skipped)} skipped.")
        spans_dirs = sorted((_OUT_DIR / "spans").iterdir())
        for d in spans_dirs[:30]:
            n_pos = sum(1 for _ in open(d / "pos.txt"))
            n_neg = sum(1 for _ in open(d / "neg.txt"))
            n_vp  = sum(1 for _ in open(d / "val_pos.txt"))
            n_vn  = sum(1 for _ in open(d / "val_neg.txt"))
            print(f"  {d.name:50s}  "
                  f"train {n_pos:4d}+{n_neg:4d}  val {n_vp:3d}+{n_vn:3d}")
        if len(spans_dirs) > 30:
            print(f"  ... and {len(spans_dirs) - 30} more")


if __name__ == '__main__':
    main()
