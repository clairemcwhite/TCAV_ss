#!/usr/bin/env python3
"""
reformat_ec_goldstandard.py

Convert a wide EC-number CSV (id, seq, ec_number) to a long-format TSV
with one row per (protein, EC number) pair.

EC number normalisation:
  - Trailing wildcard levels are stripped:  4.2.3.-  →  4.2.3
  - Fully-specified numbers are kept as-is: 4.2.3.158 →  4.2.3.158
  - The CAV directory name replaces dots with hyphens and adds ecNo_ prefix:
      4.2.3     →  ecNo_4-2-3
      4.2.3.158 →  ecNo_4-2-3-158

Usage
-----
python specific_scripts/reformat_ec_goldstandard.py \\
    --input  data/ec_gold_standard.csv \\
    --output data/ec_gold_standard_long.tsv
"""

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_ec(raw: str) -> str:
    """Strip trailing wildcard levels from an EC number string.

    '4.2.3.-'   → '4.2.3'
    '4.2.3.158' → '4.2.3.158'
    '1.-.-.-'   → '1'
    """
    parts = raw.strip().split(".")
    while parts and parts[-1].strip() == "-":
        parts.pop()
    return ".".join(parts)


def ec_to_cav_id(ec_norm: str) -> str:
    """'4.2.3.158' → 'ecNo_4-2-3-158'"""
    return "ecNo_" + ec_norm.replace(".", "-")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  required=True,
                        help="Input CSV with columns: id, seq, ec_number")
    parser.add_argument("--output", required=True,
                        help="Output long-format TSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    # Rename id → protein_id for clarity
    df = df.rename(columns={"id": "protein_id"})

    # Explode comma-separated EC numbers into separate rows
    df["ec_number"] = df["ec_number"].astype(str).str.strip().str.strip('"')
    df["ec_number"] = df["ec_number"].str.split(",")
    df = df.explode("ec_number")
    df["ec_number"] = df["ec_number"].str.strip()
    df = df[df["ec_number"] != ""].reset_index(drop=True)
    logger.info(f"After exploding multi-EC rows: {len(df)} (protein, EC) pairs")

    # Normalize: strip trailing wildcard levels
    df["ec_number_raw"] = df["ec_number"]
    df["ec_number"]     = df["ec_number"].apply(normalize_ec)

    # Drop rows that collapsed to empty (shouldn't happen, but just in case)
    df = df[df["ec_number"] != ""].reset_index(drop=True)

    # Generate CAV directory ID
    df["ec_cav_id"] = df["ec_number"].apply(ec_to_cav_id)

    # Summary
    n_full     = (df["ec_number"].str.count(r"\.") == 3).sum()
    n_partial  = (df["ec_number"].str.count(r"\.") < 3).sum()
    logger.info(f"  Fully-specified (4-level) EC pairs : {n_full}")
    logger.info(f"  Partial (wildcard-trimmed) EC pairs: {n_partial}")
    logger.info(f"  Unique EC numbers                  : {df['ec_number'].nunique()}")
    logger.info(f"  Unique proteins                    : {df['protein_id'].nunique()}")

    # Write output — drop seq to keep it lightweight; keep raw for reference
    out = df[["protein_id", "ec_number_raw", "ec_number", "ec_cav_id"]]
    out.to_csv(args.output, sep="\t", index=False)
    logger.info(f"Written to {args.output}")

    # Preview
    print("\nFirst 10 rows:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
