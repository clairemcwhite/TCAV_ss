#!/usr/bin/env python3
"""
Extract per-split protein ID lists from proFAB gaac feature files.

For each group × polarity combination present in --data-dir, reads protein IDs
from {group}_{polarity}_gaac.txt (first column) and uses integer index files to
write train/test/validation .span files (one ID per line).

Groups: temporal, target, random
Polarities: positive, negative
Splits: train, test, validation (validation skipped if indices file absent)

Output: {group}_{polarity}_{split}.span in --data-dir

Usage
-----
python specific_scripts/format_proFAB.py --data-dir /path/to/profab/data
python specific_scripts/format_proFAB.py --data-dir . --groups temporal target
"""

import argparse
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPLITS = ["train", "test", "validation"]
POLARITIES = ["positive", "negative"]


def read_ids(gaac_path: Path) -> list[str]:
    ids = []
    with open(gaac_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line.split()[0])
    return ids


def read_indices(indices_path: Path) -> list[int]:
    with open(indices_path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def write_span(ids: list[str], out_path: Path) -> None:
    with open(out_path, "w") as f:
        f.write("\n".join(ids) + "\n")
    logger.info(f"Wrote {len(ids)} IDs → {out_path}")


def detect_groups(data_dir: Path) -> list[str]:
    groups = set()
    for polarity in POLARITIES:
        for p in data_dir.glob(f"*_{polarity}_train_indices.txt"):
            group = p.stem.removesuffix(f"_{polarity}_train_indices")
            groups.add(group)
    return sorted(groups)


def process(data_dir: Path, groups: list[str], max_n: int | None, seed: int) -> None:
    rng = random.Random(seed)
    for group in groups:
        for polarity in POLARITIES:
            gaac_path = data_dir / f"{group}_{polarity}_gaac.txt"
            if not gaac_path.exists():
                logger.warning(f"Missing gaac file, skipping: {gaac_path}")
                continue

            ids = read_ids(gaac_path)
            logger.info(f"{group}/{polarity}: loaded {len(ids)} IDs from {gaac_path.name}")

            for split in SPLITS:
                indices_path = data_dir / f"{group}_{polarity}_{split}_indices.txt"
                if not indices_path.exists():
                    logger.info(f"  No indices file for {split}, skipping: {indices_path.name}")
                    continue

                indices = read_indices(indices_path)
                split_ids = [ids[i] for i in indices]

                if max_n is not None and len(split_ids) > max_n:
                    split_ids = rng.sample(split_ids, max_n)
                    logger.info(f"  Subsampled {group}/{polarity}/{split} to {max_n}")

                n_suffix = f"_{max_n}" if max_n is not None else ""
                out_path = data_dir / f"{group}_{polarity}_{split}{n_suffix}.span"
                write_span(split_ids, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-split protein ID lists from proFAB gaac files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing gaac and indices files (default: current directory).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        metavar="GROUP",
        help="Groups to process (default: auto-detect from *_train_indices.txt files).",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        metavar="N",
        help="Randomly subsample each split to at most N IDs. Adds _{N} to output filenames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling (default: 42).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        parser.error(f"--data-dir does not exist: {data_dir}")

    groups = args.groups if args.groups else detect_groups(data_dir)
    if not groups:
        parser.error(f"No groups detected in {data_dir}. Check for *_positive_train_indices.txt files.")

    logger.info(f"Processing groups: {groups}")
    process(data_dir, groups, args.max_n, args.seed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
