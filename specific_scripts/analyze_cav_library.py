#!/usr/bin/env python3
"""
analyze_cav_library.py — Use an LLM to infer the structure of a CAV library.

Reads the list of CAV names from lib_dir/cavs/, sends them to Gemini, and
writes a library_structure.json describing:

  - How many semantic levels the naming convention has
  - What each level means (cell_type, tissue, disease, ...)
  - Which level to group/color by in plots
  - Which level encodes disease state (if any)
  - Which values in the disease level represent normal/control
  - Matched normal/disease pairs and unpaired concepts

The output JSON is consumed by cav_disease_viz.py and cav_hierarchy.py so
those scripts work on any library without hard-coded parsing rules.

Usage
-----
python specific_scripts/analyze_cav_library.py \\
    --lib-dir  cav_library/9d8e5dca-03a3-457d-b7fb-844c75735c83/ \\
    --api-key  $GEMINI_API_KEY

python specific_scripts/analyze_cav_library.py \\
    --lib-dir  cav_library/3f7c572c-cd73-4b51-a313-207c7f20f188/ \\
    --api-key  $GEMINI_API_KEY

Output
------
lib_dir/library_structure.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are analyzing a set of Concept Activation Vector (CAV) names from a machine
learning library for single-cell biology. Each name is a directory that encodes
the biological concept the CAV represents.

Here are all the CAV names in this library:

{cav_names}

Analyze the naming convention and return ONLY a valid JSON object with exactly
these fields — no explanation, no markdown fences, just the JSON:

{{
  "separator": "<string used to separate levels, e.g. '__'>",
  "n_levels": <integer, number of semantic levels per name>,
  "level_names": ["<descriptive name for level 0>", "<name for level 1>", ...],
  "group_level": <integer, 0-based index of the level to use for coloring/grouping in plots>,
  "condition_level": <integer or null — index of the level that encodes a
                     contrast or condition. This includes: disease vs normal,
                     young vs old, treated vs untreated, stimulated vs resting,
                     etc. Null if all names represent independent concepts with
                     no natural pairing.>,
  "baseline_values": ["<values in condition_level that represent the baseline,
                        reference, or control state. Examples: 'normal', 'young',
                        'untreated', 'resting', 'control'. Use biological
                        judgement — 'young' is baseline relative to 'old',
                        'normal' is baseline relative to 'disease'.>"],
  "summary": "<one sentence describing what this library represents>",
  "pairs": [
    {{"group": "<group value e.g. cell type>",
      "context": "<context value if 3+ levels e.g. tissue, else null>",
      "baseline": "<full CAV name for the baseline condition>",
      "condition": "<full CAV name for the contrast condition>"}}
  ],
  "unpaired": ["<CAV names with no matched counterpart>"]
}}

Rules:
- pairs: only include entries where both a baseline AND a contrast condition
  exist for the same (group, context) combination. Examples of valid pairs:
    macrophage__lung__normal  ↔  macrophage__lung__lung_cancer
    neural_cell__young        ↔  neural_cell__old
    T_cell__stimulated        ↔  T_cell__resting
- unpaired: all CAV names that have no counterpart (e.g. a cell type with only
  a disease version and no matched normal, or an independent concept CAV).
- If the library has no condition level (pure cell-type atlas with no
  contrasts), set condition_level to null, baseline_values to [],
  pairs to [], and unpaired to all names.
"""


def get_cav_names(lib_dir: Path, version: str = "v1") -> list[str]:
    cavs_dir = lib_dir / "cavs"
    if not cavs_dir.exists():
        raise FileNotFoundError(f"No cavs/ directory under {lib_dir}")
    names = sorted(
        d.name for d in cavs_dir.iterdir()
        if d.is_dir() and (d / f"concept_{version}.npy").exists()
    )
    if not names:
        raise ValueError(f"No trained CAVs found in {cavs_dir}")
    return names


def infer_structure(cav_names: list[str], api_key: str,
                    model: str = "gemini-2.5-flash") -> dict:
    from chatlas import ChatGoogle

    names_block = "\n".join(f"  {n}" for n in cav_names)
    prompt = PROMPT_TEMPLATE.format(cav_names=names_block)

    logger.info(f"Sending {len(cav_names)} CAV names to {model}...")
    chat = ChatGoogle(api_key=api_key, model=model)
    response = chat.chat(prompt)

    # Extract text from response
    text = str(response).strip()

    # Strip markdown code fences if model added them anyway
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines
            if not l.startswith("```")
        ).strip()

    try:
        structure = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Model returned invalid JSON:\n{text}")
        raise ValueError(f"Could not parse model response as JSON: {e}")

    return structure


def validate_and_enrich(structure: dict, cav_names: list[str]) -> dict:
    """
    Sanity-check the returned structure and add a flat lookup of all names.
    """
    required = ["separator", "n_levels", "level_names", "group_level",
                "condition_level", "baseline_values", "summary", "pairs", "unpaired"]
    for field in required:
        if field not in structure:
            raise ValueError(f"Missing required field '{field}' in model response")

    # Add the full name list for reference
    structure["all_cavs"] = cav_names

    # Verify pairs reference real names
    name_set = set(cav_names)
    valid_pairs = []
    for pair in structure.get("pairs", []):
        if pair.get("baseline") in name_set and pair.get("condition") in name_set:
            valid_pairs.append(pair)
        else:
            logger.warning(f"Pair references unknown CAV names, skipping: {pair}")
    structure["pairs"] = valid_pairs

    # Ensure unpaired contains only real names
    structure["unpaired"] = [n for n in structure.get("unpaired", [])
                              if n in name_set]

    return structure


def main():
    parser = argparse.ArgumentParser(
        description="Infer CAV library structure using an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lib-dir", required=True,
                        help="CAV library directory (contains cavs/).")
    parser.add_argument("--api-key", required=True,
                        help="Gemini API key.")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model to use (default: gemini-2.5-flash).")
    parser.add_argument("--version", default="v1",
                        help="CAV artifact version suffix (default: v1).")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: lib_dir/library_structure.json).")
    args = parser.parse_args()

    lib_dir = Path(args.lib_dir)
    out_path = Path(args.out) if args.out else lib_dir / "library_structure.json"

    # ------------------------------------------------------------------ #
    # 1. Collect CAV names
    # ------------------------------------------------------------------ #
    cav_names = get_cav_names(lib_dir, version=args.version)
    logger.info(f"Found {len(cav_names)} CAVs in {lib_dir}/cavs/")

    # ------------------------------------------------------------------ #
    # 2. Ask the model
    # ------------------------------------------------------------------ #
    structure = infer_structure(cav_names, args.api_key, model=args.model)

    # ------------------------------------------------------------------ #
    # 3. Validate and save
    # ------------------------------------------------------------------ #
    structure = validate_and_enrich(structure, cav_names)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(structure, f, indent=2)
    logger.info(f"Saved library structure: {out_path}")

    # Summary
    print(f"\nLibrary: {lib_dir.name}")
    print(f"Summary: {structure['summary']}")
    print(f"Levels : {structure['level_names']}")
    print(f"Groups : level {structure['group_level']} ({structure['level_names'][structure['group_level']]})")
    if structure["condition_level"] is not None:
        print(f"Condition: level {structure['condition_level']} "
              f"({structure['level_names'][structure['condition_level']]})")
        print(f"Baseline values : {structure['baseline_values']}")
        print(f"Matched pairs   : {len(structure['pairs'])}")
        print(f"Unpaired        : {len(structure['unpaired'])}")
    else:
        print("No condition level detected (flat concept library)")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
