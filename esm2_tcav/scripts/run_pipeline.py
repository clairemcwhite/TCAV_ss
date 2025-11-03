#!/usr/bin/env python3
"""
End-to-end TCAV pipeline orchestrator.

Runs: Embedding → Training → Evaluation

Usage:
    python scripts/run_pipeline.py --config config.yaml
"""

import argparse
import yaml
import logging
import sys
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_step(script_name, config_path, extra_args=None):
    """Run a pipeline step."""
    cmd = [sys.executable, f"scripts/{script_name}", "--config", config_path]
    if extra_args:
        cmd.extend(extra_args)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"{'='*70}\n")
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full TCAV pipeline")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--skip-embed',
        action='store_true',
        help='Skip embedding extraction (use existing)'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip CAV training (use existing)'
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip evaluation (use existing)'
    )
    
    args = parser.parse_args()
    
    # Load config to get model name
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['name']
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# TCAV PIPELINE - {model_name}")
    logger.info(f"{'#'*70}\n")
    
    # Step 1: Embedding extraction
    if not args.skip_embed:
        logger.info("STEP 1/3: Embedding Extraction")
        success = run_step("run_embed.py", args.config)
        if not success:
            logger.error("Embedding extraction failed!")
            return 1
    else:
        logger.info("STEP 1/3: Skipped (using existing embeddings)")
    
    # Step 2: CAV training
    if not args.skip_train:
        logger.info("\nSTEP 2/3: CAV Training")
        success = run_step("run_train_cav.py", args.config)
        if not success:
            logger.error("CAV training failed!")
            return 1
    else:
        logger.info("STEP 2/3: Skipped (using existing CAVs)")
    
    # Step 3: Evaluation
    if not args.skip_eval:
        logger.info("\nSTEP 3/3: Evaluation")
        success = run_step("run_evaluate.py", args.config)
        if not success:
            logger.error("Evaluation failed!")
            return 1
    else:
        logger.info("STEP 3/3: Skipped (using existing evaluation)")
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# PIPELINE COMPLETE!")
    logger.info(f"{'#'*70}\n")
    
    # Print output locations
    logger.info("Output locations:")
    logger.info(f"  Embeddings: {config['embedding']['output_dir']}/{model_name}/")
    logger.info(f"  CAVs: {config['cav']['output_dir']}/{model_name}/")
    logger.info(f"  Evaluation: {config['evaluation']['output_dir']}/{model_name}/")
    logger.info(f"  Thresholds: {config['evaluation']['thresholding']['save_path']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

