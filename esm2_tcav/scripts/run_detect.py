#!/usr/bin/env python3
"""
Standalone script for motif detection in unannotated proteins.

Usage:
    python scripts/run_detect.py --config config.yaml --input proteins.fasta
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_loader import load_esm2_model
from src.detect import batch_detect

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Detect zinc finger motifs")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input FASTA file with unannotated proteins'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        help='Device (cuda/cpu, overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings if specified
    model_name = args.model or config['model']['name']
    device = args.device or config['model']['device']
    
    output_dir = args.output or str(
        Path(config['detection']['output_dir']) / Path(args.input).stem
    )
    
    logger.info(f"Running detection with {model_name}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {output_dir}")
    
    # Load model
    model, tokenizer, model_config = load_esm2_model(
        model_name,
        config['model']['registry_path'],
        device=device
    )
    
    # Setup paths
    cav_dir = Path(config['cav']['output_dir']) / model_name
    threshold_path = config['detection']['thresholds_path']
    
    # Run detection
    batch_detect(
        fasta_path=args.input,
        model_name=model_name,
        model_config=model_config,
        model=model,
        tokenizer=tokenizer,
        cav_dir=str(cav_dir),
        threshold_path=threshold_path,
        output_dir=output_dir,
        window_size=config['data']['window_length'],
        artifact_version=config['cav']['artifact_version'],
        device=device
    )
    
    logger.info(f"\n✓ Detection complete!")
    logger.info(f"  Results: {output_dir}/predictions.json")


if __name__ == "__main__":
    main()

