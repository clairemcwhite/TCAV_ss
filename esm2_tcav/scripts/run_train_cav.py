#!/usr/bin/env python3
"""
Standalone script for CAV training.

Usage:
    python scripts/run_train_cav.py --config config.yaml [--layer LAYER]
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train_cav import train_layer_cavs
from src.utils.model_loader import get_model_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train CAVs")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--layer',
        type=int,
        help='Specific layer to train (if not set, trains all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model config
    model_name = args.model or config['model']['name']
    model_config = get_model_config(
        model_name,
        config['model']['registry_path']
    )
    
    # Determine layers to train
    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = model_config['layers_to_extract']
    
    logger.info(f"Training CAVs for {model_name}, layers: {layers}")
    
    # Setup paths
    embed_dir = Path(config['embedding']['output_dir']) / model_name
    cav_dir = Path(config['cav']['output_dir']) / model_name
    
    # Training config
    train_config = {
        'use_pca': config['cav']['use_pca'],
        'pca_dim': config['cav']['pca_dim'],
        'cv_folds': config['cav']['cv_folds'],
        'regularization_C': config['cav']['regularization_C'],
        'n_random_cavs': config['cav']['n_random_cavs'],
        'random_mode': config['cav']['random_mode'],
        'random_seed': config['data']['random_seed'],
        'random_seed_base': config['cav']['random_seed_base']
    }
    
    artifact_version = config['cav']['artifact_version']
    
    # Train CAVs for each layer
    for layer in layers:
        train_layer_cavs(
            layer=layer,
            embed_dir=str(embed_dir),
            output_dir=str(cav_dir),
            config=train_config,
            artifact_version=artifact_version
        )
    
    logger.info(f"\n✓ CAV training complete!")
    logger.info(f"  Output directory: {cav_dir}")


if __name__ == "__main__":
    main()

