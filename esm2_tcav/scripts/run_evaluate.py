#!/usr/bin/env python3
"""
Standalone script for CAV evaluation.

Usage:
    python scripts/run_evaluate.py --config config.yaml
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import (
    evaluate_layer,
    save_evaluation_results,
    save_threshold_registry,
    create_evaluation_visualizations
)
from src.utils.model_loader import get_model_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAVs")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.yaml'
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
    
    layers = model_config['layers_to_extract']
    
    logger.info(f"Evaluating CAVs for {model_name}, layers: {layers}")
    
    # Setup paths
    embed_dir = Path(config['embedding']['output_dir']) / model_name
    cav_dir = Path(config['cav']['output_dir']) / model_name
    eval_dir = Path(config['evaluation']['output_dir']) / model_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_version = config['cav']['artifact_version']
    threshold_method = config['evaluation']['thresholding']['method']
    
    # Evaluate each layer
    metrics_by_layer = {}
    
    for layer in layers:
        layer_name = f"L{layer}"
        
        metrics = evaluate_layer(
            layer=layer,
            embed_dir=str(embed_dir),
            cav_dir=str(cav_dir),
            output_dir=str(eval_dir),
            artifact_version=artifact_version,
            threshold_method=threshold_method
        )
        
        metrics_by_layer[layer_name] = metrics
    
    # Save evaluation results
    save_evaluation_results(metrics_by_layer, str(eval_dir))
    
    # Save threshold registry
    threshold_path = config['evaluation']['thresholding']['save_path']
    save_threshold_registry(metrics_by_layer, threshold_path)
    
    # Create visualizations
    if config['evaluation'].get('random_cav_comparison', {}).get('save_plot', True):
        create_evaluation_visualizations(
            metrics_by_layer,
            str(cav_dir),
            str(eval_dir),
            artifact_version
        )
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    
    for layer_name, metrics in metrics_by_layer.items():
        logger.info(
            f"{layer_name}: "
            f"AUROC={metrics['auroc']:.3f}, "
            f"AUPRC={metrics['auprc']:.3f}, "
            f"Threshold={metrics['threshold']:.4f}, "
            f"Significant={'YES' if metrics['random_cav_comparison']['is_significant'] else 'NO'}"
        )
    
    logger.info(f"\n✓ Evaluation complete!")
    logger.info(f"  Results: {eval_dir}")
    logger.info(f"  Thresholds: {threshold_path}")


if __name__ == "__main__":
    main()

