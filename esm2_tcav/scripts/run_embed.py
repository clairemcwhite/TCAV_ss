#!/usr/bin/env python3
"""
Standalone script for embedding extraction.

Usage:
    python scripts/run_embed.py --config config.yaml [--model MODEL_NAME]
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_loader import load_esm2_model
from src.embed import process_dataset, split_embeddings_by_set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract ESM2 embeddings")
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
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu, overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override model if specified
    model_name = args.model or config['model']['name']
    device = args.device or config['model']['device']
    
    logger.info(f"Starting embedding extraction with {model_name}")
    
    # Load model
    model, tokenizer, model_config = load_esm2_model(
        model_name,
        config['model']['registry_path'],
        device=device
    )
    
    # Determine output directory
    model_output_dir = Path(config['embedding']['output_dir']) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process positive and negative datasets
    for dataset_type in ['pos', 'neg']:
        jsonl_key = 'pos_jsonl' if dataset_type == 'pos' else 'neg_jsonl'
        jsonl_path = config['data'][jsonl_key]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_type.upper()} dataset: {jsonl_path}")
        logger.info(f"{'='*60}")
        
        # Setup corrections log
        corrections_log = None
        if config['embedding'].get('log_corrections', True):
            corrections_log = config['embedding'].get('corrections_log')
        
        # Process dataset
        process_dataset(
            jsonl_path=jsonl_path,
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            output_dir=str(model_output_dir / dataset_type),
            device=device,
            batch_size=config['embedding']['batch_size'],
            validate_indexing=config['embedding'].get('validate_bos_offset', True),
            corrections_log_path=corrections_log
        )
    
    # Combine and split embeddings by layer
    logger.info("\nCombining positive and negative embeddings...")
    
    for layer in model_config['layers_to_extract']:
        # Load from separate pos/neg directories
        pos_emb_file = model_output_dir / 'pos' / f"L{layer}_all.npy"
        neg_emb_file = model_output_dir / 'neg' / f"L{layer}_all.npy"
        pos_meta_file = model_output_dir / 'pos' / f"L{layer}_meta.json"
        neg_meta_file = model_output_dir / 'neg' / f"L{layer}_meta.json"
        
        # Combine and save to main directory
        import numpy as np
        import json
        
        pos_emb = np.load(pos_emb_file)
        neg_emb = np.load(neg_emb_file)
        
        # Save combined
        combined_emb = np.vstack([pos_emb, neg_emb])
        np.save(model_output_dir / f"L{layer}_all.npy", combined_emb)
        
        # Also save separate
        np.save(model_output_dir / f"L{layer}_pos.npy", pos_emb)
        np.save(model_output_dir / f"L{layer}_neg.npy", neg_emb)
        
        # Combine metadata
        with open(pos_meta_file, 'r') as f:
            pos_meta = json.load(f)
        with open(neg_meta_file, 'r') as f:
            neg_meta = json.load(f)
        
        combined_meta = pos_meta + neg_meta
        with open(model_output_dir / f"L{layer}_meta.json", 'w') as f:
            json.dump(combined_meta, f, indent=2)
        
        logger.info(f"Layer {layer}: {len(pos_emb)} pos + {len(neg_emb)} neg = {len(combined_emb)} total")
    
    logger.info(f"\n✓ Embedding extraction complete!")
    logger.info(f"  Output directory: {model_output_dir}")


if __name__ == "__main__":
    main()

