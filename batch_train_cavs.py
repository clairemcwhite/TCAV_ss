#!/usr/bin/env python3
"""
Batch TCAV training for multiple motifs.

For each motif in tcav_data/:
  1. Generate ESM2 embeddings
  2. Train CAVs
  
Usage:
  python batch_train_cavs.py --data-dir ./tcav_data --config esm2_tcav/config.yaml
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# Add esm2_tcav to path
sys.path.insert(0, str(Path(__file__).parent / "esm2_tcav"))

from src.embed import process_dataset
from src.train_cav import train_layer_cavs
from src.utils.model_loader import get_model_config, load_esm2_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_motifs(data_dir: str) -> List[Dict[str, Any]]:
    """
    Find all motifs with collected data in the data directory.
    """
    data_path = Path(data_dir)
    motifs = []
    
    for motif_dir in sorted(data_path.iterdir()):
        if not motif_dir.is_dir():
            continue
        
        metadata_file = motif_dir / "metadata.json"
        if not metadata_file.exists():
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check if we have enough data
        if metadata.get("positives_collected", 0) < 50:
            logger.warning(f"Skipping {motif_dir.name}: insufficient positives")
            continue
        
        # Find the JSONL files
        pos_jsonl = list(motif_dir.glob("pos_*.jsonl"))
        neg_jsonl = list(motif_dir.glob("neg_*.jsonl"))
        
        if not pos_jsonl or not neg_jsonl:
            logger.warning(f"Skipping {motif_dir.name}: missing JSONL files")
            continue
        
        motifs.append({
            "id": motif_dir.name,
            "dir": str(motif_dir),
            "pos_jsonl": str(pos_jsonl[0]),
            "neg_jsonl": str(neg_jsonl[0]),
            "metadata": metadata
        })
    
    return motifs


def generate_motif_embeddings(
    motif: Dict[str, Any],
    model,
    tokenizer,
    model_config: Dict,
    model_name: str,
    batch_size: int,
    output_base: str,
    device: str = "cuda"
) -> str:
    """
    Generate embeddings for a single motif.
    Returns the embedding directory path.
    """
    motif_id = motif["id"]
    embed_dir = Path(output_base) / "embeddings" / model_name / motif_id
    embed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Generating embeddings for {motif_id}")
    logger.info(f"{'='*70}")
    
    # Check if already exists
    expected_files = [f"L{layer}_pos.npy" for layer in model_config['layers_to_extract']]
    expected_files += [f"L{layer}_neg.npy" for layer in model_config['layers_to_extract']]
    
    if all((embed_dir / f).exists() for f in expected_files):
        logger.info(f"✓ Embeddings already exist for {motif_id}, skipping")
        return str(embed_dir)
    
    # Generate embeddings for positives
    logger.info(f"Processing positives: {motif['pos_jsonl']}")
    process_dataset(
        jsonl_path=motif["pos_jsonl"],
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        output_dir=str(embed_dir / "pos"),
        device=device,
        batch_size=batch_size,
        validate_indexing=True
    )
    
    # Generate embeddings for negatives
    logger.info(f"Processing negatives: {motif['neg_jsonl']}")
    process_dataset(
        jsonl_path=motif["neg_jsonl"],
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        output_dir=str(embed_dir / "neg"),
        device=device,
        batch_size=batch_size,
        validate_indexing=True
    )
    
    # Copy/symlink the _all.npy files to _pos.npy and _neg.npy for CAV training
    for layer in model_config['layers_to_extract']:
        pos_src = embed_dir / "pos" / f"L{layer}_all.npy"
        pos_dst = embed_dir / f"L{layer}_pos.npy"
        neg_src = embed_dir / "neg" / f"L{layer}_all.npy"
        neg_dst = embed_dir / f"L{layer}_neg.npy"
        
        if pos_src.exists():
            shutil.copy(pos_src, pos_dst)
        if neg_src.exists():
            shutil.copy(neg_src, neg_dst)
    
    logger.info(f"✓ Embeddings saved to {embed_dir}")
    return str(embed_dir)


def train_motif_cavs(
    motif: Dict[str, Any],
    embed_dir: str,
    layers: List[int],
    train_config: Dict,
    output_base: str,
    artifact_version: str = "v1"
) -> str:
    """
    Train CAVs for a single motif.
    Returns the CAV directory path.
    """
    motif_id = motif["id"]
    cav_dir = Path(output_base) / "cavs" / motif_id
    cav_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training CAVs for {motif_id}")
    logger.info(f"{'='*70}")
    
    # Check if already trained
    expected_files = [f"L{layer}_cav_{artifact_version}.pkl" for layer in layers]
    if all((cav_dir / f).exists() for f in expected_files):
        logger.info(f"✓ CAVs already trained for {motif_id}, skipping")
        return str(cav_dir)
    
    # Train for each layer
    for layer in layers:
        logger.info(f"Training layer {layer}...")
        train_layer_cavs(
            layer=layer,
            embed_dir=embed_dir,
            output_dir=str(cav_dir),
            config=train_config,
            artifact_version=artifact_version
        )
    
    logger.info(f"✓ CAVs saved to {cav_dir}")
    return str(cav_dir)


def batch_train_all(
    data_dir: str,
    config_path: str,
    output_dir: str,
    model_name: str = None,
    layers: List[int] = None,
    batch_size: int = None,
    skip_existing: bool = True
):
    """
    Main batch training function.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model config
    model_name = model_name or config['model']['name']
    model_config = get_model_config(
        model_name,
        str(Path(config_path).parent / config['model']['registry_path'])
    )
    
    # Determine layers
    if layers is None:
        layers = model_config['layers_to_extract']
    
    # Determine batch size
    if batch_size is None:
        batch_size = config['embedding']['batch_size']
    
    device = config['model']['device']
    
    logger.info("="*70)
    logger.info("BATCH TCAV TRAINING PIPELINE")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Find all motifs
    motifs = find_motifs(data_dir)
    logger.info(f"Found {len(motifs)} motifs with valid data")
    
    if not motifs:
        logger.error("No valid motifs found!")
        return
    
    # Load model once (reuse for all motifs)
    logger.info(f"\nLoading model: {model_name}...")
    registry_path = str(Path(config_path).parent / config['model']['registry_path'])
    model, tokenizer, _ = load_esm2_model(model_name, registry_path, device=device)
    logger.info("✓ Model loaded")
    
    # CAV training config
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
    
    # Process each motif
    results = []
    for i, motif in enumerate(motifs, 1):
        logger.info(f"\n{'#'*70}")
        logger.info(f"Processing motif {i}/{len(motifs)}: {motif['id']}")
        logger.info(f"{'#'*70}")
        
        try:
            # Step 1: Generate embeddings
            embed_dir = generate_motif_embeddings(
                motif=motif,
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                model_name=model_name,
                batch_size=batch_size,
                output_base=output_dir,
                device=device
            )
            
            # Step 2: Train CAVs
            cav_dir = train_motif_cavs(
                motif=motif,
                embed_dir=embed_dir,
                layers=layers,
                train_config=train_config,
                output_base=output_dir,
                artifact_version=artifact_version
            )
            
            results.append({
                "motif_id": motif["id"],
                "status": "success",
                "embed_dir": embed_dir,
                "cav_dir": cav_dir,
                "n_positives": motif["metadata"]["positives_collected"],
                "n_negatives": motif["metadata"]["negatives_collected"]
            })
            
            logger.info(f"✓ Successfully processed {motif['id']}")
            
        except Exception as e:
            logger.error(f"✗ Error processing {motif['id']}: {e}", exc_info=True)
            results.append({
                "motif_id": motif["id"],
                "status": "failed",
                "error": str(e)
            })
    
    # Save summary
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_file = output_path / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "model": model_name,
            "layers": layers,
            "total_motifs": len(motifs),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "results": results
        }, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("BATCH TRAINING SUMMARY")
    logger.info("="*70)
    logger.info(f"Total motifs: {len(motifs)}")
    logger.info(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    logger.info(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("\n✓ Batch training complete!")


def main():
    parser = argparse.ArgumentParser(description="Batch train TCAVs for multiple motifs")
    parser.add_argument("--data-dir", required=True, help="Directory containing motif data (tcav_data/)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--output-dir", default="./tcav_outputs", help="Output directory for embeddings and CAVs")
    parser.add_argument("--model", help="Model name (overrides config)")
    parser.add_argument("--layers", type=int, nargs="+", help="Layers to process (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")
    
    args = parser.parse_args()
    
    batch_train_all(
        data_dir=args.data_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        model_name=args.model,
        layers=args.layers,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()



