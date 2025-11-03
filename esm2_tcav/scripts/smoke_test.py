#!/usr/bin/env python3
"""
Smoke test: Run end-to-end pipeline on tiny subset to verify setup.

Core feature #5: Health check before expensive HPC jobs

Usage:
    python scripts/smoke_test.py --config config.yaml [--n-samples 5]
"""

import argparse
import yaml
import logging
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_loader import load_esm2_model, get_model_config
from src.utils.data_loader import load_jsonl_data
from src.embed import process_dataset, split_embeddings_by_set
from src.train_cav import train_layer_cavs
from src.evaluate import evaluate_layer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_tiny_dataset(jsonl_path: str, n_samples: int, output_path: str):
    """Create tiny subset of dataset for testing."""
    data = load_jsonl_data(jsonl_path)
    subset = data[:n_samples]
    
    with open(output_path, 'w') as f:
        for sample in subset:
            f.write(json.dumps(sample) + '\n')
    
    return len(subset)


def run_smoke_test(config_path: str, n_samples: int = 5):
    """
    Run smoke test on small subset.
    
    Returns:
        Dictionary of test results
    """
    start_time = datetime.now()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['name']
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SMOKE TEST - {model_name}")
    logger.info(f"Testing with {n_samples} samples per class")
    logger.info(f"{'='*70}\n")
    
    # Create temporary directory for test outputs
    temp_dir = tempfile.mkdtemp(prefix='smoke_test_')
    logger.info(f"Temporary directory: {temp_dir}")
    
    results = {
        'start_time': start_time.isoformat(),
        'model': model_name,
        'n_samples': n_samples,
        'tests': {},
        'temp_dir': temp_dir
    }
    
    try:
        # TEST 1: Model Loading
        logger.info("\n[1/6] Testing model loading...")
        try:
            model, tokenizer, model_config = load_esm2_model(
                model_name,
                config['model']['registry_path'],
                device=config['model']['device']
            )
            results['tests']['model_loading'] = {
                'status': 'PASS',
                'hidden_size': model_config['hidden_size'],
                'layers': model_config['layers_to_extract']
            }
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            results['tests']['model_loading'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"✗ Model loading failed: {e}")
            return results
        
        # TEST 2: Create tiny datasets
        logger.info("\n[2/6] Creating tiny test datasets...")
        try:
            pos_tiny = Path(temp_dir) / 'pos_tiny.jsonl'
            neg_tiny = Path(temp_dir) / 'neg_tiny.jsonl'
            
            n_pos = create_tiny_dataset(
                config['data']['pos_jsonl'], 
                n_samples, 
                str(pos_tiny)
            )
            n_neg = create_tiny_dataset(
                config['data']['neg_jsonl'],
                n_samples,
                str(neg_tiny)
            )
            
            results['tests']['dataset_creation'] = {
                'status': 'PASS',
                'n_pos': n_pos,
                'n_neg': n_neg
            }
            logger.info(f"✓ Created {n_pos} pos + {n_neg} neg samples")
        except Exception as e:
            results['tests']['dataset_creation'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"✗ Dataset creation failed: {e}")
            return results
        
        # TEST 3: Embedding extraction
        logger.info("\n[3/6] Testing embedding extraction...")
        try:
            embed_dir = Path(temp_dir) / 'embeddings'
            
            # Process positive samples
            process_dataset(
                jsonl_path=str(pos_tiny),
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                output_dir=str(embed_dir),
                device=config['model']['device'],
                batch_size=2,  # Small batch for smoke test
                validate_indexing=True,
                corrections_log_path=None
            )
            
            # Process negative samples  
            process_dataset(
                jsonl_path=str(neg_tiny),
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                output_dir=str(embed_dir),
                device=config['model']['device'],
                batch_size=2,
                validate_indexing=True,
                corrections_log_path=None
            )
            
            # Check output files exist
            test_layer = model_config['layers_to_extract'][0]
            emb_file = embed_dir / f"L{test_layer}_all.npy"
            
            if emb_file.exists():
                import numpy as np
                emb = np.load(emb_file)
                results['tests']['embedding_extraction'] = {
                    'status': 'PASS',
                    'shape': list(emb.shape),
                    'dtype': str(emb.dtype)
                }
                logger.info(f"✓ Embeddings extracted: shape {emb.shape}")
            else:
                raise FileNotFoundError(f"Embedding file not created: {emb_file}")
                
        except Exception as e:
            results['tests']['embedding_extraction'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"✗ Embedding extraction failed: {e}")
            return results
        
        # Combine pos/neg embeddings for each layer
        import numpy as np
        for layer in model_config['layers_to_extract']:
            pos_file = embed_dir / f"L{layer}_all.npy"
            neg_file = embed_dir / f"L{layer}_all.npy"
            
            # For smoke test, we processed both through same path
            # Split manually
            all_emb = np.load(embed_dir / f"L{layer}_all.npy")
            mid = len(all_emb) // 2
            
            np.save(embed_dir / f"L{layer}_pos.npy", all_emb[:mid])
            np.save(embed_dir / f"L{layer}_neg.npy", all_emb[mid:])
        
        # TEST 4: CAV training
        logger.info("\n[4/6] Testing CAV training...")
        try:
            cav_dir = Path(temp_dir) / 'cavs'
            
            train_config = {
                'use_pca': config['cav']['use_pca'],
                'pca_dim': min(3, config['cav']['pca_dim']),  # Much smaller for smoke test (5 samples)
                'cv_folds': 2,  # Less folds for speed
                'regularization_C': config['cav']['regularization_C'],
                'n_random_cavs': 5,  # Fewer random CAVs
                'random_mode': config['cav']['random_mode'],
                'random_seed': config['data']['random_seed'],
                'random_seed_base': config['cav']['random_seed_base']
            }
            
            test_layer = model_config['layers_to_extract'][0]
            
            train_layer_cavs(
                layer=test_layer,
                embed_dir=str(embed_dir),
                output_dir=str(cav_dir),
                config=train_config,
                artifact_version='smoke_v1'
            )
            
            # Check CAV file exists
            cav_file = cav_dir / f"L{test_layer}_concept_smoke_v1.npy"
            if cav_file.exists():
                import numpy as np
                cav = np.load(cav_file)
                results['tests']['cav_training'] = {
                    'status': 'PASS',
                    'cav_shape': list(cav.shape)
                }
                logger.info(f"✓ CAV trained: shape {cav.shape}")
            else:
                raise FileNotFoundError(f"CAV file not created: {cav_file}")
                
        except Exception as e:
            results['tests']['cav_training'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"✗ CAV training failed: {e}")
            return results
        
        # TEST 5: Evaluation
        logger.info("\n[5/6] Testing evaluation...")
        try:
            eval_dir = Path(temp_dir) / 'evaluation'
            
            metrics = evaluate_layer(
                layer=test_layer,
                embed_dir=str(embed_dir),
                cav_dir=str(cav_dir),
                output_dir=str(eval_dir),
                artifact_version='smoke_v1',
                threshold_method='f1_max'
            )
            
            results['tests']['evaluation'] = {
                'status': 'PASS',
                'auroc': metrics['auroc'],
                'auprc': metrics['auprc'],
                'threshold': metrics['threshold']
            }
            logger.info(
                f"✓ Evaluation complete: "
                f"AUROC={metrics['auroc']:.3f}, "
                f"AUPRC={metrics['auprc']:.3f}"
            )
            
        except Exception as e:
            results['tests']['evaluation'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"✗ Evaluation failed: {e}")
            return results
        
        # TEST 6: Artifact validation
        logger.info("\n[6/6] Validating artifacts...")
        try:
            required_files = [
                cav_dir / f"L{test_layer}_concept_smoke_v1.npy",
                cav_dir / f"L{test_layer}_scaler_smoke_v1.pkl",
                cav_dir / f"L{test_layer}_report_smoke_v1.json",
            ]
            
            missing = [f for f in required_files if not f.exists()]
            
            if missing:
                raise FileNotFoundError(f"Missing artifacts: {missing}")
            
            results['tests']['artifact_validation'] = {
                'status': 'PASS',
                'required_files': len(required_files),
                'found': len(required_files) - len(missing)
            }
            logger.info(f"✓ All {len(required_files)} required artifacts created")
            
        except Exception as e:
            results['tests']['artifact_validation'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"✗ Artifact validation failed: {e}")
            return results
        
        # All tests passed!
        results['overall_status'] = 'PASS'
        logger.info(f"\n{'='*70}")
        logger.info("✓ ALL SMOKE TESTS PASSED!")
        logger.info(f"{'='*70}")
        
    except Exception as e:
        results['overall_status'] = 'FAIL'
        results['unexpected_error'] = str(e)
        logger.error(f"\n✗ Unexpected error: {e}")
        
    finally:
        # Cleanup
        if '--keep-temp' not in sys.argv:
            logger.info(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        else:
            logger.info(f"\nTemporary directory kept: {temp_dir}")
        
        results['end_time'] = datetime.now().isoformat()
        results['duration_seconds'] = (
            datetime.now() - start_time
        ).total_seconds()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run smoke test")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5,
        help='Number of samples per class to test (default: 5)'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary directory after test'
    )
    
    args = parser.parse_args()
    
    # Run smoke test
    results = run_smoke_test(args.config, args.n_samples)
    
    # Save results
    results_file = 'smoke_test_metrics.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Overall Status: {results.get('overall_status', 'UNKNOWN')}")
    print(f"Duration: {results.get('duration_seconds', 0):.1f}s")
    print(f"\nTest Results:")
    
    for test_name, test_result in results.get('tests', {}).items():
        status = test_result.get('status', 'UNKNOWN')
        symbol = '✓' if status == 'PASS' else '✗'
        print(f"  {symbol} {test_name}: {status}")
    
    print(f"{'='*70}\n")
    
    # Exit code
    if results.get('overall_status') == 'PASS':
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

