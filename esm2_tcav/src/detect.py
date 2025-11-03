"""
Detection module for applying trained CAVs to unannotated proteins.

Uses threshold registry from evaluation.
"""

import torch
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats

from .utils.data_loader import parse_fasta
from .utils.model_loader import load_esm2_model, get_layers_to_extract
from .embed import extract_layer_embeddings, pool_window_embeddings
from .evaluate import load_cav_artifacts, compute_projections

logger = logging.getLogger(__name__)


def load_threshold_registry(threshold_path: str) -> Dict:
    """
    Load threshold registry from evaluation.
    
    Args:
        threshold_path: Path to thresholds.json
        
    Returns:
        Dictionary mapping layer names to threshold info
    """
    with open(threshold_path, 'r') as f:
        thresholds = json.load(f)
    
    logger.info(f"Loaded thresholds for {len(thresholds)} layers")
    return thresholds


def sliding_window_scan(
    sequence: str,
    model: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    cav_artifacts: Dict,
    window_size: int = 41,
    stride: int = 1,
    device: str = "cuda"
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Scan sequence with sliding window and compute CAV projections.
    
    Args:
        sequence: Protein sequence
        model: ESM2 model
        tokenizer: ESM2 tokenizer
        layer: Layer to extract from
        cav_artifacts: CAV artifacts dict
        window_size: Window size
        stride: Stride for sliding
        device: Device to run on
        
    Returns:
        Tuple of (projection_scores, window_spans)
    """
    # Extract full sequence embedding
    sequences = [("scan", sequence)]
    layer_embeddings = extract_layer_embeddings(
        model, tokenizer, sequences, [layer], device, batch_size=1
    )
    
    seq_emb = layer_embeddings[layer][0]  # (seq_len, hidden_dim)
    
    # Slide window
    projection_scores = []
    window_spans = []
    
    for start in range(0, len(sequence) - window_size + 1, stride):
        end = start + window_size
        window_span = (start, end)
        
        # Pool window
        pooled = pool_window_embeddings(seq_emb, window_span, bos_offset=1)
        
        # Compute projection
        # Need to expand to 2D for preprocessing
        pooled_2d = pooled.reshape(1, -1)
        projection = compute_projections(
            pooled_2d,
            cav_artifacts['concept_cav'],
            cav_artifacts['scaler'],
            cav_artifacts['pca']
        )[0]
        
        projection_scores.append(projection)
        window_spans.append(window_span)
    
    return np.array(projection_scores), window_spans


def detect_motifs_in_sequence(
    accession: str,
    sequence: str,
    model: torch.nn.Module,
    tokenizer: Any,
    layers: List[int],
    cav_artifacts_by_layer: Dict[int, Dict],
    thresholds: Dict[str, Dict],
    window_size: int = 41,
    device: str = "cuda",
    confidence_levels: Dict[str, float] = None
) -> Dict:
    """
    Detect motifs in a single protein sequence.
    
    Args:
        accession: Protein accession
        sequence: Protein sequence
        model: ESM2 model
        tokenizer: ESM2 tokenizer
        layers: Layers to use
        cav_artifacts_by_layer: Dict mapping layer -> CAV artifacts
        thresholds: Threshold registry
        window_size: Window size
        device: Device
        confidence_levels: Z-score thresholds for confidence
        
    Returns:
        Detection results dictionary
    """
    if confidence_levels is None:
        confidence_levels = {'high': 3.0, 'medium': 2.0, 'low': 1.0}
    
    logger.info(f"Scanning {accession} (length {len(sequence)})...")
    
    detections_by_layer = {}
    
    for layer in layers:
        layer_name = f"L{layer}"
        
        # Scan with sliding window
        projection_scores, window_spans = sliding_window_scan(
            sequence,
            model,
            tokenizer,
            layer,
            cav_artifacts_by_layer[layer],
            window_size=window_size,
            device=device
        )
        
        # Get threshold
        threshold = thresholds[layer_name]['threshold']
        
        # Find windows above threshold
        above_threshold = projection_scores > threshold
        
        if above_threshold.any():
            # Get top window
            top_idx = np.argmax(projection_scores)
            top_score = projection_scores[top_idx]
            top_window = window_spans[top_idx]
            
            # Compute z-score (relative to random baseline ~0.5 AUROC)
            # Approximate: use threshold as reference
            z_score = (top_score - threshold) / (threshold * 0.1 + 1e-6)
            
            # Determine confidence
            if z_score >= confidence_levels['high']:
                confidence = 'high'
            elif z_score >= confidence_levels['medium']:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            # Extract sequence
            start, end = top_window
            window_seq = sequence[start:end]
            
            detections_by_layer[layer_name] = {
                'detected': True,
                'top_window': {
                    'span_0based': list(top_window),
                    'projection_score': float(top_score),
                    'z_score': float(z_score),
                    'confidence': confidence,
                    'sequence': window_seq
                },
                'n_windows_above_threshold': int(above_threshold.sum()),
                'max_projection': float(projection_scores.max()),
                'threshold_used': float(threshold)
            }
        else:
            detections_by_layer[layer_name] = {
                'detected': False,
                'max_projection': float(projection_scores.max()),
                'threshold_used': float(threshold)
            }
    
    # Ensemble prediction (if multiple layers)
    has_detection = any(d.get('detected', False) for d in detections_by_layer.values())
    
    ensemble_result = {
        'has_ZnF': has_detection,
        'n_layers_detected': sum(1 for d in detections_by_layer.values() if d.get('detected', False)),
        'layers_used': [f"L{l}" for l in layers]
    }
    
    # If detected, find consensus window
    if has_detection:
        detected_windows = [
            d['top_window']['span_0based']
            for d in detections_by_layer.values()
            if d.get('detected', False)
        ]
        
        # Simple consensus: average start/end
        avg_start = int(np.mean([w[0] for w in detected_windows]))
        avg_end = int(np.mean([w[1] for w in detected_windows]))
        
        ensemble_result['consensus_window'] = [avg_start, avg_end]
        ensemble_result['consensus_sequence'] = sequence[avg_start:avg_end]
    
    result = {
        'accession': accession,
        'sequence_length': len(sequence),
        'detections_by_layer': detections_by_layer,
        'ensemble': ensemble_result
    }
    
    return result


def batch_detect(
    fasta_path: str,
    model_name: str,
    model_config: Dict,
    model: torch.nn.Module,
    tokenizer: Any,
    cav_dir: str,
    threshold_path: str,
    output_dir: str,
    window_size: int = 41,
    artifact_version: str = "v1",
    device: str = "cuda"
) -> None:
    """
    Detect motifs in batch of sequences from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
        model_name: Model name
        model_config: Model configuration
        model: Loaded ESM2 model
        tokenizer: ESM2 tokenizer
        cav_dir: CAV directory
        threshold_path: Path to thresholds.json
        output_dir: Output directory
        window_size: Window size
        artifact_version: CAV version
        device: Device
    """
    # Load sequences
    sequences = parse_fasta(fasta_path)
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    
    # Load thresholds
    thresholds = load_threshold_registry(threshold_path)
    
    # Get layers
    layers = model_config['layers_to_extract']
    
    # Load CAV artifacts for all layers
    cav_artifacts_by_layer = {}
    for layer in layers:
        cav_artifacts_by_layer[layer] = load_cav_artifacts(
            layer, cav_dir, artifact_version
        )
    
    # Detect in each sequence
    all_results = []
    
    for accession, sequence in sequences.items():
        try:
            result = detect_motifs_in_sequence(
                accession,
                sequence,
                model,
                tokenizer,
                layers,
                cav_artifacts_by_layer,
                thresholds,
                window_size=window_size,
                device=device
            )
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Error detecting in {accession}: {e}")
            continue
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "predictions.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"✓ Saved predictions: {results_file}")
    
    # Summary statistics
    n_detected = sum(1 for r in all_results if r['ensemble']['has_ZnF'])
    logger.info(
        f"\nDetection summary:\n"
        f"  Total sequences: {len(all_results)}\n"
        f"  Detected ZnF: {n_detected}\n"
        f"  Detection rate: {n_detected/len(all_results):.1%}"
    )
    
    # Save summary
    summary = {
        'n_sequences': len(all_results),
        'n_detected': n_detected,
        'detection_rate': n_detected / len(all_results),
        'model_used': model_name,
        'layers_used': [f"L{l}" for l in layers],
        'window_size': window_size
    }
    
    summary_file = output_path / "detection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✓ Saved summary: {summary_file}")


