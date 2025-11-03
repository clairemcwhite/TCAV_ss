#!/usr/bin/env python3
"""
Expand prediction intervals using greedy optimization to maximize TCAV scores.

For each prediction in the cleaned detection results, this script:
1. Loads the protein sequence
2. Uses on-the-fly embedding to score the current interval
3. Greedily expands left/right/both to maximize the score
4. Saves the expanded predictions

Usage:
    python expand_intervals.py \
        --detection-results test_detection_results_cleaned.json \
        --test-data-dir ./test_data \
        --tcav-dir ./tcav_outputs_650m \
        --model-name esm2_t33_650M_UR50D \
        --output expanded_predictions.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import EsmModel, EsmTokenizer
from joblib import load as joblib_load
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add esm2_tcav to path for model loading
sys.path.insert(0, str(Path(__file__).parent / "esm2_tcav"))


# ----------------------------
# Data Loading Functions
# ----------------------------

def load_test_proteins(test_data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all test proteins from test_data directory.
    
    Returns dict mapping accession -> {sequence, length, ground_truth}
    """
    proteins = {}
    
    for motif_dir in sorted(test_data_dir.iterdir()):
        if not motif_dir.is_dir() or not motif_dir.name.startswith("PF"):
            continue
        
        jsonl_file = motif_dir / "test_100.jsonl"
        if not jsonl_file.exists():
            continue
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line.strip())
                accession = data.get("accession", "")
                
                # Avoid duplicates (same protein may appear in multiple motif folders)
                if accession not in proteins:
                    proteins[accession] = {
                        "sequence": data.get("sequence", ""),
                        "length": data.get("length", len(data.get("sequence", ""))),
                        "ground_truth": data.get("ground_truth_annotations", [])
                    }
    
    return proteins


def load_cav_artifacts(tcav_dir: Path, motif_id: str, layer: int, version: str = "v1") -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """
    Load CAV vector and scaler for a specific motif and layer.
    
    Returns:
        (cav_vector, scaler) or (None, None) if not found
    """
    cav_dir = tcav_dir / "cavs" / motif_id
    cav_file = cav_dir / f"L{layer}_concept_{version}.npy"
    scaler_file = cav_dir / f"L{layer}_scaler_{version}.pkl"
    
    if not cav_file.exists() or not scaler_file.exists():
        return None, None
    
    try:
        cav = np.load(cav_file)
        scaler = joblib_load(scaler_file)
        return cav, scaler
    except Exception as e:
        print(f"[WARN] Failed to load CAV artifacts for {motif_id} L{layer}: {e}")
        return None, None


# ----------------------------
# Embedding & Scoring Functions
# ----------------------------

def mean_pool_from_hidden(hidden_states, attention_mask, layer_idx_hf):
    """
    Extract mean-pooled embeddings from hidden states.
    
    Args:
        hidden_states: Model output hidden states
        attention_mask: Attention mask
        layer_idx_hf: Layer index (HuggingFace format)
    
    Returns:
        Mean-pooled embedding (excluding BOS/EOS tokens)
    """
    H = hidden_states[layer_idx_hf]  # (B, T, d)
    H_no_special = H[:, 1:-1, :]  # Exclude BOS/EOS
    mask_no_special = attention_mask[:, 1:-1]
    denom = torch.clamp(mask_no_special.sum(dim=1, keepdim=True), min=1).unsqueeze(-1)
    pooled = (H_no_special * mask_no_special.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
    return pooled.squeeze(1)


def score_interval(
    sequence: str,
    span: List[int],
    motif_id: str,
    cav: np.ndarray,
    scaler: Any,
    model,
    tokenizer,
    layer: int,
    device: str
) -> float:
    """
    Score a single interval using on-the-fly embedding and TCAV scoring.
    
    Args:
        sequence: Full protein sequence
        span: [start, end] indices
        motif_id: Motif ID (for logging)
        cav: CAV vector
        scaler: StandardScaler for the motif/layer
        model: ESM2 model
        tokenizer: ESM2 tokenizer
        layer: Layer index
        device: Device (cpu/cuda)
    
    Returns:
        TCAV score for the interval
    """
    start, end = span
    
    # Validate span
    if start < 0 or end > len(sequence) or start >= end:
        return float('-inf')
    
    # Extract subsequence
    subseq = sequence[start:end]
    
    # Tokenize
    try:
        enc = tokenizer([subseq], return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}
    except Exception as e:
        print(f"[WARN] Tokenization failed for {motif_id} span {span}: {e}")
        return float('-inf')
    
    # Get embeddings and score
    try:
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            
            # Mean pool from target layer
            pooled = mean_pool_from_hidden(hs, enc["attention_mask"], layer_idx_hf=layer)
            X = pooled.detach().cpu().numpy()
            
            # Standardize and compute TCAV score
            Xs = scaler.transform(X)
            score = float(Xs @ cav)
            
            return score
    except Exception as e:
        print(f"[WARN] Scoring failed for {motif_id} span {span}: {e}")
        return float('-inf')


# ----------------------------
# Greedy Expansion Algorithm
# ----------------------------

def expand_interval_greedy(
    sequence: str,
    initial_span: List[int],
    motif_id: str,
    cav: np.ndarray,
    scaler: Any,
    model,
    tokenizer,
    layer: int,
    device: str,
    expansion_step: int = 5,
    min_length: int = 20,
    max_length: int = 500,
    score_threshold: float = 0.01,
    max_iterations: int = 50
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Greedily expand an interval to maximize TCAV score.
    
    Algorithm:
        1. Start with initial span
        2. Try expanding left, right, and both directions
        3. Keep the expansion that gives the best score improvement
        4. Repeat until no improvement or max iterations
    
    Args:
        sequence: Full protein sequence
        initial_span: [start, end] initial prediction
        motif_id: Motif ID
        cav: CAV vector
        scaler: StandardScaler
        model: ESM2 model
        tokenizer: ESM2 tokenizer
        layer: Layer index
        device: Device
        expansion_step: Number of residues to expand at a time
        min_length: Minimum valid interval length
        max_length: Maximum valid interval length
        score_threshold: Minimum score improvement to continue
        max_iterations: Maximum expansion iterations
    
    Returns:
        (expanded_span, final_score, stats)
    """
    current_span = initial_span.copy()
    current_score = score_interval(sequence, current_span, motif_id, cav, scaler, model, tokenizer, layer, device)
    
    seq_length = len(sequence)
    iteration = 0
    total_expansions = 0
    
    stats = {
        "initial_span": initial_span.copy(),
        "initial_score": current_score,
        "iterations": 0,
        "total_left_expansions": 0,
        "total_right_expansions": 0,
        "total_both_expansions": 0
    }
    
    while iteration < max_iterations:
        iteration += 1
        best_span = None
        best_score = current_score
        best_direction = None
        
        start, end = current_span
        current_length = end - start
        
        # Try expanding LEFT
        if start > 0 and current_length + expansion_step <= max_length:
            candidate_start = max(0, start - expansion_step)
            candidate_span = [candidate_start, end]
            candidate_score = score_interval(sequence, candidate_span, motif_id, cav, scaler, model, tokenizer, layer, device)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_span = candidate_span
                best_direction = "left"
        
        # Try expanding RIGHT
        if end < seq_length and current_length + expansion_step <= max_length:
            candidate_end = min(seq_length, end + expansion_step)
            candidate_span = [start, candidate_end]
            candidate_score = score_interval(sequence, candidate_span, motif_id, cav, scaler, model, tokenizer, layer, device)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_span = candidate_span
                best_direction = "right"
        
        # Try expanding BOTH
        if start > 0 and end < seq_length and current_length + 2 * expansion_step <= max_length:
            candidate_start = max(0, start - expansion_step)
            candidate_end = min(seq_length, end + expansion_step)
            candidate_span = [candidate_start, candidate_end]
            candidate_score = score_interval(sequence, candidate_span, motif_id, cav, scaler, model, tokenizer, layer, device)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_span = candidate_span
                best_direction = "both"
        
        # Check if we found an improvement
        if best_span is not None and (best_score - current_score) >= score_threshold:
            current_span = best_span
            current_score = best_score
            total_expansions += 1
            
            # Track direction statistics
            if best_direction == "left":
                stats["total_left_expansions"] += 1
            elif best_direction == "right":
                stats["total_right_expansions"] += 1
            elif best_direction == "both":
                stats["total_both_expansions"] += 1
        else:
            # No improvement, stop
            break
    
    stats["iterations"] = iteration
    stats["final_span"] = current_span.copy()
    stats["final_score"] = current_score
    stats["score_improvement"] = current_score - stats["initial_score"]
    stats["length_change"] = (current_span[1] - current_span[0]) - (initial_span[1] - initial_span[0])
    
    return current_span, current_score, stats


# ----------------------------
# Main Processing
# ----------------------------

def expand_all_predictions(
    detection_results: Dict[str, Any],
    protein_sequences: Dict[str, Dict[str, Any]],
    model,
    tokenizer,
    tcav_dir: Path,
    layer: int,
    device: str,
    expansion_step: int = 5,
    min_length: int = 20,
    max_length: int = 500,
    score_threshold: float = 0.01,
    max_iterations: int = 50,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Expand all predictions in detection results.
    
    Args:
        detection_results: Loaded detection results JSON
        protein_sequences: Dict mapping accession -> {sequence, ...}
        model: ESM2 model
        tokenizer: ESM2 tokenizer
        tcav_dir: Path to TCAV outputs directory
        layer: Layer to use for scoring
        device: Device
        expansion_step: Expansion step size
        min_length: Minimum interval length
        max_length: Maximum interval length
        score_threshold: Minimum score improvement
        max_iterations: Max expansion iterations
        verbose: Print progress
    
    Returns:
        Updated detection results with expanded predictions
    """
    # Cache CAV artifacts to avoid reloading
    cav_cache = {}
    
    def get_cav_artifacts(motif_id: str) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        if motif_id not in cav_cache:
            cav, scaler = load_cav_artifacts(tcav_dir, motif_id, layer)
            cav_cache[motif_id] = (cav, scaler)
        return cav_cache[motif_id]
    
    # Statistics
    global_stats = {
        "total_predictions": 0,
        "predictions_expanded": 0,
        "predictions_failed": 0,
        "predictions_skipped": 0,  # No CAV artifacts
        "total_score_improvement": 0.0,
        "total_length_change": 0,
        "avg_iterations": 0.0
    }
    
    results_copy = detection_results.copy()
    results_copy['results'] = []
    
    # Process each protein
    protein_results = detection_results['results']
    
    iterator = tqdm(protein_results, desc="Expanding predictions") if verbose else protein_results
    
    for protein_result in iterator:
        accession = protein_result['accession']
        predictions = protein_result['predictions']
        
        # Get protein sequence
        if accession not in protein_sequences:
            if verbose:
                print(f"\n[WARN] Sequence not found for {accession}, skipping...")
            results_copy['results'].append(protein_result)
            continue
        
        sequence = protein_sequences[accession]['sequence']
        
        # Expand each prediction
        expanded_predictions = []
        
        for pred in predictions:
            global_stats["total_predictions"] += 1
            
            motif_id = pred['motif_id']
            initial_span = pred['span']
            
            # Get CAV artifacts
            cav, scaler = get_cav_artifacts(motif_id)
            
            if cav is None or scaler is None:
                # No CAV artifacts, keep original
                expanded_predictions.append(pred)
                global_stats["predictions_skipped"] += 1
                continue
            
            # Expand interval
            try:
                expanded_span, final_score, stats = expand_interval_greedy(
                    sequence=sequence,
                    initial_span=initial_span,
                    motif_id=motif_id,
                    cav=cav,
                    scaler=scaler,
                    model=model,
                    tokenizer=tokenizer,
                    layer=layer,
                    device=device,
                    expansion_step=expansion_step,
                    min_length=min_length,
                    max_length=max_length,
                    score_threshold=score_threshold,
                    max_iterations=max_iterations
                )
                
                # Update prediction
                pred_expanded = pred.copy()
                pred_expanded['span'] = expanded_span
                pred_expanded['span_before_expansion'] = initial_span
                pred_expanded[f'L{layer}_score_before_expansion'] = stats['initial_score']
                pred_expanded[f'L{layer}_score_after_expansion'] = final_score
                pred_expanded['expansion_stats'] = stats
                
                # Update ranking score to reflect expanded score
                pred_expanded['ranking_score'] = final_score
                pred_expanded[f'per_layer_scores']['L{layer}'] = final_score
                
                expanded_predictions.append(pred_expanded)
                
                # Update global stats
                global_stats["predictions_expanded"] += 1
                global_stats["total_score_improvement"] += stats['score_improvement']
                global_stats["total_length_change"] += stats['length_change']
                global_stats["avg_iterations"] += stats['iterations']
                
            except Exception as e:
                if verbose:
                    print(f"\n[ERROR] Failed to expand {accession} {motif_id} {initial_span}: {e}")
                expanded_predictions.append(pred)
                global_stats["predictions_failed"] += 1
        
        # Sort by new ranking score (descending)
        expanded_predictions.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        # Update protein result
        protein_result_copy = protein_result.copy()
        protein_result_copy['predictions'] = expanded_predictions
        results_copy['results'].append(protein_result_copy)
    
    # Compute average statistics
    if global_stats["predictions_expanded"] > 0:
        global_stats["avg_score_improvement"] = global_stats["total_score_improvement"] / global_stats["predictions_expanded"]
        global_stats["avg_length_change"] = global_stats["total_length_change"] / global_stats["predictions_expanded"]
        global_stats["avg_iterations"] = global_stats["avg_iterations"] / global_stats["predictions_expanded"]
    else:
        global_stats["avg_score_improvement"] = 0.0
        global_stats["avg_length_change"] = 0.0
        global_stats["avg_iterations"] = 0.0
    
    # Add expansion metadata
    results_copy['expansion'] = {
        'layer': layer,
        'expansion_step': expansion_step,
        'min_length': min_length,
        'max_length': max_length,
        'score_threshold': score_threshold,
        'max_iterations': max_iterations,
        'stats': global_stats
    }
    
    return results_copy


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Expand prediction intervals using greedy optimization"
    )
    
    parser.add_argument("--detection-results", "-i",
                       default="./test_detection_results_cleaned.json",
                       help="Input detection results JSON (after NMS)")
    
    parser.add_argument("--test-data-dir", 
                       default="./test_data",
                       help="Directory with test protein sequences")
    
    parser.add_argument("--tcav-dir",
                       default="./tcav_outputs_650m",
                       help="TCAV outputs directory with CAV artifacts")
    
    parser.add_argument("--model-name",
                       default="esm2_t33_650M_UR50D",
                       help="ESM2 model name or path")
    
    parser.add_argument("--layer", "-l",
                       type=int,
                       default=22,
                       help="Layer to use for scoring (default: 22)")
    
    parser.add_argument("--output", "-o",
                       default="./expanded_predictions.json",
                       help="Output JSON file with expanded predictions")
    
    parser.add_argument("--expansion-step",
                       type=int,
                       default=5,
                       help="Number of residues to expand at each step (default: 5)")
    
    parser.add_argument("--min-length",
                       type=int,
                       default=20,
                       help="Minimum interval length (default: 20)")
    
    parser.add_argument("--max-length",
                       type=int,
                       default=500,
                       help="Maximum interval length (default: 500)")
    
    parser.add_argument("--score-threshold",
                       type=float,
                       default=0.01,
                       help="Minimum score improvement to continue expansion (default: 0.01)")
    
    parser.add_argument("--max-iterations",
                       type=int,
                       default=50,
                       help="Maximum expansion iterations per prediction (default: 50)")
    
    parser.add_argument("--device",
                       choices=["cpu", "cuda"],
                       default="cuda",
                       help="Device to use (default: cuda)")
    
    parser.add_argument("--quiet", "-q",
                       action="store_true",
                       help="Suppress progress output")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 80)
        print("GREEDY INTERVAL EXPANSION")
        print("=" * 80)
        print(f"Detection results: {args.detection_results}")
        print(f"Test data dir: {args.test_data_dir}")
        print(f"TCAV dir: {args.tcav_dir}")
        print(f"Model: {args.model_name}")
        print(f"Layer: {args.layer}")
        print(f"Output: {args.output}")
        print()
        print("Expansion parameters:")
        print(f"  Step size: {args.expansion_step} residues")
        print(f"  Min length: {args.min_length}")
        print(f"  Max length: {args.max_length}")
        print(f"  Score threshold: {args.score_threshold}")
        print(f"  Max iterations: {args.max_iterations}")
        print()
    
    # Validate inputs
    if not Path(args.detection_results).exists():
        print(f"[ERROR] Detection results not found: {args.detection_results}")
        return 1
    
    if not Path(args.test_data_dir).exists():
        print(f"[ERROR] Test data directory not found: {args.test_data_dir}")
        return 1
    
    if not Path(args.tcav_dir).exists():
        print(f"[ERROR] TCAV directory not found: {args.tcav_dir}")
        return 1
    
    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("[WARN] CUDA not available, using CPU")
        device = "cpu"
    
    # Load detection results
    if verbose:
        print("[1/5] Loading detection results...")
    
    with open(args.detection_results, 'r') as f:
        detection_results = json.load(f)
    
    n_proteins = len(detection_results['results'])
    total_predictions = sum(len(r['predictions']) for r in detection_results['results'])
    
    if verbose:
        print(f"✓ Loaded {n_proteins} proteins with {total_predictions} predictions")
    
    # Load protein sequences
    if verbose:
        print("\n[2/5] Loading protein sequences...")
    
    test_data_dir = Path(args.test_data_dir)
    protein_sequences = load_test_proteins(test_data_dir)
    
    if verbose:
        print(f"✓ Loaded {len(protein_sequences)} protein sequences")
    
    if len(protein_sequences) == 0:
        print("[ERROR] No protein sequences found!")
        print("Run collect_test_data.py first.")
        return 1
    
    # Load ESM2 model
    if verbose:
        print(f"\n[3/5] Loading ESM2 model ({args.model_name})...")
    
    try:
        # Try loading from registry
        from src.utils.model_loader import load_esm2_model, get_model_config
        
        registry_path = str(Path("esm2_tcav") / "models" / "model_registry.yaml")
        if os.path.exists(registry_path):
            try:
                model_config = get_model_config(args.model_name, registry_path)
                model, tokenizer, _ = load_esm2_model(args.model_name, registry_path, device=device)
                if verbose:
                    print(f"✓ Loaded from registry: {args.model_name}")
            except (KeyError, ValueError):
                raise ValueError("Not in registry")
        else:
            raise ValueError("Registry not found")
    except (ImportError, ValueError):
        # Fallback: HuggingFace
        if verbose:
            print("  Loading from HuggingFace...")
        tokenizer = EsmTokenizer.from_pretrained(args.model_name, do_lower_case=False)
        model = EsmModel.from_pretrained(args.model_name)
        model.to(device)
        if verbose:
            print(f"✓ Loaded from HuggingFace: {args.model_name}")
    
    model.eval()
    
    if verbose:
        print(f"✓ Model ready on {device}")
    
    # Expand predictions
    if verbose:
        print(f"\n[4/5] Expanding {total_predictions} predictions...")
        print()
    
    tcav_dir = Path(args.tcav_dir)
    
    try:
        expanded_results = expand_all_predictions(
            detection_results=detection_results,
            protein_sequences=protein_sequences,
            model=model,
            tokenizer=tokenizer,
            tcav_dir=tcav_dir,
            layer=args.layer,
            device=device,
            expansion_step=args.expansion_step,
            min_length=args.min_length,
            max_length=args.max_length,
            score_threshold=args.score_threshold,
            max_iterations=args.max_iterations,
            verbose=verbose
        )
    except Exception as e:
        print(f"\n[ERROR] Expansion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save results
    if verbose:
        print(f"\n[5/5] Saving expanded predictions...")
    
    with open(args.output, 'w') as f:
        json.dump(expanded_results, f, indent=2)
    
    if verbose:
        print(f"✓ Saved to {args.output}")
        print()
        print("=" * 80)
        print("EXPANSION COMPLETE")
        print("=" * 80)
        
        stats = expanded_results['expansion']['stats']
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Successfully expanded: {stats['predictions_expanded']}")
        print(f"Skipped (no CAV): {stats['predictions_skipped']}")
        print(f"Failed: {stats['predictions_failed']}")
        print()
        
        if stats['predictions_expanded'] > 0:
            print("Expansion statistics:")
            print(f"  Avg score improvement: {stats['avg_score_improvement']:.4f}")
            print(f"  Avg length change: {stats['avg_length_change']:.1f} residues")
            print(f"  Avg iterations: {stats['avg_iterations']:.1f}")
            print()
        
        print(f"Next step: python evaluate_metrics.py --detection-results {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())

