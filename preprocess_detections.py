#!/usr/bin/env python3
"""
Preprocess detection results by removing overlapping predictions.

Applies Non-Maximum Suppression (NMS) per motif to handle sliding window redundancy.
NMS is applied separately for each motif, so different motifs can still have 
overlapping predictions in the final result.

Usage:
    # Standard preprocessing (recommended)
    python preprocess_detections.py --input test_detection_results.json --iou-threshold 0.3
    
    # With negative score filtering (optional)
    python preprocess_detections.py --input test_detection_results.json --filter-negative
    
    # Custom IoU threshold (0.3-0.5 recommended)
    python preprocess_detections.py --input test_detection_results.json --iou-threshold 0.4
    
    # Dry run (test without saving)
    python preprocess_detections.py --input test_detection_results.json --dry-run

Recommended IoU thresholds:
    - 0.3: Aggressive filtering (removes more overlaps)
    - 0.4: Balanced
    - 0.5: Conservative (removes fewer overlaps)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def calculate_iou(span1: List[int], span2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two spans.
    
    Args:
        span1: [start, end] of first span
        span2: [start, end] of second span
    
    Returns:
        IoU value between 0 and 1
    """
    start1, end1 = span1
    start2, end2 = span2
    
    # Calculate intersection
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)
    intersection = max(0, intersect_end - intersect_start)
    
    # Calculate union
    union = (end1 - start1) + (end2 - start2) - intersection
    
    return intersection / union if union > 0 else 0


def calculate_overlap_percentage(span1: List[int], span2: List[int]) -> float:
    """
    Calculate what percentage of span1 overlaps with span2.
    
    Returns:
        Percentage of span1 that overlaps with span2 (0 to 1)
    """
    start1, end1 = span1
    start2, end2 = span2
    
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)
    intersection = max(0, intersect_end - intersect_start)
    
    span1_length = end1 - start1
    
    return intersection / span1_length if span1_length > 0 else 0


def filter_negative_scores(predictions: List[Dict[str, Any]], 
                          layer_key: str = None,
                          verbose: bool = False) -> tuple:
    """
    Filter out predictions with negative scores.
    
    Args:
        predictions: List of prediction dictionaries
        layer_key: Specific layer key to check (e.g., "L22"). If None, uses ranking_score
        verbose: Print statistics
    
    Returns:
        Tuple of (filtered_predictions, stats_dict)
    """
    if not predictions:
        return [], {'removed': 0, 'kept': 0}
    
    filtered = []
    removed_count = 0
    
    for pred in predictions:
        # Determine which score to check
        if layer_key and layer_key in pred.get('per_layer_scores', {}):
            score = pred['per_layer_scores'][layer_key]
        else:
            score = pred.get('ranking_score', 0)
        
        if score >= 0:
            filtered.append(pred)
        else:
            removed_count += 1
    
    stats = {
        'removed': removed_count,
        'kept': len(filtered),
        'total': len(predictions)
    }
    
    if verbose and removed_count > 0:
        print(f"  Removed {removed_count} predictions with negative scores")
    
    return filtered, stats


def nms_per_motif(predictions: List[Dict[str, Any]], 
                  iou_threshold: float = 0.5,
                  verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression separately for each motif.
    
    For each motif:
    1. Sort predictions by ranking_score (descending)
    2. Keep highest score, remove overlapping predictions
    3. Repeat for remaining predictions
    
    Different motifs are processed independently, so they can have
    overlapping spans in the final result.
    
    Args:
        predictions: List of prediction dictionaries
        iou_threshold: IoU threshold for considering predictions as overlapping
        verbose: Print detailed statistics
    
    Returns:
        Tuple of (cleaned_predictions, stats_dict)
    """
    if not predictions:
        return [], {'total_before': 0, 'total_after': 0, 'total_removed': 0, 'by_motif_before': {}, 'by_motif_after': {}, 'removed_per_motif': {}}
    
    # Group predictions by motif_id
    by_motif = defaultdict(list)
    for pred in predictions:
        motif_id = pred['motif_id']
        by_motif[motif_id].append(pred)
    
    # Track statistics
    stats = {
        'total_before': len(predictions),
        'by_motif_before': {},
        'by_motif_after': {},
        'removed_per_motif': {}
    }
    
    # Apply NMS to each motif group independently
    cleaned = []
    
    for motif_id, motif_preds in sorted(by_motif.items()):
        stats['by_motif_before'][motif_id] = len(motif_preds)
        
        # Sort by ranking score (descending - highest first)
        motif_preds = sorted(motif_preds, 
                            key=lambda x: x['ranking_score'], 
                            reverse=True)
        
        keep = []
        removed_count = 0
        
        for pred in motif_preds:
            # Check if it overlaps with any already-kept prediction
            overlaps = False
            for kept_pred in keep:
                iou = calculate_iou(pred['span'], kept_pred['span'])
                if iou > iou_threshold:
                    overlaps = True
                    removed_count += 1
                    break
            
            if not overlaps:
                keep.append(pred)
        
        cleaned.extend(keep)
        stats['by_motif_after'][motif_id] = len(keep)
        stats['removed_per_motif'][motif_id] = removed_count
    
    # Re-sort globally by ranking score (descending) to maintain original order
    cleaned.sort(key=lambda x: x['ranking_score'], reverse=True)
    
    stats['total_after'] = len(cleaned)
    stats['total_removed'] = stats['total_before'] - stats['total_after']
    
    if verbose and stats['total_removed'] > 0:
        print(f"  Removed {stats['total_removed']} overlapping predictions:")
        for motif_id in sorted(stats['removed_per_motif'].keys()):
            removed = stats['removed_per_motif'][motif_id]
            if removed > 0:
                before = stats['by_motif_before'][motif_id]
                after = stats['by_motif_after'][motif_id]
                print(f"    {motif_id}: {before} → {after} ({removed} removed)")
    
    return cleaned, stats


def preprocess_detection_results(input_file: str, 
                                 output_file: str,
                                 iou_threshold: float = 0.5,
                                 filter_negative: bool = False,
                                 verbose: bool = True):
    """
    Load detection results, optionally filter negative scores, apply NMS per motif, and save cleaned results.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        iou_threshold: IoU threshold for NMS
        filter_negative: Remove predictions with negative scores (default: False)
        verbose: Print progress and statistics
    """
    if verbose:
        print("=" * 80)
        print("PREPROCESSING DETECTION RESULTS")
        print("=" * 80)
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print(f"IoU threshold: {iou_threshold}")
        print(f"Filter negative scores: {filter_negative}")
        print()
    
    # Load input file
    if verbose:
        print("[1/4] Loading detection results...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    n_proteins = data.get('n_proteins', len(data.get('results', [])))
    rank_by_layer = data.get('rank_by_layer', None)
    layer_key = f"L{rank_by_layer}" if rank_by_layer else None
    
    if verbose:
        print(f"✓ Loaded {n_proteins} proteins")
        if layer_key:
            print(f"✓ Will filter based on {layer_key} scores")
    
    # Process each protein
    global_stats = {
        'total_predictions_original': 0,
        'total_predictions_after_negative_filter': 0,
        'total_predictions_final': 0,
        'negative_score_removals': 0,
        'nms_removals': 0,
        'proteins_processed': 0,
        'proteins_with_negative_removals': 0,
        'proteins_with_nms_removals': 0,
    }
    
    # Step 1: Filter negative scores
    if filter_negative and verbose:
        print(f"\n[2/4] Filtering negative scores...")
    
    for result in data['results']:
        predictions = result['predictions']
        n_original = len(predictions)
        global_stats['total_predictions_original'] += n_original
        
        if filter_negative:
            predictions, neg_stats = filter_negative_scores(
                predictions,
                layer_key=layer_key,
                verbose=False
            )
            n_after_neg_filter = len(predictions)
            n_neg_removed = neg_stats['removed']
            
            global_stats['negative_score_removals'] += n_neg_removed
            global_stats['total_predictions_after_negative_filter'] += n_after_neg_filter
            
            if n_neg_removed > 0:
                global_stats['proteins_with_negative_removals'] += 1
            
            result['predictions'] = predictions
        else:
            global_stats['total_predictions_after_negative_filter'] = global_stats['total_predictions_original']
    
    if verbose and filter_negative:
        print(f"✓ Removed {global_stats['negative_score_removals']} predictions with negative scores")
        print(f"  Affected {global_stats['proteins_with_negative_removals']}/{n_proteins} proteins")
    
    # Step 2: Apply NMS
    if verbose:
        print(f"\n[3/4] Applying NMS (per motif, IoU={iou_threshold})...")
    
    for result in data['results']:
        accession = result['accession']
        predictions_before = result['predictions']
        n_before = len(predictions_before)
        
        # Apply NMS per motif
        predictions_after, stats = nms_per_motif(
            predictions_before, 
            iou_threshold=iou_threshold,
            verbose=False
        )
        
        n_after = len(predictions_after)
        n_removed = n_before - n_after
        
        # Update result
        result['predictions'] = predictions_after
        result['total_hits_original'] = result.get('total_hits', n_before)
        
        # Update global stats
        global_stats['total_predictions_final'] += n_after
        global_stats['nms_removals'] += n_removed
        global_stats['proteins_processed'] += 1
        
        if n_removed > 0:
            global_stats['proteins_with_nms_removals'] += 1
    
    # Update metadata
    data['preprocessing'] = {
        'filter_negative_scores': filter_negative,
        'nms_method': 'per_motif',
        'iou_threshold': iou_threshold,
        'stats': global_stats
    }
    
    if verbose:
        print(f"✓ Removed {global_stats['nms_removals']} overlapping predictions")
        print(f"  Affected {global_stats['proteins_with_nms_removals']}/{n_proteins} proteins")
    
    # Save results
    if verbose:
        print(f"\n[4/4] Saving cleaned results...")
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    if verbose:
        print(f"✓ Saved to {output_file}")
        print()
        print("=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Proteins processed: {global_stats['proteins_processed']}")
        print(f"Original predictions: {global_stats['total_predictions_original']}")
        
        if filter_negative:
            print(f"After negative filter: {global_stats['total_predictions_after_negative_filter']} ({global_stats['negative_score_removals']} removed)")
        
        print(f"After NMS: {global_stats['total_predictions_final']} ({global_stats['nms_removals']} removed)")
        
        total_removed = global_stats['total_predictions_original'] - global_stats['total_predictions_final']
        reduction = (total_removed / global_stats['total_predictions_original'] * 100) if global_stats['total_predictions_original'] > 0 else 0
        print(f"\nTotal reduction: {total_removed} predictions ({reduction:.1f}%)")
        print(f"\nReady for evaluation: python evaluate_metrics.py --detection-results {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess detection results by removing overlapping predictions using NMS"
    )
    
    parser.add_argument("--input", "-i", 
                       default="./test_detection_results.json",
                       help="Input JSON file with detection results")
    
    parser.add_argument("--output", "-o",
                       default="./test_detection_results_cleaned.json", 
                       help="Output JSON file for cleaned results")
    
    parser.add_argument("--iou-threshold", "-t",
                       type=float,
                       default=0.5,
                       help="IoU threshold for considering predictions as overlapping (default: 0.5)")
    
    parser.add_argument("--filter-negative", 
                       action="store_true",
                       default=False,
                       help="Remove predictions with negative scores (default: False)")
    
    parser.add_argument("--no-filter-negative",
                       action="store_false",
                       dest="filter_negative",
                       help="Keep predictions with negative scores (default)")
    
    parser.add_argument("--quiet", "-q",
                       action="store_true",
                       help="Suppress progress output")
    
    parser.add_argument("--dry-run",
                       action="store_true",
                       help="Run without saving output (for testing)")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if args.iou_threshold < 0 or args.iou_threshold > 1:
        print(f"Error: IoU threshold must be between 0 and 1 (got {args.iou_threshold})")
        return 1
    
    # Run preprocessing
    if args.dry_run:
        if not args.quiet:
            print("[DRY RUN MODE - no output will be saved]")
        output_file = "/dev/null"
    else:
        output_file = args.output
    
    try:
        preprocess_detection_results(
            input_file=args.input,
            output_file=output_file,
            iou_threshold=args.iou_threshold,
            filter_negative=args.filter_negative,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

