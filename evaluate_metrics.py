#!/usr/bin/env python3
"""
Evaluate motif detection performance with precision, recall, and F1 metrics.

Computes metrics for:
- Different k values (1-20): top-k predictions
- Different overlap thresholds (80%, 85%, 90%, 95%, 100%)

Outputs tables (CSV and text) for each overlap threshold.

Usage:
    python evaluate_metrics.py --detection-results ./test_detection_results.json --embeddings-dir ./test_data/embeddings --output-dir ./evaluation_tables

Features:
- Calculates precision, recall, F1 for each (k, overlap) combination
- Handles multi-motif proteins correctly
- Outputs easy-to-read tables
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

# ----------------------------
# Overlap Calculation
# ----------------------------

def calculate_overlap_percentage(
    pred_span: Tuple[int, int],  # [start, end) in 0-based half-open
    true_span: Tuple[int, int],  # [start, end) 1-based inclusive (from UniProt)
) -> float:
    """
    Calculate overlap percentage between prediction and ground truth.
    
    Overlap = (intersection length / ground truth length) * 100
    
    Note: pred_span is 0-based half-open [start, end)
          true_span is 1-based inclusive [start, end] (from UniProt annotations)
    """
    # Convert true_span from 1-based inclusive to 0-based half-open
    true_start = true_span[0] - 1  # 1-based to 0-based
    true_end = true_span[1]  # inclusive to half-open is same value
    
    pred_start, pred_end = pred_span
    
    # Calculate intersection
    intersect_start = max(pred_start, true_start)
    intersect_end = min(pred_end, true_end)
    
    if intersect_start >= intersect_end:
        # No overlap
        return 0.0
    
    intersect_len = intersect_end - intersect_start
    true_len = true_end - true_start
    
    if true_len == 0:
        return 0.0
    
    overlap_pct = (intersect_len / true_len) * 100.0
    
    return overlap_pct


def is_true_positive(
    prediction: Dict[str, Any],
    ground_truth_annotations: List[Dict[str, Any]],
    overlap_threshold: float
) -> bool:
    """
    Check if a prediction is a true positive.
    
    TP criteria:
    - Predicted motif ID matches a ground truth motif ID
    - Overlap percentage >= threshold
    
    Args:
        prediction: Dict with 'motif_id' and 'span' [start, end) 0-based half-open
        ground_truth_annotations: List of ground truth annotations with 'motif_id', 'start', 'end' (1-based inclusive)
        overlap_threshold: Minimum overlap percentage (e.g., 80.0)
    
    Returns:
        True if prediction matches any ground truth with sufficient overlap
    """
    pred_motif = prediction["motif_id"]
    pred_span = tuple(prediction["span"])
    
    for gt_ann in ground_truth_annotations:
        if gt_ann["motif_id"] == pred_motif:
            # Same motif type - check overlap
            gt_span = (gt_ann["start"], gt_ann["end"])  # 1-based inclusive
            overlap_pct = calculate_overlap_percentage(pred_span, gt_span)
            
            if overlap_pct >= overlap_threshold:
                return True
    
    return False


# ----------------------------
# Metrics Calculation
# ----------------------------

def compute_metrics_for_k_and_overlap(
    detection_results: List[Dict[str, Any]],
    protein_ground_truths: Dict[str, List[Dict[str, Any]]],
    k: int,
    overlap_threshold: float
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for a specific k and overlap threshold.
    
    Args:
        detection_results: List of detection results (one per protein)
        protein_ground_truths: Dict mapping accession -> ground truth annotations
        k: Top-k predictions to consider
        overlap_threshold: Overlap threshold percentage (e.g., 80.0)
    
    Returns:
        Dict with precision, recall, f1, tp, fp, fn counts
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for result in detection_results:
        accession = result["accession"]
        predictions = result["predictions"][:k]  # Top-k predictions
        
        ground_truth = protein_ground_truths.get(accession, [])
        
        if len(ground_truth) == 0:
            # No ground truth - all predictions are FP
            total_fp += len(predictions)
            continue
        
        # Track which ground truth annotations were detected
        detected_gt_indices = set()
        
        # Check each prediction
        for pred in predictions:
            is_tp = False
            
            for gt_idx, gt_ann in enumerate(ground_truth):
                if is_true_positive(pred, [gt_ann], overlap_threshold):
                    # This is a TP
                    is_tp = True
                    detected_gt_indices.add(gt_idx)
                    break  # One prediction can match one GT
            
            if is_tp:
                total_tp += 1
            else:
                total_fp += 1
        
        # Count FN: ground truth annotations not detected in top-k
        total_fn += len(ground_truth) - len(detected_gt_indices)
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }


# ----------------------------
# Table Generation
# ----------------------------

def generate_metrics_table(
    detection_results: List[Dict[str, Any]],
    protein_ground_truths: Dict[str, List[Dict[str, Any]]],
    k_values: List[int],
    overlap_threshold: float
) -> pd.DataFrame:
    """
    Generate metrics table for all k values at a specific overlap threshold.
    
    Returns:
        DataFrame with columns: k, precision, recall, f1
    """
    rows = []
    
    for k in k_values:
        metrics = compute_metrics_for_k_and_overlap(
            detection_results=detection_results,
            protein_ground_truths=protein_ground_truths,
            k=k,
            overlap_threshold=overlap_threshold
        )
        
        rows.append({
            "k": k,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"]
        })
    
    df = pd.DataFrame(rows)
    return df


def save_table(
    df: pd.DataFrame,
    output_dir: Path,
    overlap_threshold: float
):
    """
    Save table as both CSV and formatted text.
    """
    overlap_int = int(overlap_threshold)
    
    # Save CSV
    csv_file = output_dir / f"metrics_overlap_{overlap_int}.csv"
    df.to_csv(csv_file, index=False, float_format="%.4f")
    
    # Save formatted text
    txt_file = output_dir / f"metrics_overlap_{overlap_int}.txt"
    with open(txt_file, 'w') as f:
        f.write(f"Motif Detection Performance Metrics\n")
        f.write(f"Overlap Threshold: {overlap_threshold}%\n")
        f.write(f"=" * 80 + "\n\n")
        
        # Format table
        f.write(f"{'k':<6} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in df.iterrows():
            f.write(
                f"{int(row['k']):<6} "
                f"{row['precision']:<12.4f} "
                f"{row['recall']:<12.4f} "
                f"{row['f1']:<12.4f} "
                f"{int(row['tp']):<8} "
                f"{int(row['fp']):<8} "
                f"{int(row['fn']):<8}\n"
            )
    
    return csv_file, txt_file


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate motif detection performance"
    )
    
    parser.add_argument("--detection-results", default="./test_detection_results.json",
                       help="Detection results JSON file (includes ground truth)")
    parser.add_argument("--output-dir", default="./evaluation_tables",
                       help="Output directory for tables")
    parser.add_argument("--k-max", type=int, default=20,
                       help="Maximum k value to evaluate")
    parser.add_argument("--overlap-thresholds", type=float, nargs="+",
                       default=[80.0, 85.0, 90.0, 95.0, 100.0],
                       help="Overlap thresholds to evaluate")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EVALUATION: PRECISION, RECALL, F1")
    print("=" * 80)
    
    # Load detection results
    print("\n[1/3] Loading detection results...")
    
    if not os.path.exists(args.detection_results):
        print(f"[ERROR] Detection results not found: {args.detection_results}")
        print("Run run_test_detection.py first.")
        return
    
    with open(args.detection_results, 'r') as f:
        detection_data = json.load(f)
    
    detection_results = detection_data["results"]
    print(f"✓ Loaded detection results for {len(detection_results)} proteins")
    
    # Load ground truth from detection results
    print("\n[2/3] Loading ground truth annotations...")
    
    protein_ground_truths = {}
    
    for result in detection_results:
        accession = result["accession"]
        # Ground truth is now included in detection results
        protein_ground_truths[accession] = result.get("ground_truth", [])
    
    print(f"✓ Loaded ground truth for {len(protein_ground_truths)} proteins")
    
    # Compute metrics for all k and overlap combinations
    print(f"\n[3/3] Computing metrics...")
    
    k_values = list(range(1, args.k_max + 1))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  K values: 1 to {args.k_max}")
    print(f"  Overlap thresholds: {args.overlap_thresholds}")
    
    all_tables = {}
    
    for overlap_threshold in args.overlap_thresholds:
        print(f"\n  Computing for overlap {overlap_threshold}%...")
        
        df = generate_metrics_table(
            detection_results=detection_results,
            protein_ground_truths=protein_ground_truths,
            k_values=k_values,
            overlap_threshold=overlap_threshold
        )
        
        csv_file, txt_file = save_table(df, output_dir, overlap_threshold)
        
        all_tables[overlap_threshold] = df
        
        print(f"    ✓ CSV saved: {csv_file}")
        print(f"    ✓ TXT saved: {txt_file}")
    
    # Save summary
    summary_file = output_dir / "evaluation_summary.json"
    
    summary = {
        "n_proteins": len(detection_results),
        "n_proteins_with_ground_truth": len(protein_ground_truths),
        "k_max": args.k_max,
        "overlap_thresholds": args.overlap_thresholds,
        "tables": {
            f"overlap_{int(thresh)}": {
                "csv": f"metrics_overlap_{int(thresh)}.csv",
                "txt": f"metrics_overlap_{int(thresh)}.txt"
            }
            for thresh in args.overlap_thresholds
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_file}")
    
    # Print sample results
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Proteins evaluated: {len(detection_results)}")
    print(f"K values: 1-{args.k_max}")
    print(f"Overlap thresholds: {args.overlap_thresholds}")
    print(f"\nResults saved to: {output_dir}/")
    
    # Show sample metrics
    print(f"\nSample Results (Overlap 80%, k=1,5,10,20):")
    if 80.0 in all_tables:
        df = all_tables[80.0]
        sample_k = [1, 5, 10, 20]
        sample_rows = df[df['k'].isin(sample_k)]
        
        print(f"\n{'k':<6} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 42)
        for _, row in sample_rows.iterrows():
            print(
                f"{int(row['k']):<6} "
                f"{row['precision']:<12.4f} "
                f"{row['recall']:<12.4f} "
                f"{row['f1']:<12.4f}"
            )
    
    print("\n✓ All tables generated successfully!")


if __name__ == "__main__":
    main()

