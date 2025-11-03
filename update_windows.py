#!/usr/bin/env python3
"""
Update window coordinates in existing data to use motif-specific window sizes.

For each motif:
1. Calculate median domain length from positives
2. Set window_length = median + 5 (buffer)
3. Recenter windows with new length
4. Update JSONL files in place
5. Save window_length to metadata.json

Usage:
  python update_windows.py --data-dir ./tcav_data
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Constraints
MIN_WINDOW = 20
MAX_WINDOW = 300
BUFFER = 10

def center_window(i0: int, i1: int, L: int, win: int) -> Tuple[int, int]:
    """
    Domain span [i0, i1] is 1-based inclusive.
    Return 0-based half-open [s, e) window of length win, clipped inside [0, L].
    """
    start0 = max(0, i0 - 1)
    end0 = min(L - 1, i1 - 1)
    center = (start0 + end0) // 2
    half = win // 2
    s = max(0, center - half)
    e = s + win
    if e > L:
        e = L
        s = max(0, L - win)
    return s, e

def sample_neg_window(L: int, win: int) -> Tuple[int, int]:
    """Sample random window for negatives."""
    import random
    max_residue_end = min(L, 1021)  # ESM limit
    if max_residue_end < win:
        return (0, L)
    s_max = max_residue_end - win
    if s_max < 0:
        return (0, min(L, win))
    s = random.randint(0, s_max)
    e = s + win
    return (s, e)

def calculate_optimal_window(motif_dir: Path) -> int:
    """
    Calculate optimal window size for a motif based on domain lengths.
    Returns median_domain_length + BUFFER, clamped to [MIN_WINDOW, MAX_WINDOW].
    """
    # Load positive samples
    pos_files = list(motif_dir.glob("pos_*.jsonl"))
    if not pos_files:
        return 40  # fallback
    
    with open(pos_files[0], 'r') as f:
        pos_samples = [json.loads(line) for line in f]
    
    # Extract domain lengths
    domain_lengths = []
    for sample in pos_samples:
        span = sample.get("domain_span_1based_inclusive")
        if span and len(span) == 2:
            length = span[1] - span[0] + 1
            domain_lengths.append(length)
    
    if not domain_lengths:
        return 40  # fallback
    
    # Calculate median + buffer
    median_length = int(np.median(domain_lengths))
    optimal_window = median_length + BUFFER
    
    # Clamp to constraints
    optimal_window = max(MIN_WINDOW, min(MAX_WINDOW, optimal_window))
    
    print(f"  Domain lengths: min={min(domain_lengths)}, median={median_length}, max={max(domain_lengths)}")
    print(f"  → Optimal window: {optimal_window} aa (median {median_length} + buffer {BUFFER})")
    
    return optimal_window

def update_motif_windows(motif_dir: Path, window_length: int) -> Dict[str, int]:
    """
    Update window coordinates for a motif's positive and negative samples.
    Returns stats.
    """
    stats = {"pos_updated": 0, "neg_updated": 0}
    
    # Update positives
    pos_files = list(motif_dir.glob("pos_*.jsonl"))
    for pos_file in pos_files:
        with open(pos_file, 'r') as f:
            pos_samples = [json.loads(line) for line in f]
        
        for sample in pos_samples:
            # Recalculate window centered on domain
            domain_span = sample.get("domain_span_1based_inclusive")
            if domain_span and len(domain_span) == 2:
                seq_len = sample["sequence_length"]
                w_s, w_e = center_window(domain_span[0], domain_span[1], seq_len, window_length)
                
                # Update
                sample["window_span_0based_halfopen"] = [w_s, w_e]
                sample["window_length"] = w_e - w_s
                stats["pos_updated"] += 1
        
        # Write back
        with open(pos_file, 'w') as f:
            for sample in pos_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Update negatives
    neg_files = list(motif_dir.glob("neg_*.jsonl"))
    for neg_file in neg_files:
        with open(neg_file, 'r') as f:
            neg_samples = [json.loads(line) for line in f]
        
        for sample in neg_samples:
            # Resample random window with new length
            seq_len = sample["sequence_length"]
            if seq_len < window_length:
                w_s, w_e = 0, seq_len
            else:
                w_s, w_e = sample_neg_window(seq_len, window_length)
            
            # Update
            sample["window_span_0based_halfopen"] = [w_s, w_e]
            sample["window_length"] = w_e - w_s
            stats["neg_updated"] += 1
        
        # Write back
        with open(neg_file, 'w') as f:
            for sample in neg_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    return stats

def update_metadata(motif_dir: Path, window_length: int):
    """Update metadata.json with new window_length."""
    metadata_file = motif_dir / "metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    metadata["window_length"] = window_length
    metadata["window_updated"] = True
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Update window sizes in collected data")
    parser.add_argument("--data-dir", required=True, help="Data directory (tcav_data/)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying files")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return
    
    # Find all motif directories
    motif_dirs = [d for d in sorted(data_path.iterdir()) if d.is_dir() and d.name.startswith("PF")]
    
    print("=" * 70)
    print("UPDATING WINDOW SIZES")
    print("=" * 70)
    print(f"Data directory: {data_path}")
    print(f"Motifs found: {len(motif_dirs)}")
    print(f"Buffer: {BUFFER} aa")
    print(f"Constraints: {MIN_WINDOW}-{MAX_WINDOW} aa")
    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be modified]")
    print()
    
    summary = []
    
    for motif_dir in motif_dirs:
        motif_id = motif_dir.name
        print(f"\n{'='*70}")
        print(f"Processing: {motif_id}")
        print(f"{'='*70}")
        
        try:
            # Calculate optimal window
            optimal_window = calculate_optimal_window(motif_dir)
            
            if args.dry_run:
                print(f"  [DRY RUN] Would set window_length={optimal_window}")
                summary.append({
                    "motif": motif_id,
                    "window_length": optimal_window,
                    "status": "dry_run"
                })
                continue
            
            # Update windows
            print(f"  Updating windows to {optimal_window} aa...")
            stats = update_motif_windows(motif_dir, optimal_window)
            
            # Update metadata
            update_metadata(motif_dir, optimal_window)
            
            print(f"  ✓ Updated {stats['pos_updated']} positives, {stats['neg_updated']} negatives")
            
            summary.append({
                "motif": motif_id,
                "window_length": optimal_window,
                "pos_updated": stats["pos_updated"],
                "neg_updated": stats["neg_updated"],
                "status": "success"
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            summary.append({
                "motif": motif_id,
                "status": "failed",
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if args.dry_run:
        print("\nProposed window sizes:")
        for s in summary:
            if s["status"] == "dry_run":
                print(f"  {s['motif']}: {s['window_length']} aa")
    else:
        successful = [s for s in summary if s["status"] == "success"]
        failed = [s for s in summary if s["status"] == "failed"]
        
        print(f"\nTotal motifs: {len(motif_dirs)}")
        print(f"  ✓ Updated: {len(successful)}")
        print(f"  ✗ Failed: {len(failed)}")
        
        if successful:
            print("\nWindow size distribution:")
            windows = [s["window_length"] for s in successful]
            print(f"  Min: {min(windows)} aa")
            print(f"  Median: {int(np.median(windows))} aa")
            print(f"  Max: {max(windows)} aa")
        
        # Save summary
        summary_file = data_path / "window_update_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Summary saved to: {summary_file}")
        
        print("\nNext steps:")
        print("  1. Review the updated windows")
        print("  2. Retrain CAVs with: python batch_train_cavs.py ...")
        print("     (Embeddings will be regenerated with new window sizes)")

if __name__ == "__main__":
    main()

