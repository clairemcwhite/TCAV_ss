#!/usr/bin/env python3
"""
Quick script to check window sizes for all motifs in tcav_data.

Usage:
    python check_window_sizes.py
    python check_window_sizes.py --threshold 100
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def check_window_sizes(tcav_data_dir: str, threshold: int = 100):
    """Check window sizes for all motifs."""
    
    tcav_path = Path(tcav_data_dir)
    
    if not tcav_path.exists():
        print(f"ERROR: Directory not found: {tcav_data_dir}")
        return
    
    results = []
    size_distribution = defaultdict(int)
    
    # Scan all PF* directories
    for motif_dir in sorted(tcav_path.iterdir()):
        if not motif_dir.is_dir() or not motif_dir.name.startswith("PF"):
            continue
        
        motif_id = motif_dir.name
        metadata_file = motif_dir / "metadata.json"
        
        if not metadata_file.exists():
            print(f"⚠️  {motif_id}: No metadata.json found")
            continue
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            window_length = metadata.get("window_length", None)
            
            if window_length is None:
                print(f"⚠️  {motif_id}: No window_length in metadata")
                continue
            
            results.append({
                "motif_id": motif_id,
                "window_length": window_length
            })
            
            # Track distribution
            size_distribution[window_length] += 1
            
        except Exception as e:
            print(f"✗ {motif_id}: Error reading metadata - {e}")
            continue
    
    # Sort by window length
    results.sort(key=lambda x: x["window_length"])
    
    # Count by threshold
    below_threshold = [r for r in results if r["window_length"] < threshold]
    at_or_above_threshold = [r for r in results if r["window_length"] >= threshold]
    
    # Print results
    print("=" * 80)
    print("WINDOW SIZE ANALYSIS")
    print("=" * 80)
    print(f"Total motifs analyzed: {len(results)}")
    print(f"Window size threshold: {threshold}")
    print()
    
    print(f"📊 Distribution:")
    print(f"  < {threshold}:  {len(below_threshold)} motifs")
    print(f"  ≥ {threshold}: {len(at_or_above_threshold)} motifs")
    print()
    
    # Show unique window sizes
    print(f"📏 Unique window sizes:")
    for size in sorted(size_distribution.keys()):
        count = size_distribution[size]
        print(f"  {size:3d}: {count:2d} motifs")
    print()
    
    # Show motifs below threshold
    if below_threshold:
        print(f"🔍 Motifs with window_length < {threshold}:")
        print(f"  {'Motif ID':<12} {'Window Size':<12}")
        print(f"  {'-'*24}")
        for r in below_threshold:
            print(f"  {r['motif_id']:<12} {r['window_length']:<12}")
    else:
        print(f"✓ No motifs found with window_length < {threshold}")
    
    print()
    
    # Show motifs at/above threshold
    if at_or_above_threshold and len(at_or_above_threshold) <= 20:
        print(f"📋 Motifs with window_length ≥ {threshold}:")
        print(f"  {'Motif ID':<12} {'Window Size':<12}")
        print(f"  {'-'*24}")
        for r in at_or_above_threshold:
            print(f"  {r['motif_id']:<12} {r['window_length']:<12}")
    elif at_or_above_threshold:
        print(f"📋 {len(at_or_above_threshold)} motifs with window_length ≥ {threshold} (showing first 10):")
        print(f"  {'Motif ID':<12} {'Window Size':<12}")
        print(f"  {'-'*24}")
        for r in at_or_above_threshold[:10]:
            print(f"  {r['motif_id']:<12} {r['window_length']:<12}")
        print(f"  ... and {len(at_or_above_threshold) - 10} more")
    
    print()
    print("=" * 80)
    
    # Summary stats
    if results:
        window_sizes = [r["window_length"] for r in results]
        avg_size = sum(window_sizes) / len(window_sizes)
        min_size = min(window_sizes)
        max_size = max(window_sizes)
        
        print(f"📈 Statistics:")
        print(f"  Average window size: {avg_size:.1f}")
        print(f"  Min window size: {min_size}")
        print(f"  Max window size: {max_size}")
        print(f"  Range: {max_size - min_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check window sizes for all motifs in tcav_data"
    )
    parser.add_argument("--tcav-data-dir", default="./tcav_data",
                       help="TCAV data directory")
    parser.add_argument("--threshold", type=int, default=100,
                       help="Window size threshold for comparison")
    
    args = parser.parse_args()
    
    check_window_sizes(args.tcav_data_dir, args.threshold)




