#!/usr/bin/env python3
"""
Create a smaller test set by taking first N samples from each motif.

Takes existing test_data and creates test_data_small with fewer samples per motif.

Usage:
    python create_small_test_set.py --input-dir ./test_data --output-dir ./test_data_small --n-samples 5
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def sample_motif_data(input_dir: Path, output_dir: Path, motif_id: str, n_samples: int, max_length: int = 1000) -> int:
    """
    Copy first n_samples from a motif folder (filtering by max_length).
    
    Returns number of samples copied.
    """
    # Input files
    input_jsonl = input_dir / motif_id / "test_100.jsonl"
    
    if not input_jsonl.exists():
        return 0
    
    # Create output directory
    output_motif_dir = output_dir / motif_id
    output_motif_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and sample (filtering by length)
    samples = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            if len(samples) >= n_samples:
                break
            if line.strip():
                sample = json.loads(line.strip())
                # Filter by length
                if sample.get("length", 0) <= max_length:
                    samples.append(sample)
    
    # Write JSONL
    output_jsonl = output_motif_dir / "test_100.jsonl"
    with open(output_jsonl, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    # Write FASTA
    output_fasta = output_motif_dir / "test_100.fasta"
    with open(output_fasta, 'w') as f:
        for sample in samples:
            f.write(f">{sample['accession']}\n")
            f.write(f"{sample['sequence']}\n")
    
    return len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Create smaller test set from existing test data"
    )
    
    parser.add_argument("--input-dir", default="./test_data",
                       help="Input test data directory")
    parser.add_argument("--output-dir", default="./test_data_small",
                       help="Output directory for small test set")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of samples to take per motif")
    parser.add_argument("--max-length", type=int, default=1000,
                       help="Maximum protein length to include (default: 1000)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CREATING SMALL TEST SET")
    print("="*80)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Samples per motif: {args.n_samples}")
    print(f"Max protein length: {args.max_length} aa")
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"\n[ERROR] Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each motif
    print(f"\nProcessing motifs...")
    
    total_motifs = 0
    total_samples = 0
    stats = {}
    
    for motif_dir in sorted(input_dir.iterdir()):
        if not motif_dir.is_dir() or not motif_dir.name.startswith("PF"):
            continue
        
        motif_id = motif_dir.name
        n_copied = sample_motif_data(input_dir, output_dir, motif_id, args.n_samples, args.max_length)
        
        if n_copied > 0:
            total_motifs += 1
            total_samples += n_copied
            stats[motif_id] = n_copied
            if n_copied < args.n_samples:
                print(f"  {motif_id}: copied {n_copied} samples (WARNING: less than {args.n_samples}, likely filtered by length)")
            else:
                print(f"  {motif_id}: copied {n_copied} samples")
    
    # Save summary
    summary_file = output_dir / "subset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "source_dir": str(input_dir),
            "n_samples_per_motif": args.n_samples,
            "max_length": args.max_length,
            "total_motifs": total_motifs,
            "total_samples": total_samples,
            "per_motif": stats
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Total motifs: {total_motifs}")
    print(f"Total samples: {total_samples}")
    print(f"Average per motif: {total_samples/total_motifs:.1f}")
    print(f"Length filter: ≤{args.max_length} aa")
    print(f"\nSmall test set saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")
    
    print(f"\n✓ You can now run:")
    print(f"  python run_test_detection_onthefly.py --test-data-dir {output_dir}")


if __name__ == "__main__":
    main()

