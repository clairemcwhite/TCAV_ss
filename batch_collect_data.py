#!/usr/bin/env python3
"""
Batch data collection for all motifs in motifs_list.json.

Systematically collects training data for TCAV across all selected motifs.

Usage:
  python batch_collect_data.py --motifs-file motifs_list.json --output-dir ./tcav_data

Features:
- Parallel processing support
- Resume capability (skips already collected motifs)
- Progress tracking and summary statistics
- Error handling per motif (continues on failure)
"""

import os
import json
import argparse
import subprocess
import time
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Config
# ----------------------------
DEFAULT_POS_N = 100
DEFAULT_NEG_N = 100
DEFAULT_WINDOW_LEN = 40
DEFAULT_MAX_PROTEINS = 2000
DEFAULT_WORKERS = 5
SEED = 42

# ----------------------------
# Collection worker
# ----------------------------
def collect_motif_data(motif_id: str, output_dir: str, pos_n: int, neg_n: int, 
                       window_len: int, seed: int, max_proteins: int, workers: int) -> Dict[str, Any]:
    """
    Run data collection for a single motif.
    Returns status dict.
    """
    motif_dir = os.path.join(output_dir, motif_id)
    metadata_file = os.path.join(motif_dir, "metadata.json")
    
    # Check if already collected
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                meta = json.load(f)
            # Verify it has reasonable data
            if meta.get("positives_collected", 0) >= 50:  # Minimum threshold
                return {
                    "motif_id": motif_id,
                    "status": "skipped",
                    "reason": "already_collected",
                    "positives": meta.get("positives_collected", 0),
                    "negatives": meta.get("negatives_collected", 0)
                }
        except Exception:
            pass  # Re-collect if metadata is corrupted
    
    # Run collection script
    cmd = [
        "python", "collect_data_motif.py",
        "--motif", motif_id,
        "--pos-n", str(pos_n),
        "--neg-n", str(neg_n),
        "--window-len", str(window_len),
        "--output-dir", output_dir,
        "--seed", str(seed),
        "--max-proteins", str(max_proteins),
        "--workers", str(workers)
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per motif
        )
        elapsed = time.time() - start_time
        
        # Check if successful by looking for metadata file
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                meta = json.load(f)
            
            pos_count = meta.get("positives_collected", 0)
            neg_count = meta.get("negatives_collected", 0)
            
            if pos_count >= 50:  # Minimum viable threshold
                return {
                    "motif_id": motif_id,
                    "status": "success",
                    "positives": pos_count,
                    "negatives": neg_count,
                    "elapsed_seconds": elapsed
                }
            else:
                return {
                    "motif_id": motif_id,
                    "status": "insufficient_data",
                    "positives": pos_count,
                    "negatives": neg_count,
                    "elapsed_seconds": elapsed
                }
        else:
            return {
                "motif_id": motif_id,
                "status": "failed",
                "error": "metadata file not created",
                "stderr": result.stderr[-500:] if result.stderr else "",
                "elapsed_seconds": elapsed
            }
    
    except subprocess.TimeoutExpired:
        return {
            "motif_id": motif_id,
            "status": "timeout",
            "error": "Collection timed out after 600s"
        }
    except Exception as e:
        return {
            "motif_id": motif_id,
            "status": "error",
            "error": str(e)
        }

# ----------------------------
# Batch processing
# ----------------------------
def batch_collect_sequential(motifs: List[Dict[str, Any]], output_dir: str, 
                             pos_n: int, neg_n: int, window_len: int, seed: int, max_proteins: int, workers: int) -> List[Dict[str, Any]]:
    """
    Collect data for all motifs sequentially.
    """
    results = []
    
    for motif in tqdm(motifs, desc="Collecting motif data"):
        motif_id = motif["accession"]
        result = collect_motif_data(motif_id, output_dir, pos_n, neg_n, window_len, seed, max_proteins, workers)
        results.append(result)
        
        # Print status
        status = result["status"]
        if status == "success":
            print(f"✓ {motif_id}: {result['positives']} pos, {result['negatives']} neg")
        elif status == "skipped":
            print(f"⊘ {motif_id}: Already collected")
        elif status == "insufficient_data":
            print(f"⚠ {motif_id}: Insufficient data ({result['positives']} positives)")
        else:
            print(f"✗ {motif_id}: {status} - {result.get('error', 'unknown error')}")
    
    return results

def batch_collect_parallel(motifs: List[Dict[str, Any]], output_dir: str, 
                           pos_n: int, neg_n: int, window_len: int, seed: int, 
                           max_proteins: int, workers: int, max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Collect data for all motifs in parallel.
    """
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for motif in motifs:
            motif_id = motif["accession"]
            future = executor.submit(
                collect_motif_data, motif_id, output_dir, pos_n, neg_n, window_len, seed, max_proteins, workers
            )
            futures[future] = motif_id
        
        # Progress bar
        with tqdm(total=len(futures), desc="Collecting motif data") as pbar:
            for future in as_completed(futures):
                motif_id = futures[future]
                result = future.result()
                results.append(result)
                
                # Print status
                status = result["status"]
                if status == "success":
                    print(f"✓ {motif_id}: {result['positives']} pos, {result['negatives']} neg")
                elif status == "skipped":
                    print(f"⊘ {motif_id}: Already collected")
                elif status == "insufficient_data":
                    print(f"⚠ {motif_id}: Insufficient data ({result['positives']} positives)")
                else:
                    print(f"✗ {motif_id}: {status}")
                
                pbar.update(1)
    
    return results

# ----------------------------
# Summary reporting
# ----------------------------
def print_summary(results: List[Dict[str, Any]], output_file: str = None):
    """
    Print and save summary statistics.
    """
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    insufficient = sum(1 for r in results if r["status"] == "insufficient_data")
    failed = total - success - skipped - insufficient
    
    total_pos = sum(r.get("positives", 0) for r in results if r["status"] in ["success", "skipped"])
    total_neg = sum(r.get("negatives", 0) for r in results if r["status"] in ["success", "skipped"])
    
    summary = {
        "total_motifs": total,
        "successful": success,
        "skipped": skipped,
        "insufficient_data": insufficient,
        "failed": failed,
        "total_positives_collected": total_pos,
        "total_negatives_collected": total_neg,
        "results": results
    }
    
    print("\n" + "=" * 70)
    print("BATCH COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Total motifs processed:    {total}")
    print(f"  ✓ Successful:            {success}")
    print(f"  ⊘ Skipped (existing):    {skipped}")
    print(f"  ⚠ Insufficient data:     {insufficient}")
    print(f"  ✗ Failed:                {failed}")
    print(f"\nTotal sequences collected:")
    print(f"  Positives:               {total_pos:,}")
    print(f"  Negatives:               {total_neg:,}")
    print(f"\nUsable motifs for TCAV:  {success + skipped}")
    
    # Save summary
    if output_file:
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Summary saved to: {output_file}")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch collect TCAV data for multiple motifs")
    parser.add_argument("--motifs-file", default="motifs_list.json", 
                       help="JSON file with motif list")
    parser.add_argument("--output-dir", default="./tcav_data", 
                       help="Output directory for all motif data")
    parser.add_argument("--pos-n", type=int, default=DEFAULT_POS_N, 
                       help="Number of positive examples per motif")
    parser.add_argument("--neg-n", type=int, default=DEFAULT_NEG_N, 
                       help="Number of negative examples per motif")
    parser.add_argument("--window-len", type=int, default=DEFAULT_WINDOW_LEN, 
                       help="Window length")
    parser.add_argument("--seed", type=int, default=SEED, 
                       help="Random seed")
    parser.add_argument("--max-proteins", type=int, default=DEFAULT_MAX_PROTEINS,
                       help="Maximum proteins to check per motif (prevents slowdown on abundant motifs)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                       help="Number of parallel workers for FASTA fetching within each motif (default 10)")
    parser.add_argument("--parallel", action="store_true", 
                       help="Use parallel processing for multiple motifs")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Number of parallel motif collection processes (only with --parallel)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit to first N motifs (for testing)")
    
    args = parser.parse_args()
    
    # Load motifs
    if not os.path.exists(args.motifs_file):
        print(f"Error: Motifs file not found: {args.motifs_file}")
        print("Run fetch_motifs.py first to generate the motif list.")
        return
    
    with open(args.motifs_file, "r") as f:
        data = json.load(f)
    
    motifs = data["motifs"]
    if args.limit:
        motifs = motifs[:args.limit]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("TCAV BATCH DATA COLLECTION")
    print("=" * 70)
    print(f"Motifs to process:  {len(motifs)}")
    print(f"Target per motif:   {args.pos_n} pos, {args.neg_n} neg")
    print(f"Max proteins check: {args.max_proteins} per motif")
    print(f"FASTA workers:      {args.workers} per motif")
    print(f"Output directory:   {args.output_dir}")
    print(f"Processing mode:    {'Parallel' if args.parallel else 'Sequential'}")
    if args.parallel:
        print(f"Motif parallelism:  {args.max_workers} motifs at once")
    print()
    
    # Collect data
    if args.parallel:
        results = batch_collect_parallel(
            motifs, args.output_dir, args.pos_n, args.neg_n, 
            args.window_len, args.seed, args.max_proteins, args.workers, args.max_workers
        )
    else:
        results = batch_collect_sequential(
            motifs, args.output_dir, args.pos_n, args.neg_n, 
            args.window_len, args.seed, args.max_proteins, args.workers
        )
    
    # Print summary
    summary_file = os.path.join(args.output_dir, "collection_summary.json")
    print_summary(results, summary_file)
    
    print("\n✓ Batch collection complete!")
    print(f"\nNext steps:")
    print(f"  1. Review summary: {summary_file}")
    print(f"  2. Train CAVs for each motif using the collected data")
    print(f"  3. Run TCAV analysis on your target proteins")

if __name__ == "__main__":
    main()

