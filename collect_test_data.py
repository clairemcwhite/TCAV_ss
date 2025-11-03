#!/usr/bin/env python3
"""
Collect test dataset for evaluating motif detection system.

Collects 100 samples per Pfam motif, ensuring NO overlap with training data
(both positive and negative samples). Each test protein includes ground truth
annotations for ALL motifs we have trained CAVs for.

Usage:
    python collect_test_data.py --tcav-data-dir ./tcav_data --output-dir ./test_data --motifs-file motifs_list.json

Features:
- Excludes proteins from ALL training folders (pos and neg)
- Collects ALL motif annotations for each protein
- Ensures comprehensive ground truth for accurate evaluation
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Set, Any
from tqdm import tqdm
import time
import random

# UniProt REST API endpoint
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

# ----------------------------
# Helper Functions
# ----------------------------

def load_training_protein_ids(tcav_data_dir: str) -> Set[str]:
    """
    Load all protein IDs from training data (both pos and neg).
    Returns a set of accession IDs to exclude from test set.
    """
    excluded_ids = set()
    
    tcav_path = Path(tcav_data_dir)
    if not tcav_path.exists():
        raise FileNotFoundError(f"TCAV data directory not found: {tcav_data_dir}")
    
    # Find all PF* directories
    for motif_dir in sorted(tcav_path.iterdir()):
        if not motif_dir.is_dir() or not motif_dir.name.startswith("PF"):
            continue
        
        # Check both positive and negative JSONL files
        for jsonl_file in ["pos_100.jsonl", "neg_100.jsonl"]:
            jsonl_path = motif_dir / jsonl_file
            if jsonl_path.exists():
                try:
                    with open(jsonl_path, 'r') as f:
                        for line in f:
                            sample = json.loads(line.strip())
                            accession = sample.get("accession", "")
                            if accession:
                                excluded_ids.add(accession)
                except Exception as e:
                    print(f"[WARN] Error reading {jsonl_path}: {e}")
                    continue
    
    return excluded_ids


def get_trained_motif_list(tcav_data_dir: str) -> List[str]:
    """
    Get list of all motif IDs that have training data.
    Returns sorted list of PF IDs.
    """
    tcav_path = Path(tcav_data_dir)
    motif_ids = []
    
    for motif_dir in sorted(tcav_path.iterdir()):
        if not motif_dir.is_dir() or not motif_dir.name.startswith("PF"):
            continue
        
        # Check if it has positive training data
        pos_file = motif_dir / "pos_100.jsonl"
        if pos_file.exists():
            motif_ids.append(motif_dir.name)
    
    return sorted(motif_ids)


def query_uniprot_for_motif(motif_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Query UniProt for proteins containing a specific Pfam motif.
    
    Returns list of dicts with:
    - accession: UniProt ID
    - sequence: Protein sequence
    - length: Sequence length
    - protein_name: Name/description
    - gene_name: Gene name (if available)
    """
    results = []
    
    # Query UniProt for reviewed proteins with this Pfam domain
    query = f"(xref:pfam-{motif_id}) AND (reviewed:true)"
    
    params = {
        "query": query,
        "format": "json",
        "size": limit,
        "fields": "accession,sequence,length,protein_name,gene_primary,xref_pfam"
    }
    
    try:
        response = requests.get(UNIPROT_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        for entry in data.get("results", []):
            accession = entry.get("primaryAccession", "")
            sequence = entry.get("sequence", {}).get("value", "")
            length = entry.get("sequence", {}).get("length", 0)
            
            protein_name = ""
            if "proteinDescription" in entry:
                rec_name = entry["proteinDescription"].get("recommendedName", {})
                protein_name = rec_name.get("fullName", {}).get("value", "")
            
            gene_name = ""
            if "genes" in entry and len(entry["genes"]) > 0:
                gene_name = entry["genes"][0].get("geneName", {}).get("value", "")
            
            if accession and sequence and length > 0:
                results.append({
                    "accession": accession,
                    "sequence": sequence,
                    "length": length,
                    "protein_name": protein_name,
                    "gene_name": gene_name
                })
        
        return results
    
    except Exception as e:
        print(f"[ERROR] Failed to query UniProt for {motif_id}: {e}")
        return []


def get_all_pfam_annotations(accession: str, trained_motifs: List[str]) -> List[Dict[str, Any]]:
    """
    Get ALL Pfam annotations for a protein using InterPro API.
    
    InterPro provides direct Pfam domain locations without name matching.
    Only returns annotations for motifs we have trained CAVs for.
    
    Returns list of annotations:
    - motif_id: PF ID
    - start: 1-based start position
    - end: 1-based end position (inclusive)
    - name: Entry name
    """
    annotations = []
    
    # Use InterPro API which provides Pfam locations directly
    url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{accession}?format=json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # InterPro returns results grouped by entry
        results = data.get("results", [])
        
        for entry in results:
            metadata = entry.get("metadata", {})
            pfam_id = metadata.get("accession", "")
            pfam_name = metadata.get("name", "")
            
            if not pfam_id:
                continue
            
            # Only include if we have trained CAV for this motif
            if pfam_id not in trained_motifs:
                continue
            
            # Get protein matches with locations
            proteins = entry.get("proteins", [])
            for protein in proteins:
                # Case-insensitive match
                if protein.get("accession", "").lower() == accession.lower():
                    # Get all entry locations for this protein
                    entry_protein_locations = protein.get("entry_protein_locations", [])
                    
                    for loc_group in entry_protein_locations:
                        fragments = loc_group.get("fragments", [])
                        
                        for fragment in fragments:
                            start = fragment.get("start")
                            end = fragment.get("end")
                            
                            if start and end:
                                annotations.append({
                                    "motif_id": pfam_id,
                                    "start": int(start),
                                    "end": int(end),
                                    "name": pfam_name
                                })
        
        return annotations
    
    except Exception as e:
        print(f"[WARN] Failed to get annotations for {accession}: {e}")
        return []


def collect_test_samples_for_motif(
    motif_id: str,
    trained_motifs: List[str],
    excluded_proteins: Set[str],
    n_samples: int = 100,
    max_query: int = 500
) -> List[Dict[str, Any]]:
    """
    Collect test samples for a specific motif.
    
    Each sample includes:
    - accession
    - sequence
    - length
    - protein_name
    - gene_name
    - ground_truth_annotations: List of ALL motif annotations
    """
    print(f"\n[{motif_id}] Querying UniProt...")
    
    # Query UniProt for proteins with this motif
    candidates = query_uniprot_for_motif(motif_id, limit=max_query)
    print(f"[{motif_id}] Found {len(candidates)} candidates")
    
    # Filter out training proteins
    candidates = [c for c in candidates if c["accession"] not in excluded_proteins]
    print(f"[{motif_id}] After excluding training set: {len(candidates)} candidates")
    
    if len(candidates) == 0:
        print(f"[{motif_id}] No valid candidates found!")
        return []
    
    # Shuffle for random selection
    random.shuffle(candidates)
    
    # Collect samples with full annotations
    samples = []
    
    with tqdm(total=min(n_samples, len(candidates)), desc=f"Collecting {motif_id}") as pbar:
        for candidate in candidates:
            if len(samples) >= n_samples:
                break
            
            accession = candidate["accession"]
            
            # Get ALL Pfam annotations for this protein
            annotations = get_all_pfam_annotations(accession, trained_motifs)
            
            if len(annotations) == 0:
                # Skip if no annotations found (API issue)
                continue
            
            # Verify it has the target motif
            has_target = any(ann["motif_id"] == motif_id for ann in annotations)
            if not has_target:
                continue
            
            # Add sample with full ground truth
            sample = {
                "accession": accession,
                "sequence": candidate["sequence"],
                "length": candidate["length"],
                "protein_name": candidate["protein_name"],
                "gene_name": candidate["gene_name"],
                "target_motif": motif_id,  # The motif we queried for
                "ground_truth_annotations": annotations  # ALL motifs in this protein
            }
            
            samples.append(sample)
            pbar.update(1)
            
            # Rate limiting
            time.sleep(0.1)
    
    print(f"[{motif_id}] Collected {len(samples)} test samples")
    return samples


# ----------------------------
# Main Collection
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect test dataset for motif detection evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--tcav-data-dir", default="./tcav_data",
                       help="Directory with training data")
    parser.add_argument("--output-dir", default="./test_data",
                       help="Output directory for test data")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of test samples per motif (default: 5 for on-the-fly embedding)")
    parser.add_argument("--max-query", type=int, default=500,
                       help="Maximum proteins to query per motif")
    parser.add_argument("--motifs-file", default="motifs_list.json",
                       help="Optional motifs list JSON for metadata")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit to first N motifs (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true",
                       help="Force re-collection even if data already exists")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 80)
    print("TEST DATA COLLECTION FOR MOTIF DETECTION EVALUATION")
    print("=" * 80)
    
    # Load excluded protein IDs from training
    print("\n[1/4] Loading training protein IDs to exclude...")
    excluded_proteins = load_training_protein_ids(args.tcav_data_dir)
    print(f"✓ Loaded {len(excluded_proteins)} protein IDs from training set")
    
    # Get list of trained motifs
    print("\n[2/4] Finding trained motifs...")
    trained_motifs = get_trained_motif_list(args.tcav_data_dir)
    print(f"✓ Found {len(trained_motifs)} trained motifs: {', '.join(trained_motifs[:5])}...")
    
    if args.limit:
        trained_motifs = trained_motifs[:args.limit]
        print(f"  (Limited to first {args.limit} motifs)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect test data for each motif
    print(f"\n[3/4] Collecting {args.n_samples} test samples per motif...")
    
    all_results = {}
    collection_stats = {
        "total_motifs": len(trained_motifs),
        "successful": 0,
        "failed": 0,
        "total_samples": 0,
        "excluded_training_proteins": len(excluded_proteins),
        "per_motif_counts": {}
    }
    
    for motif_id in trained_motifs:
        # Check if already collected (unless --force is specified)
        motif_dir = Path(args.output_dir) / motif_id
        jsonl_file = motif_dir / "test_100.jsonl"
        
        if not args.force and jsonl_file.exists():
            try:
                # Count existing samples
                with open(jsonl_file, 'r') as f:
                    existing_samples = sum(1 for line in f if line.strip())
                
                # If we have at least 80% of target samples, skip
                if existing_samples >= int(args.n_samples * 0.8):
                    all_results[motif_id] = {
                        "n_samples": existing_samples,
                        "status": "skipped_existing"
                    }
                    collection_stats["successful"] += 1
                    collection_stats["total_samples"] += existing_samples
                    collection_stats["per_motif_counts"][motif_id] = existing_samples
                    print(f"⊘ {motif_id}: Already collected ({existing_samples} samples)")
                    continue
            except Exception as e:
                print(f"[WARN] {motif_id}: Error checking existing data, will re-collect - {e}")
        
        try:
            samples = collect_test_samples_for_motif(
                motif_id=motif_id,
                trained_motifs=trained_motifs,
                excluded_proteins=excluded_proteins,
                n_samples=args.n_samples,
                max_query=args.max_query
            )
            
            if len(samples) > 0:
                # Save to motif-specific directory
                motif_dir = Path(args.output_dir) / motif_id
                motif_dir.mkdir(parents=True, exist_ok=True)
                
                # Save JSONL
                jsonl_file = motif_dir / "test_100.jsonl"
                with open(jsonl_file, 'w') as f:
                    for sample in samples:
                        f.write(json.dumps(sample) + '\n')
                
                # Save FASTA
                fasta_file = motif_dir / "test_100.fasta"
                with open(fasta_file, 'w') as f:
                    for sample in samples:
                        header = f">{sample['accession']} {sample['protein_name']}"
                        f.write(f"{header}\n{sample['sequence']}\n")
                
                all_results[motif_id] = {
                    "n_samples": len(samples),
                    "status": "success"
                }
                
                collection_stats["successful"] += 1
                collection_stats["total_samples"] += len(samples)
                collection_stats["per_motif_counts"][motif_id] = len(samples)
                
                print(f"✓ {motif_id}: {len(samples)} samples saved")
            else:
                all_results[motif_id] = {
                    "n_samples": 0,
                    "status": "no_samples"
                }
                collection_stats["failed"] += 1
                print(f"✗ {motif_id}: No samples collected")
        
        except Exception as e:
            print(f"✗ {motif_id}: Error - {e}")
            all_results[motif_id] = {
                "n_samples": 0,
                "status": "error",
                "error": str(e)
            }
            collection_stats["failed"] += 1
        
        # Rate limiting between motifs
        time.sleep(1)
    
    # Save summary
    print("\n[4/4] Saving collection summary...")
    
    summary_file = Path(args.output_dir) / "test_collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "stats": collection_stats,
            "results": all_results,
            "trained_motifs": trained_motifs
        }, f, indent=2)
    
    print(f"✓ Summary saved to {summary_file}")
    
    # Print final stats
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total motifs:         {collection_stats['total_motifs']}")
    print(f"Successful:           {collection_stats['successful']}")
    print(f"Failed:               {collection_stats['failed']}")
    print(f"Total test samples:   {collection_stats['total_samples']}")
    print(f"Excluded proteins:    {collection_stats['excluded_training_proteins']}")
    print(f"\n✓ Test data ready for evaluation!")
    print(f"\nNext steps:")
    print(f"  1. Generate embeddings: python generate_test_embeddings.py")
    print(f"  2. Run detection: python run_test_detection.py")
    print(f"  3. Evaluate metrics: python evaluate_metrics.py")


if __name__ == "__main__":
    main()

