#!/usr/bin/env python3
"""
Test a single protein against ALL trained TCAVs and rank motifs by top z-mean score.

This script:
1. Loads one protein sequence
2. For EACH motif (using its specific window size):
   - Generates windows with that motif's window size
   - Extracts embeddings for those windows
   - Computes CAV scores and z-scores (normalized within motif)
   - Gets the top z-mean (ensemble across layers)
3. Ranks all motifs by their top z-mean score
4. Displays top 10 motifs

Usage:
  python batch_detect_all_motifs.py --fasta my_protein.fasta
  python batch_detect_all_motifs.py --seq ACDEFGHIKLMNPQRSTVWY --header "MyProtein"
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import EsmModel, EsmTokenizer
from joblib import load as joblib_load
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add esm2_tcav to path for model loading
sys.path.insert(0, str(Path(__file__).parent / "esm2_tcav"))

# -------------------------
# Config
# -------------------------
DEFAULT_TCAV_DIR = "./tcav_outputs"
DEFAULT_DATA_DIR = "./tcav_data"
DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEFAULT_LAYERS = [2, 4, 6]
DEFAULT_WINDOW_LEN = 40

AA_VALID = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

# -------------------------
# Helpers
# -------------------------
def clean_seq(seq: str) -> str:
    """Clean and validate sequence."""
    seq = re.sub(r"\s+", "", seq.strip().upper())
    return "".join(ch if ch in AA_VALID else "X" for ch in seq)

def generate_windows(L, win, stride):
    """Generate [start, end) spans for sliding windows."""
    for s in range(0, max(1, L - win + 1), stride):
        e = min(L, s + win)
        if e - s < win:
            break
        yield s, e

def mean_pool_from_hidden(hidden_states, attention_mask, layer_idx_hf):
    """Extract mean-pooled embeddings from hidden states."""
    H = hidden_states[layer_idx_hf]  # (B, T, d)
    H_no_special = H[:, 1:-1, :]  # Exclude BOS/EOS
    mask_no_special = attention_mask[:, 1:-1]
    denom = torch.clamp(mask_no_special.sum(dim=1, keepdim=True), min=1).unsqueeze(-1)
    pooled = (H_no_special * mask_no_special.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
    return pooled.squeeze(1)

def load_motif_names(motifs_list_path: str = "motifs_list.json") -> Dict[str, str]:
    """
    Load motif ID to name mapping from motifs_list.json.
    Returns dict mapping accession -> name.
    """
    try:
        with open(motifs_list_path, 'r') as f:
            data = json.load(f)
        
        name_map = {}
        for motif in data.get("motifs", []):
            accession = motif.get("accession")
            name = motif.get("name", motif.get("short_name", ""))
            if accession:
                name_map[accession] = name
        
        return name_map
    except Exception as e:
        # If file not found, return empty dict
        return {}

def find_all_motifs(tcav_dir: str, data_dir: str, name_map: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Find all trained motifs in tcav_outputs/cavs/.
    Returns list of dicts with motif_id, motif_name, cav_dir, window_length.
    """
    cav_base = Path(tcav_dir) / "cavs"
    
    if not cav_base.exists():
        raise FileNotFoundError(f"CAV directory not found: {cav_base}")
    
    if name_map is None:
        name_map = {}
    
    motifs = []
    
    for motif_dir in sorted(cav_base.iterdir()):
        if not motif_dir.is_dir() or not motif_dir.name.startswith("PF"):
            continue
        
        motif_id = motif_dir.name
        motif_name = name_map.get(motif_id, "")
        
        # Try to get window_length from metadata
        window_length = DEFAULT_WINDOW_LEN
        metadata_file = Path(data_dir) / motif_id / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    window_length = metadata.get("window_length", DEFAULT_WINDOW_LEN)
            except:
                pass
        
        motifs.append({
            "motif_id": motif_id,
            "motif_name": motif_name,
            "cav_dir": str(motif_dir),
            "window_length": window_length
        })
    
    return motifs

def load_layer_artifacts(cav_dir: str, layer: int, version: str = "v1"):
    """Load CAV vector and scaler for a specific layer."""
    cav_file = Path(cav_dir) / f"L{layer}_concept_{version}.npy"
    scaler_file = Path(cav_dir) / f"L{layer}_scaler_{version}.pkl"
    
    if not cav_file.exists() or not scaler_file.exists():
        return None, None
    
    try:
        cav = np.load(cav_file)
        scaler = joblib_load(scaler_file)
        return cav, scaler
    except:
        return None, None

def detect_motif(
    seq: str,
    motif_info: Dict[str, Any],
    model,
    tokenizer,
    layers: List[int],
    device: str,
    version: str = "v1"
) -> Dict[str, Any]:
    """
    Detect a single motif in the sequence (mimics detect_motif.py).
    
    Returns dict with:
    - top_zmean: Best z-mean score across all windows (ensemble)
    - top_window_span: [start, end] of best window
    - top_window_seq: Sequence of best window
    - per_layer_top: Top hit per layer
    - per_layer_max_score: Max score per layer (for ranking)
    - per_layer_max_zscore: Max z-score per layer (for ranking)
    """
    motif_id = motif_info["motif_id"]
    cav_dir = motif_info["cav_dir"]
    window_len = motif_info["window_length"]
    stride = window_len // 2
    
    Lseq = len(seq)
    
    # Generate windows for this motif's window size
    if Lseq < window_len:
        spans = [(0, Lseq)]
    else:
        spans = list(generate_windows(Lseq, window_len, stride))
    
    if not spans:
        return None
    
    # Tokenize windows
    seq_windows = [seq[s:e] for (s, e) in spans]
    enc = tokenizer(seq_windows, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    
    # Extract embeddings and compute scores
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        
        per_layer = {}
        
        for L in layers:
            cav, scaler = load_layer_artifacts(cav_dir, L, version)
            
            if cav is None or scaler is None:
                continue
            
            # Pool embeddings
            pooled = mean_pool_from_hidden(hs, enc["attention_mask"], layer_idx_hf=L)
            X = pooled.detach().cpu().numpy()
            
            # Standardize and score
            try:
                Xs = scaler.transform(X)
                scores = Xs @ cav
                
                # Compute z-scores (normalized within this motif's windows)
                zscores = (scores - scores.mean()) / (scores.std() + 1e-8)
                
                per_layer[L] = {
                    "scores": scores,
                    "zscores": zscores
                }
            except:
                continue
    
    if not per_layer:
        return None
    
    # Compute ensemble z-mean across layers
    zmat = np.stack([per_layer[L]["zscores"] for L in per_layer.keys()], axis=1)
    zmean = zmat.mean(axis=1)  # (n_windows,)
    
    # Get per-layer max values (for alternative ranking if needed)
    per_layer_max_score = {}
    per_layer_max_zscore = {}
    
    for L in per_layer.keys():
        scores = per_layer[L]["scores"]
        zscores = per_layer[L]["zscores"]
        per_layer_max_score[f"L{L}"] = float(scores.max())
        per_layer_max_zscore[f"L{L}"] = float(zscores.max())
    
    # Return ALL windows with their scores
    all_hits = []
    for i, (start, end) in enumerate(spans):
        hit = {
            "motif_id": motif_id,
            "motif_name": motif_info.get("motif_name", ""),
            "window_length": window_len,
            "span": [start, end],
            "sequence": seq[start:end],
            "zmean": float(zmean[i]),
            "per_layer_scores": {f"L{L}": float(per_layer[L]["scores"][i]) for L in per_layer.keys()},
            "per_layer_zscores": {f"L{L}": float(per_layer[L]["zscores"][i]) for L in per_layer.keys()}
        }
        all_hits.append(hit)
    
    return {
        "motif_id": motif_id,
        "motif_name": motif_info.get("motif_name", ""),
        "window_length": window_len,
        "n_windows": len(spans),
        "n_layers": len(per_layer),
        "per_layer_max_score": per_layer_max_score,
        "per_layer_max_zscore": per_layer_max_zscore,
        "all_hits": all_hits  # All windows with their z-means
    }

# -------------------------
# Main
# -------------------------
def main(args):
    # Parse input sequence
    if args.seq:
        header = args.header or "query"
        seq = clean_seq(args.seq)
    else:
        with open(args.fasta) as f:
            raw = f.read().strip().splitlines()
        if raw[0].startswith(">"):
            header = raw[0][1:].strip()
            seq = clean_seq("".join(raw[1:]))
        else:
            header = "query"
            seq = clean_seq("".join(raw))
    
    Lseq = len(seq)
    
    # Parse motif filter list
    motif_filter = None
    if args.motifs:
        motif_filter = set(args.motifs)
    elif args.motifs_file:
        with open(args.motifs_file, 'r') as f:
            motif_filter = set(line.strip() for line in f if line.strip())
    
    # Determine if we're filtering motifs
    filter_mode = motif_filter is not None
    
    print(f"\n{'='*80}")
    if filter_mode:
        print(f"BATCH MOTIF DETECTION: Testing Against {len(motif_filter)} Specified Motifs")
    else:
        print(f"BATCH MOTIF DETECTION: Testing Against ALL Motifs")
    print(f"{'='*80}")
    print(f"Protein: {header}")
    print(f"Length: {Lseq} aa")
    print(f"Layers: {args.layers}")
    if filter_mode:
        print(f"Motifs to test: {sorted(motif_filter)}")
        print(f"Window scores will be saved to: {args.window_scores_dir}")
    print()
    
    # Load motif names
    name_map = load_motif_names("motifs_list.json")
    if name_map:
        print(f"✓ Loaded {len(name_map)} motif names from motifs_list.json")
    
    # Find all available motifs
    print("Finding trained motifs...")
    try:
        all_motifs = find_all_motifs(args.tcav_dir, args.data_dir, name_map)
        print(f"✓ Found {len(all_motifs)} trained motifs")
        
        # Filter motifs if requested
        if filter_mode:
            motifs = [m for m in all_motifs if m["motif_id"] in motif_filter]
            print(f"✓ Filtered to {len(motifs)} requested motifs")
            
            # Check for missing motifs
            found_ids = {m["motif_id"] for m in motifs}
            missing = motif_filter - found_ids
            if missing:
                print(f"[WARN] Requested motifs not found: {sorted(missing)}")
        else:
            motifs = all_motifs
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    if not motifs:
        print("[ERROR] No trained motifs found!")
        return
    
    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = "cpu"
    
    # Load model
    print(f"\nLoading ESM model: {args.model_name}...")
    
    # Try to load from registry first
    try:
        from src.utils.model_loader import get_model_config, load_esm2_model
        
        # Try loading from registry (supports model names like 'ESMplusplus_large')
        registry_path = str(Path("esm2_tcav") / "models" / "model_registry.yaml")
        if os.path.exists(registry_path):
            try:
                model_config = get_model_config(args.model_name, registry_path)
                model, tokenizer, _ = load_esm2_model(args.model_name, registry_path, device=device)
                print(f"✓ Model loaded from registry: {args.model_name}")
                print(f"  Hidden size: {model_config.get('hidden_size', 'N/A')}")
                print(f"  Num layers: {model_config.get('num_layers', 'N/A')}")
            except KeyError:
                # Not in registry, try as path or HF model
                raise ValueError("Not in registry")
        else:
            raise ValueError("Registry not found")
            
    except (ImportError, ValueError, KeyError):
        # Fallback: load directly as path or HuggingFace model
        if os.path.exists(args.model_name):
            # Local model - check if it's a HuggingFace format
            if os.path.isdir(args.model_name) and os.path.exists(os.path.join(args.model_name, "config.json")):
                tokenizer = EsmTokenizer.from_pretrained(args.model_name, do_lower_case=False)
                model = EsmModel.from_pretrained(args.model_name)
                model.to(device)
            else:
                print(f"[ERROR] Path exists but not a valid HuggingFace model: {args.model_name}")
                return
        else:
            # HuggingFace hub model
            tokenizer = EsmTokenizer.from_pretrained(args.model_name, do_lower_case=False)
            model = EsmModel.from_pretrained(args.model_name)
            model.to(device)
        print(f"✓ Model loaded: {args.model_name}")
    
    model.eval()
    print("✓ Model ready")
    
    # Test each motif with its own window size and collect ALL hits
    print(f"\nTesting {len(motifs)} motifs (each with its own window size)...")
    all_hits = []  # Collect all individual window hits
    motif_summaries = []  # Summary per motif
    
    for i, motif_info in enumerate(motifs, 1):
        motif_id = motif_info["motif_id"]
        
        if (i-1) % 10 == 0 and i > 1:
            print(f"  Processed {i-1}/{len(motifs)} motifs...")
        
        try:
            result = detect_motif(
                seq=seq,
                motif_info=motif_info,
                model=model,
                tokenizer=tokenizer,
                layers=args.layers,
                device=device,
                version=args.version
            )
            
            if result and result.get("all_hits"):
                # Add all individual hits from this motif
                all_hits.extend(result["all_hits"])
                
                # Keep summary for this motif
                motif_summaries.append({
                    "motif_id": result["motif_id"],
                    "motif_name": result["motif_name"],
                    "n_hits": len(result["all_hits"]),
                    "best_zmean": max(h["zmean"] for h in result["all_hits"]),
                    "per_layer_max_score": result["per_layer_max_score"],
                    "per_layer_max_zscore": result["per_layer_max_zscore"]
                })
            
        except Exception as e:
            if args.verbose:
                print(f"  [WARN] Error testing {motif_id}: {e}")
            continue
    
    print(f"✓ Successfully tested {len(motif_summaries)} motifs")
    print(f"✓ Total hits detected: {len(all_hits)}")
    
    if not all_hits:
        print("\n[ERROR] No successful detections!")
        return
    
    # Determine ranking method
    if args.rank_by_layer is not None:
        rank_key = f"L{args.rank_by_layer}"
        if args.rank_by == "score":
            rank_metric = "Max Score"
            rank_label = f"{rank_metric} in Layer {args.rank_by_layer}"
            sort_key = lambda x: x["per_layer_scores"].get(rank_key, float('-inf'))
        else:  # zscore
            rank_metric = "Max Z-Score"
            rank_label = f"{rank_metric} in Layer {args.rank_by_layer}"
            sort_key = lambda x: x["per_layer_zscores"].get(rank_key, float('-inf'))
    else:
        rank_metric = "Z-Mean"
        rank_label = "Ensemble Z-Mean (across layers)"
        sort_key = lambda x: x["zmean"]
    
    # Sort ALL hits by chosen metric (highest first)
    all_hits.sort(key=sort_key, reverse=True)
    
    # Display top results (individual hits)
    print(f"\n{'='*110}")
    print(f"TOP {args.topk} HITS (Ranked by {rank_label})")
    print(f"{'='*110}")
    print(f"\n{'Rank':<6} {'Motif ID':<12} {'Domain Name':<35} {'Value':<10} {'Location':<12} {'Sequence'}")
    print("-" * 110)
    
    for rank, hit in enumerate(all_hits[:args.topk], 1):
        motif_id = hit["motif_id"]
        motif_name = hit.get("motif_name", "")[:33]  # Truncate if too long
        span = hit["span"]
        location = f"{span[0]:>4}-{span[1]:<4}"
        sequence = hit["sequence"][:40] + "..." if len(hit["sequence"]) > 40 else hit["sequence"]
        
        # Get the rank value based on ranking method
        if args.rank_by_layer is not None:
            rank_key = f"L{args.rank_by_layer}"
            if args.rank_by == "score":
                rank_value = hit["per_layer_scores"].get(rank_key, 0.0)
            else:
                rank_value = hit["per_layer_zscores"].get(rank_key, 0.0)
        else:
            rank_value = hit["zmean"]
        
        print(f"{rank:<6} {motif_id:<12} {motif_name:<35} {rank_value:>9.3f} {location:<12} {sequence}")
    
    # Detailed view of top hit
    if all_hits and args.show_detail:
        print(f"\n{'='*110}")
        print(f"DETAILED VIEW: Top Hit")
        print(f"{'='*110}")
        
        top_hit = all_hits[0]
        
        print(f"\nMotif ID: {top_hit['motif_id']}")
        print(f"Domain Name: {top_hit.get('motif_name', 'N/A')}")
        print(f"Ranking Metric: {rank_label}")
        
        if args.rank_by_layer is not None:
            rank_key = f"L{args.rank_by_layer}"
            if args.rank_by == "score":
                print(f"Rank Value (Layer {args.rank_by_layer} Score): {top_hit['per_layer_scores'].get(rank_key, 0.0):.3f}")
            else:
                print(f"Rank Value (Layer {args.rank_by_layer} Z-Score): {top_hit['per_layer_zscores'].get(rank_key, 0.0):.3f}")
        else:
            print(f"Rank Value (Ensemble Z-Mean): {top_hit['zmean']:.3f}")
        
        print(f"Location: {top_hit['span'][0]}-{top_hit['span'][1]}")
        print(f"Sequence: {top_hit['sequence']}")
        
        print(f"\nPer-layer scores for this hit:")
        for layer_name in sorted(top_hit["per_layer_scores"].keys()):
            score = top_hit['per_layer_scores'][layer_name]
            z = top_hit['per_layer_zscores'][layer_name]
            print(f"  {layer_name}: score={score:>7.3f} | z-score={z:>6.3f}")
        
        # Show other top hits from same motif
        same_motif_hits = [h for h in all_hits[:20] if h["motif_id"] == top_hit["motif_id"]]
        if len(same_motif_hits) > 1:
            print(f"\nOther top hits for {top_hit['motif_id']} in top 20:")
            for i, h in enumerate(same_motif_hits[1:], 2):
                print(f"  #{i}: pos {h['span'][0]:>4}-{h['span'][1]:<4} | z-mean={h['zmean']:>6.3f}")
    
    # Save results
    os.makedirs(args.outdir, exist_ok=True)
    
    # Save full results (all hits)
    out_json = os.path.join(args.outdir, f"batch_detection_{header.replace(' ', '_')}.json")
    with open(out_json, "w") as f:
        json.dump({
            "protein": header,
            "length": Lseq,
            "layers": args.layers,
            "ranking_method": rank_label,
            "total_motifs_tested": len(motif_summaries),
            "total_hits": len(all_hits),
            "motif_summaries": motif_summaries,
            "top_hits": all_hits[:args.topk * 2]  # Save top N*2 hits to file
        }, f, indent=2)
    print(f"\n✓ Full results saved to: {out_json}")
    
    # Save detailed window scores ONLY when specific motifs were requested
    if filter_mode:
        os.makedirs(args.window_scores_dir, exist_ok=True)
        print(f"\nSaving detailed window scores for {len(motif_summaries)} motifs...")
        
        # Group hits by motif
        hits_by_motif = {}
        for hit in all_hits:
            motif_id = hit["motif_id"]
            if motif_id not in hits_by_motif:
                hits_by_motif[motif_id] = []
            hits_by_motif[motif_id].append(hit)
        
        # Save one file per motif
        for motif_id, hits in hits_by_motif.items():
            # Get motif info
            motif_info = next((m for m in motif_summaries if m["motif_id"] == motif_id), None)
            
            window_data = {
                "motif_id": motif_id,
                "motif_name": motif_info["motif_name"] if motif_info else "",
                "protein": header,
                "protein_length": Lseq,
                "window_length": hits[0]["window_length"] if hits else 0,
                "n_windows": len(hits),
                "windows": hits
            }
            
            window_file = os.path.join(args.window_scores_dir, f"{motif_id}_windows.json")
            with open(window_file, "w") as f:
                json.dump(window_data, f, indent=2)
        
        print(f"✓ Window scores saved to: {args.window_scores_dir}")
        print(f"  - {len(hits_by_motif)} motif files created")
    
    # Save top-k summary
    summary_file = os.path.join(args.outdir, f"top_{args.topk}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Top {args.topk} Hit Detections for: {header}\n")
        f.write(f"Protein length: {Lseq} aa\n")
        f.write(f"Total motifs tested: {len(motif_summaries)}\n")
        f.write(f"Total hits: {len(all_hits)}\n")
        f.write(f"Ranking method: {rank_label}\n")
        f.write(f"{'='*110}\n\n")
        f.write(f"{'Rank':<6} {'Motif ID':<12} {'Domain Name':<35} {'Value':<12} {'Location':<12} {'Sequence'}\n")
        f.write("-" * 110 + "\n")
        
        for rank, hit in enumerate(all_hits[:args.topk], 1):
            span = hit["span"]
            location = f"{span[0]}-{span[1]}"
            motif_name = hit.get("motif_name", "")[:33]
            sequence = hit["sequence"][:40] + "..." if len(hit["sequence"]) > 40 else hit["sequence"]
            
            # Get rank value based on method
            if args.rank_by_layer is not None:
                rank_key = f"L{args.rank_by_layer}"
                if args.rank_by == "score":
                    rank_value = hit["per_layer_scores"].get(rank_key, 0.0)
                else:
                    rank_value = hit["per_layer_zscores"].get(rank_key, 0.0)
            else:
                rank_value = hit["zmean"]
            
            f.write(f"{rank:<6} {hit['motif_id']:<12} {motif_name:<35} {rank_value:>11.3f} {location:<12} {sequence}\n")
        
        if args.show_detail and all_hits:
            f.write(f"\n\nDetailed View of Top Hit:\n")
            f.write(f"{'='*110}\n")
            top_hit = all_hits[0]
            f.write(f"Motif ID: {top_hit['motif_id']}\n")
            f.write(f"Domain Name: {top_hit.get('motif_name', 'N/A')}\n")
            f.write(f"Z-Mean: {top_hit['zmean']:.3f}\n")
            f.write(f"Location: {top_hit['span'][0]}-{top_hit['span'][1]}\n")
            f.write(f"Sequence: {top_hit['sequence']}\n")
    
    print(f"✓ Top-{args.topk} summary saved to: {summary_file}")
    
    print("\n✓ Batch detection complete!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Test a protein against all trained motif CAVs and rank by top z-mean score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test ALL motifs (default behavior)
  python batch_detect_all_motifs.py --fasta my_protein.fasta
  
  # Test specific motifs and save ALL window scores
  python batch_detect_all_motifs.py --fasta my_protein.fasta --motifs PF00096 PF00641 PF07679
  
  # Test motifs from file and save window scores to custom directory
  python batch_detect_all_motifs.py --fasta my_protein.fasta --motifs-file my_motifs.txt --window-scores-dir my_scores/
  
  # Rank by max score in layer 4
  python batch_detect_all_motifs.py --fasta my_protein.fasta --rank-by-layer 4 --rank-by score
  
  # Show top 20 with detailed view
  python batch_detect_all_motifs.py --fasta test.fasta --topk 20 --show-detail
  
  # Use GPU
  python batch_detect_all_motifs.py --fasta test.fasta --device cuda --layers 2 4 6
        """
    )
    
    # Input
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", type=str, help="Raw amino acid sequence")
    src.add_argument("--fasta", type=str, help="FASTA file path")
    
    # Paths
    ap.add_argument("--tcav-dir", type=str, default=DEFAULT_TCAV_DIR,
                   help=f"TCAV outputs directory (default: {DEFAULT_TCAV_DIR})")
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                   help=f"Data directory with metadata (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HuggingFace ESM2 model name (default: {DEFAULT_MODEL_NAME})")
    
    # Motif filtering
    motif_group = ap.add_mutually_exclusive_group()
    motif_group.add_argument("--motifs", type=str, nargs="+",
                            help="Specific motif IDs to test (e.g., PF00001 PF00002). When provided, saves ALL window scores.")
    motif_group.add_argument("--motifs-file", type=str,
                            help="File containing motif IDs to test (one per line). When provided, saves ALL window scores.")
    
    # Parameters
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                   help=f"Layers to use (default: {DEFAULT_LAYERS})")
    ap.add_argument("--topk", type=int, default=20,
                   help="Number of top hits to display (default: 20)")
    ap.add_argument("--version", type=str, default="v1",
                   help="CAV artifact version (default: v1)")
    
    # Ranking options
    ap.add_argument("--rank-by-layer", type=int, default=None,
                   help="Rank by max score/z-score in specific layer (e.g., 2, 4, 6). If not set, ranks by ensemble z-mean")
    ap.add_argument("--rank-by", type=str, choices=["score", "zscore"], default="score",
                   help="When using --rank-by-layer, rank by 'score' or 'zscore' (default: score)")
    
    # Output
    ap.add_argument("--header", type=str, help="Optional header for --seq input")
    ap.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                   help="Device (default: cpu)")
    ap.add_argument("--outdir", type=str, default="batch_detection_outputs",
                   help="Output directory (default: batch_detection_outputs)")
    ap.add_argument("--window-scores-dir", type=str, default="window_scores",
                   help="Directory to save detailed window scores when using --motifs (default: window_scores)")
    ap.add_argument("--show-detail", action="store_true",
                   help="Show detailed view of top match")
    ap.add_argument("--verbose", action="store_true",
                   help="Show warnings for individual motif failures")
    
    args = ap.parse_args()
    main(args)


