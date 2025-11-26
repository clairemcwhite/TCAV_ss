#!/usr/bin/env python3
"""
Run motif detection on test proteins using on-the-fly window embedding.

Similar to batch_detect_all_motifs.py but processes multiple test proteins.
No pre-computed embeddings needed - embeds windows on-the-fly.
Processes proteins sequentially to avoid race conditions.

Usage:
    python run_test_detection_onthefly.py --test-data-dir ./test_data --tcav-dir ./tcav_outputs_650m --model-name esm2_t33_650M_UR50D --layers 22
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
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add esm2_tcav to path for model loading
sys.path.insert(0, str(Path(__file__).parent / "esm2_tcav"))

# ----------------------------
# Config
# ----------------------------
DEFAULT_TCAV_DIR = "./tcav_outputs"
DEFAULT_DATA_DIR = "./tcav_data"
DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEFAULT_LAYERS = [2, 4, 6]

# ----------------------------
# Helper Functions (from batch_detect)
# ----------------------------

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


def find_all_motifs(tcav_dir: str, data_dir: str) -> List[Dict[str, Any]]:
    """
    Find all trained motifs in tcav_outputs/cavs/.
    Returns list of dicts with motif_id, cav_dir, window_length.
    """
    cav_base = Path(tcav_dir) / "cavs"
    
    if not cav_base.exists():
        raise FileNotFoundError(f"CAV directory not found: {cav_base}")
    
    motifs = []
    
    for motif_dir in sorted(cav_base.iterdir()):
        if not motif_dir.is_dir() or not motif_dir.name.startswith("PF"):
            continue
        
        motif_id = motif_dir.name
        
        # Get window length from metadata
        window_length = 40  # default
        metadata_file = Path(data_dir) / motif_id / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    window_length = metadata.get("window_length", 40)
            except:
                pass
        
        motifs.append({
            "motif_id": motif_id,
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


def detect_motif_onthefly(
    seq: str,
    motif_info: Dict[str, Any],
    model,
    tokenizer,
    layers: List[int],
    device: str,
    rank_by_layer: int = None,
    rank_by: str = "score",
    version: str = "v1"
) -> Dict[str, Any]:
    """
    Detect a single motif in the sequence using on-the-fly embedding.
    
    Returns dict with all window hits and their scores.
    """
    motif_id = motif_info["motif_id"]
    cav_dir = motif_info["cav_dir"]
    window_len = motif_info["window_length"]
    stride = window_len // 2
    
    Lseq = len(seq)
    
    # Generate windows
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
    
    # Return all hits with their scores
    all_hits = []
    for i, (start, end) in enumerate(spans):
        hit = {
            "motif_id": motif_id,
            "window_length": window_len,
            "span": [start, end],
            "per_layer_scores": {f"L{L}": float(per_layer[L]["scores"][i]) for L in per_layer.keys()},
            "per_layer_zscores": {f"L{L}": float(per_layer[L]["zscores"][i]) for L in per_layer.keys()}
        }
        all_hits.append(hit)
    
    return all_hits


def load_test_proteins(test_data_dir: Path, trained_motif_ids: set = None) -> Dict[str, Dict[str, Any]]:
    """
    Load all test proteins from test_data directory.
    
    Args:
        test_data_dir: Path to test data directory
        trained_motif_ids: Set of motif IDs that were actually trained (for filtering ground truth)
    
    Returns dict mapping accession -> {sequence, length, ground_truth}
    
    Note: Ground truth is filtered to only include motifs that were trained.
    This ensures fair evaluation - we only test detection of motifs we trained for.
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
                    all_ground_truth = data.get("ground_truth_annotations", [])
                    
                    # Filter ground truth to only include trained motifs
                    if trained_motif_ids is not None:
                        filtered_ground_truth = [
                            gt for gt in all_ground_truth
                            if gt["motif_id"] in trained_motif_ids
                        ]
                    else:
                        filtered_ground_truth = all_ground_truth
                    
                    proteins[accession] = {
                        "sequence": data.get("sequence", ""),
                        "length": data.get("length", len(data.get("sequence", ""))),
                        "ground_truth": filtered_ground_truth
                    }
    
    return proteins


def run_detection_on_protein(
    accession: str,
    sequence: str,
    trained_motifs: List[Dict[str, Any]],
    model,
    tokenizer,
    layers: List[int],
    device: str,
    rank_by_layer: int = None,
    rank_by: str = "score",
    topk: int = 20
) -> Dict[str, Any]:
    """
    Run detection for all motifs on a single protein.
    """
    all_hits = []
    
    # Detect each motif
    for motif_info in trained_motifs:
        try:
            hits = detect_motif_onthefly(
                seq=sequence,
                motif_info=motif_info,
                model=model,
                tokenizer=tokenizer,
                layers=layers,
                device=device,
                rank_by_layer=rank_by_layer,
                rank_by=rank_by
            )
            
            if hits:
                all_hits.extend(hits)
        except Exception as e:
            # Skip motifs that fail
            continue
    
    # Determine ranking method
    if rank_by == "zmean":
        # Compute ensemble z-mean for each hit
        for hit in all_hits:
            zscores = [hit["per_layer_zscores"][f"L{L}"] for L in layers if f"L{L}" in hit["per_layer_zscores"]]
            hit["ranking_score"] = float(np.mean(zscores)) if zscores else 0.0
    else:
        # Use layer score or zscore
        if rank_by_layer is None:
            rank_by_layer = layers[0]
        
        rank_key = f"L{rank_by_layer}"
        
        for hit in all_hits:
            if rank_by == "score":
                hit["ranking_score"] = hit["per_layer_scores"].get(rank_key, float('-inf'))
            else:  # zscore
                hit["ranking_score"] = hit["per_layer_zscores"].get(rank_key, float('-inf'))
    
    # Sort by ranking score (descending)
    all_hits.sort(key=lambda x: x["ranking_score"], reverse=True)
    
    # Keep top-k
    top_hits = all_hits[:topk]
    
    return {
        "accession": accession,
        "total_hits": len(all_hits),
        "top_k": topk,
        "rank_by": rank_by,
        "rank_by_layer": rank_by_layer if rank_by != "zmean" else None,
        "predictions": top_hits
    }


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run motif detection on test proteins using on-the-fly window embedding"
    )
    
    parser.add_argument("--test-data-dir", default="./test_data",
                       help="Directory with test data")
    parser.add_argument("--tcav-dir", default=DEFAULT_TCAV_DIR,
                       help="TCAV outputs directory")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                       help="Data directory with metadata")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                       help="ESM2 model name or path")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                       help="Layers to use")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                       help="Device to use")
    parser.add_argument("--output-file", default="./test_detection_results.json",
                       help="Output JSON file")
    parser.add_argument("--topk", type=int, default=20,
                       help="Number of top predictions per protein")
    
    # Ranking options
    parser.add_argument("--rank-by-layer", type=int, default=None,
                       help="Layer to use for ranking (e.g., 22). If not set, uses first layer in --layers")
    parser.add_argument("--rank-by", type=str, choices=["score", "zscore", "zmean"], default="score",
                       help="Ranking method: 'score' (raw TCAV score, default), 'zscore' (layer z-score), or 'zmean' (ensemble)")
    
    parser.add_argument("--version", default="v1",
                       help="CAV version")
    
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision (fp16) for faster inference")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST DETECTION (ON-THE-FLY EMBEDDING)")
    print("=" * 80)
    
    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = "cpu"
    
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Layers: {args.layers}")
    
    # Find trained motifs first (needed for ground truth filtering)
    print("\n[1/5] Loading trained motifs...")
    trained_motifs = find_all_motifs(args.tcav_dir, args.data_dir)
    print(f"✓ Found {len(trained_motifs)} trained motifs")
    
    if len(trained_motifs) == 0:
        print("[ERROR] No trained motifs found!")
        return
    
    # Create set of trained motif IDs for ground truth filtering
    trained_motif_ids = {motif["motif_id"] for motif in trained_motifs}
    
    # Load test proteins (with ground truth filtered to trained motifs only)
    print("\n[2/5] Loading test proteins...")
    test_data_dir = Path(args.test_data_dir)
    test_proteins = load_test_proteins(test_data_dir, trained_motif_ids)
    print(f"✓ Found {len(test_proteins)} unique test proteins")
    print(f"  (Ground truth filtered to {len(trained_motif_ids)} trained motifs)")
    
    if len(test_proteins) == 0:
        print("[ERROR] No test proteins found!")
        print("Run collect_test_data.py first.")
        return
    
    # Load model
    print(f"\n[3/5] Loading ESM2 model...")
    
    try:
        # Try loading from registry
        from src.utils.model_loader import load_esm2_model, get_model_config
        
        registry_path = str(Path("esm2_tcav") / "models" / "model_registry.yaml")
        if os.path.exists(registry_path):
            try:
                model_config = get_model_config(args.model_name, registry_path)
                model, tokenizer, _ = load_esm2_model(args.model_name, registry_path, device=device)
                print(f"✓ Loaded from registry: {args.model_name}")
            except (KeyError, ValueError):
                raise ValueError("Not in registry")
        else:
            raise ValueError("Registry not found")
    except (ImportError, ValueError):
        # Fallback: HuggingFace
        tokenizer = EsmTokenizer.from_pretrained(args.model_name, do_lower_case=False)
        model = EsmModel.from_pretrained(args.model_name)
        model.to(device)
        print(f"✓ Loaded from HuggingFace: {args.model_name}")
    
    model.eval()
    
    # Enable mixed precision if requested
    if args.fp16 and device == "cuda":
        print("✓ Using mixed precision (fp16)")
        model = model.half()
    
    # Run detection on all proteins
    print(f"\n[4/5] Running detection on {len(test_proteins)} proteins...")
    
    # Determine ranking layer
    rank_by_layer = args.rank_by_layer if args.rank_by_layer is not None else args.layers[0]
    
    if args.rank_by == "score":
        rank_label = f"Layer {rank_by_layer} Score"
    elif args.rank_by == "zscore":
        rank_label = f"Layer {rank_by_layer} Z-Score"
    else:
        rank_label = "Ensemble Z-Mean"
    
    print(f"  Ranking by: {rank_label}")
    print(f"  Testing against: {len(trained_motifs)} motifs per protein")
    
    results = []
    
    # Process proteins sequentially (thread-safe, no race conditions)
    for accession, protein_data in tqdm(test_proteins.items(), desc="Detecting"):
        try:
            detection_result = run_detection_on_protein(
                accession=accession,
                sequence=protein_data["sequence"],
                trained_motifs=trained_motifs,
                model=model,
                tokenizer=tokenizer,
                layers=args.layers,
                device=device,
                rank_by_layer=args.rank_by_layer,
                rank_by=args.rank_by,
                topk=args.topk
            )
            
            # Add ground truth
            detection_result["ground_truth"] = protein_data["ground_truth"]
            detection_result["length"] = protein_data["length"]
            
            results.append(detection_result)
        except Exception as e:
            print(f"\n[WARN] Error processing {accession}: {e}")
    
    # Save results
    print(f"\n[5/5] Saving results...")
    
    # Calculate ground truth statistics
    total_gt_annotations = sum(len(r["ground_truth"]) for r in results)
    proteins_with_gt = sum(1 for r in results if len(r["ground_truth"]) > 0)
    
    output_data = {
        "n_proteins": len(results),
        "n_motifs": len(trained_motifs),
        "trained_motif_ids": sorted(list(trained_motif_ids)),
        "topk": args.topk,
        "layers": args.layers,
        "rank_by": args.rank_by,
        "rank_by_layer": rank_by_layer if args.rank_by != "zmean" else None,
        "model": args.model_name,
        "ground_truth_stats": {
            "total_annotations": total_gt_annotations,
            "proteins_with_ground_truth": proteins_with_gt,
            "note": "Ground truth filtered to only include trained motifs"
        },
        "results": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to {args.output_file}")
    
    print("\n" + "=" * 80)
    print("DETECTION COMPLETE")
    print("=" * 80)
    print(f"Proteins processed: {len(results)}")
    print(f"Motifs tested per protein: {len(trained_motifs)}")
    print(f"Top-k per protein: {args.topk}")
    print(f"Ranking method: {rank_label}")
    print(f"\nGround Truth Statistics:")
    print(f"  Total annotations: {total_gt_annotations}")
    print(f"  Proteins with ground truth: {proteins_with_gt}/{len(results)}")
    print(f"  (Filtered to {len(trained_motif_ids)} trained motifs only)")
    print(f"\nNext step: python evaluate_metrics.py --detection-results {args.output_file}")


if __name__ == "__main__":
    main()




