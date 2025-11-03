#!/usr/bin/env python3
"""
Single protein CAV detection script.
Takes a protein sequence and runs CAV-based zinc finger detection to show predicted domain locations.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from external_eval import (
    load_esm_from_config, tokenize_batch, load_layer_artifacts, 
    standardize, layer_scores, fits_esm_limit
)

def parse_fasta_header(header: str) -> Tuple[str, str]:
    """Parse FASTA header to extract accession and description."""
    parts = header.split('|')
    if len(parts) >= 2:
        accession = parts[1]
        description = ' '.join(parts[2:]) if len(parts) > 2 else ""
    else:
        accession = header.split()[0] if header else "unknown"
        description = header
    return accession, description

def sliding_window_scan(sequence: str, window_len: int, stride: int = 1) -> List[Tuple[int, int]]:
    """Generate sliding window positions for the sequence."""
    windows = []
    seq_len = len(sequence)
    
    for start in range(0, seq_len - window_len + 1, stride):
        end = start + window_len
        if fits_esm_limit(end):
            windows.append((start, end))
    
    return windows

def embed_sequence_windows(cfg: Dict[str, Any], sequence: str, windows: List[Tuple[int, int]], 
                          layers: List[int]) -> Dict[int, np.ndarray]:
    """Embed sequence windows for all specified layers."""
    loader = load_esm_from_config(cfg)
    model = loader["model"]
    device = loader["device"]
    
    # Initialize results with correct dimensions
    num_windows = len(windows)
    results = {L: [] for L in layers}
    batch_size = cfg.get("embedding", {}).get("batch_size", 8)
    bos_shift = 1
    
    # Process windows in batches
    for i in tqdm(range(0, num_windows, batch_size), desc="Embedding windows"):
        batch_windows = windows[i:i+batch_size]
        batch_seqs = [sequence[start:end] for start, end in batch_windows]
        
        # Tokenize batch
        toks = tokenize_batch(loader, batch_seqs)
        
        # Get embeddings
        if loader["flavor"] == "hf":
            out = model(**toks, output_hidden_states=True)
            hidden_states = out.hidden_states
        else:
            out = model(toks["tokens"], repr_layers=set(layers), return_contacts=False)
            reps = out["representations"]
        
        # Extract window embeddings - ensure we process all windows in this batch
        for bi in range(len(batch_windows)):
            start, end = batch_windows[bi]
            
            for L in layers:
                if loader["flavor"] == "hf":
                    H = hidden_states[L][bi]  # [T, D]
                else:
                    H = reps[L][bi]  # [T, D]
                
                Tlen = H.shape[0]
                s_tok = start + bos_shift
                e_tok = end + bos_shift
                
                s = min(max(0, s_tok), Tlen)
                e = min(max(s, e_tok), Tlen)
                
                if e <= s:
                    # Invalid window - use zero vector
                    if results[L]:
                        results[L].append(np.zeros_like(results[L][0]))
                    else:
                        # First window - determine dimension from model
                        if isinstance(H, torch.Tensor):
                            dim = H.shape[1]
                        else:
                            dim = H.shape[1]
                        results[L].append(np.zeros(dim))
                    continue
                
                if isinstance(H, torch.Tensor):
                    window = H[s:e].float()
                    v = window.mean(dim=0).detach().cpu().numpy().astype(np.float32)
                else:
                    v = H[s:e].astype(np.float32).mean(axis=0)
                
                if np.isfinite(v).all():
                    results[L].append(v)
                else:
                    # Invalid embedding - use zero vector
                    if results[L]:
                        results[L].append(np.zeros_like(results[L][0]))
                    else:
                        dim = v.shape[0] if hasattr(v, 'shape') else 1280
                        results[L].append(np.zeros(dim))
    
    # Verify we have the right number of embeddings
    for L in layers:
        if len(results[L]) != num_windows:
            print(f"Warning: Layer {L} has {len(results[L])} embeddings but {num_windows} windows")
            # Pad with zeros if needed
            while len(results[L]) < num_windows:
                if results[L]:
                    results[L].append(np.zeros_like(results[L][0]))
                else:
                    results[L].append(np.zeros(1280))
    
    # Convert to numpy arrays
    for L in layers:
        if results[L]:
            results[L] = np.vstack(results[L])
        else:
            results[L] = np.array([])
    
    return results

def detect_domains(sequence: str, cfg: Dict[str, Any], model_name: str, 
                  layers: List[int], window_len: int = 41, stride: int = 5,
                  threshold: float = None) -> Dict[str, Any]:
    """Run CAV-based domain detection on a single protein sequence."""
    
    # Generate sliding windows
    windows = sliding_window_scan(sequence, window_len, stride)
    print(f"Generated {len(windows)} windows for sequence of length {len(sequence)}")
    
    if len(windows) == 0:
        return {"error": "No valid windows generated"}
    
    # Embed windows
    embeddings = embed_sequence_windows(cfg, sequence, windows, layers)
    
    # Debug: check embedding dimensions
    for L in layers:
        if L in embeddings:
            print(f"Layer {L}: {embeddings[L].shape[0]} embeddings, expected {len(windows)}")
        else:
            print(f"Layer {L}: No embeddings found")
    
    # Load CAV artifacts
    cav_dir = os.path.join("outputs", "cavs", model_name)
    results = {
        "sequence_length": len(sequence),
        "num_windows": len(windows),
        "window_length": window_len,
        "stride": stride,
        "layers": {}
    }
    
    for L in layers:
        print(f"Processing layer {L}...")
        
        # Load CAV and scaler
        try:
            cav, scaler, _ = load_layer_artifacts(model_name, L, cav_dir)
        except Exception as e:
            print(f"Error loading artifacts for layer {L}: {e}")
            continue
        
        # Get embeddings for this layer
        if L not in embeddings or len(embeddings[L]) == 0:
            print(f"No embeddings for layer {L}")
            continue
        
        X = embeddings[L]
        
        # Standardize and score
        Xz = standardize(X, scaler)
        scores = layer_scores(Xz, cav)
        
        # Apply threshold if provided
        if threshold is not None:
            predicted_domains = scores >= threshold
        else:
            # Use top 10% of scores as predicted domains
            threshold = np.percentile(scores, 90)
            predicted_domains = scores >= threshold
        
        # Collect domain predictions
        domain_windows = []
        for i, (start, end) in enumerate(windows):
            if predicted_domains[i]:
                domain_windows.append({
                    "start": start,
                    "end": end,
                    "score": float(scores[i]),
                    "sequence": sequence[start:end]
                })
        
        # Sort by score (highest first)
        domain_windows.sort(key=lambda x: x["score"], reverse=True)
        
        results["layers"][f"L{L}"] = {
            "threshold": float(threshold) if threshold is not None else float(np.percentile(scores, 90)),
            "num_predictions": len(domain_windows),
            "domains": domain_windows,
            "all_scores": scores.tolist(),
            "window_positions": windows
        }
    
    return results

def format_output(results: Dict[str, Any], accession: str, description: str) -> str:
    """Format detection results for display."""
    output = []
    output.append(f"Protein: {accession}")
    output.append(f"Description: {description}")
    output.append(f"Sequence length: {results['sequence_length']} amino acids")
    output.append(f"Windows analyzed: {results['num_windows']}")
    output.append("=" * 80)
    
    for layer_name, layer_data in results["layers"].items():
        output.append(f"\n{layer_name} Results:")
        output.append(f"Threshold: {layer_data['threshold']:.4f}")
        output.append(f"Predicted domains: {layer_data['num_predictions']}")
        
        if layer_data['domains']:
            output.append("\nTop predicted domains:")
            for i, domain in enumerate(layer_data['domains'][:10]):  # Show top 10
                output.append(f"  {i+1}. Positions {domain['start']+1}-{domain['end']} "
                           f"(score: {domain['score']:.4f})")
                output.append(f"     Sequence: {domain['sequence']}")
        else:
            output.append("  No domains predicted above threshold")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Run CAV detection on a single protein")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 4, 6], help="Layers to analyze")
    parser.add_argument("--window_len", type=int, default=41, help="Window length")
    parser.add_argument("--stride", type=int, default=5, help="Sliding window stride")
    parser.add_argument("--threshold", type=float, default=None, help="Score threshold for predictions")
    parser.add_argument("--output", default=None, help="Output file (default: print to stdout)")
    parser.add_argument("--sequence", default=None, help="Custom protein sequence (optional)")
    parser.add_argument("--header", default=None, help="Custom protein header (optional)")
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    if args.model:
        cfg["model"]["name"] = args.model
    
    model_name = cfg["model"]["name"]
    
    # Use custom sequence if provided, otherwise use default
    if args.sequence:
        sequence = args.sequence
        header = args.header or "Custom protein sequence"
    else:
        # Default protein data
        header = "tr|F4IRZ1|F4IRZ1_ARATH LOW protein: UPF0503-like protein, putative (DUF740) OS=Arabidopsis thaliana OX=3702 GN=OPL2 PE=1 SV=1"
        sequence = ("MVMNNPANNNPVAASSASAVALAPPPHPPQPHRPSTSCDRHPDERFTGFCPSCLFDRLSV"
                    "LDITGKNNNAVASSSKKPPSSSAALKAIFKPSSSSGSFFPELRRTKSFSASKAEAFSLGA"
                    "FEPQRRSCDVRVRNTLWSLFHEDAEHNSQTKEGLSVNCSEIDLERINSIVKSPVFEEETE"
                    "IESEQDNEKDIKFETFKEPRSVIDEIVEEEEEEETKKVEDFTMEFNPQTTAKKTNRDFKE"
                    "IAGSFWSAASVFSKKLQKWRQKQKLKKHRTGNLGAGSSALPVEKAIGRQLRDTQSEIAEY"
                    "GYGRRSCDTDPRFSIDAGRFSLDAGRVSVDDPRYSFEEPRASWDGYLIGRAAAPMRMPSM"
                    "LSVVEDSPVRNHVHRSDTHIPVEKSPQVSEAVIDEIVPGGSAQTREYYLDSSSSRRRKSL"
                    "DRSSSTRKLSASVMAEIDELKLTQDREAKDLVSHSNSLRDDCCSVENNYEMGVRENVGTI"
                    "ECNKKRTKKSRWSWNIFGLLHRKNGNKYEEEERRSGVDRTFSGSWNVEPRNGFDPKMIRS"
                    "NSSVSWRSSGTTGGGLQRNSVDGYISGKKKVSKAENGMLKFYLTPGKGRRRGSGNSTAPT"
                    "SRPVPASQPFGSRNVMNFY")
    
    accession, description = parse_fasta_header(header)
    
    print(f"Running CAV detection on protein: {accession}")
    print(f"Model: {model_name}")
    print(f"Layers: {args.layers}")
    
    # Run detection
    results = detect_domains(
        sequence, cfg, model_name, args.layers, 
        args.window_len, args.stride, args.threshold
    )
    
    # Format and output results
    output_text = format_output(results, accession, description)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Results saved to {args.output}")
    else:
        print("\n" + output_text)
    
    # Also save detailed JSON results
    json_output = args.output.replace(".txt", ".json") if args.output else f"{accession}_detection_results.json"
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {json_output}")

if __name__ == "__main__":
    main()
