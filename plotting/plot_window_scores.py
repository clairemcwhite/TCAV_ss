#!/usr/bin/env python3
"""
Plot position-wise aggregated window scores for motif detection.

This script:
1. Reads window score JSON files from a directory
2. Aggregates overlapping window scores by position (using MAX)
3. Plots all motifs on the same plot for comparison

Usage:
  python plot_window_scores.py --window-scores-dir window_scores/ --layer 22
  python plot_window_scores.py --window-scores-dir window_scores/ --layer 22 --output my_plot.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d


def load_window_scores(window_scores_dir, layer_key, aggregation='max'):
    """
    Load all window score files from directory.
    Returns dict: motif_id -> {positions, scores, motif_name, etc.}
    
    Args:
        window_scores_dir: Directory containing window score JSON files
        layer_key: Layer key (e.g., "L22")
        aggregation: Aggregation method - 'max' or 'mean'
    """
    window_dir = Path(window_scores_dir)
    
    if not window_dir.exists():
        raise FileNotFoundError(f"Window scores directory not found: {window_scores_dir}")
    
    motif_data = {}
    
    for json_file in sorted(window_dir.glob("*_windows.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        motif_id = data["motif_id"]
        motif_name = data.get("motif_name", "")
        protein_length = data["protein_length"]
        windows = data["windows"]
        
        # Aggregate scores by position
        scores_per_position = defaultdict(list)
        
        for window in windows:
            start, end = window["span"]
            
            # Get the score for the specified layer
            score = window["per_layer_scores"].get(layer_key)
            
            if score is None:
                continue
            
            # Add score to all positions covered by this window
            for pos in range(start, end):
                scores_per_position[pos].append(score)
        
        # Aggregate using specified method
        positions = list(range(protein_length))
        aggregated_scores = []
        
        for pos in positions:
            if pos in scores_per_position:
                if aggregation == 'max':
                    aggregated_scores.append(max(scores_per_position[pos]))
                elif aggregation == 'mean':
                    aggregated_scores.append(np.mean(scores_per_position[pos]))
                else:
                    aggregated_scores.append(max(scores_per_position[pos]))
            else:
                aggregated_scores.append(np.nan)  # No coverage at this position
        
        motif_data[motif_id] = {
            "motif_name": motif_name,
            "positions": np.array(positions) + 1,  # 1-indexed for display
            "scores": np.array(aggregated_scores),
            "protein_length": protein_length,
            "n_windows": len(windows)
        }
    
    return motif_data


def plot_motif_scores(motif_data, layer_key, aggregation='max', sigma=0, output_file=None):
    """
    Plot position-wise scores for all motifs on the same plot.
    
    Args:
        motif_data: Dict of motif data
        layer_key: Layer identifier (e.g., "L22")
        aggregation: Aggregation method used ('max' or 'mean')
        sigma: Gaussian smoothing sigma (0 = no smoothing)
        output_file: Optional output file path
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(motif_data)))
    
    for i, (motif_id, data) in enumerate(sorted(motif_data.items())):
        positions = data["positions"]
        scores = data["scores"]
        motif_name = data["motif_name"]
        
        # Apply smoothing if requested
        if sigma > 0:
            # Handle NaN values for smoothing
            valid_mask = ~np.isnan(scores)
            if valid_mask.any():
                smoothed_scores = scores.copy()
                # Only smooth non-NaN regions
                smoothed_scores[valid_mask] = gaussian_filter1d(
                    scores[valid_mask], 
                    sigma=sigma, 
                    mode='nearest'
                )
                scores = smoothed_scores
        
        # Create label
        if motif_name:
            label = f"{motif_id} ({motif_name})"
        else:
            label = motif_id
        
        # Plot line
        ax.plot(positions, scores, label=label, color=colors[i], linewidth=2, alpha=0.8)
    
    # Formatting
    ax.set_xlabel("Sequence Position", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"Score ({layer_key})", fontsize=12, fontweight='bold')
    
    # Build title
    agg_text = aggregation.upper()
    smooth_text = f", σ={sigma}" if sigma > 0 else ""
    ax.set_title(f"Motif Detection Scores by Position\n(Layer {layer_key.replace('L', '')} scores, {agg_text} aggregation{smooth_text})", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add horizontal line at y=0 if scores cross zero
    if any(np.nanmin(data["scores"]) < 0 for data in motif_data.values()):
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_motif_scores_alternative(motif_data, layer_key, aggregation='max', sigma=0, output_file=None):
    """
    Plot position-wise scores with alternative style:
    - High contrast colors
    - No grid
    - No legend
    - 0th and last index forced to 0
    
    Args:
        motif_data: Dict of motif data
        layer_key: Layer identifier (e.g., "L22")
        aggregation: Aggregation method used ('max' or 'mean')
        sigma: Gaussian smoothing sigma (0 = no smoothing)
        output_file: Optional output file path
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use high contrast color scheme
    contrasting_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', 
                         '#FFFF00', '#00FFFF', '#FF8000', '#8000FF',
                         '#FF0080', '#80FF00', '#0080FF', '#FF8080']
    
    for i, (motif_id, data) in enumerate(sorted(motif_data.items())):
        positions = data["positions"]
        scores = data["scores"].copy()  # Make a copy to modify
        motif_name = data["motif_name"]
        

        
        # Apply smoothing if requested
        if sigma > 0:
            # Handle NaN values for smoothing
            valid_mask = ~np.isnan(scores)
            if valid_mask.any():
                smoothed_scores = scores.copy()
                # Only smooth non-NaN regions
                smoothed_scores[valid_mask] = gaussian_filter1d(
                    scores[valid_mask], 
                    sigma=sigma, 
                    mode='nearest'
                )
                scores = smoothed_scores
        
        # Use contrasting colors (cycle if more motifs than colors)
        color = contrasting_colors[i % len(contrasting_colors)]
        
        # Plot line (no label for legend)
        ax.plot(positions, scores, color=color, linewidth=2.5, alpha=0.9)
    
    # Formatting
    ax.set_xlabel("Sequence Position", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"Score ({layer_key})", fontsize=12, fontweight='bold')
    
    # Build title
    agg_text = aggregation.upper()
    smooth_text = f", σ={sigma}" if sigma > 0 else ""
    ax.set_title(f"Motif Detection Scores by Position\n(Layer {layer_key.replace('L', '')} scores, {agg_text} aggregation{smooth_text})", 
                 fontsize=14, fontweight='bold')
    
    # No legend
    # No grid
    ax.grid(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Alternative plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main(args):
    layer_key = f"L{args.layer}"
    
    print(f"\n{'='*80}")
    print(f"PLOTTING MOTIF WINDOW SCORES")
    print(f"{'='*80}")
    print(f"Window scores directory: {args.window_scores_dir}")
    print(f"Layer: {args.layer} ({layer_key})")
    print(f"Aggregation: {args.aggregation.upper()}")
    print(f"Smoothing (σ): {args.sigma}")
    print(f"Plot style: {args.style}")
    print(f"Output format: {args.format.upper()}")
    print()
    
    # Load window scores
    print("Loading window score files...")
    try:
        motif_data = load_window_scores(args.window_scores_dir, layer_key, args.aggregation)
        print(f"✓ Loaded {len(motif_data)} motifs")
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    if not motif_data:
        print("[ERROR] No motif data found!")
        return
    
    # Print summary
    print("\nMotifs loaded:")
    for motif_id, data in sorted(motif_data.items()):
        motif_name = data["motif_name"] or "N/A"
        n_windows = data["n_windows"]
        max_score = np.nanmax(data["scores"])
        print(f"  - {motif_id}: {motif_name[:40]} | {n_windows} windows | max score: {max_score:.3f}")
    
    # Handle output file with format
    output_file = args.output
    if output_file:
        # Ensure the output file has the correct extension
        from pathlib import Path
        output_path = Path(output_file)
        if output_path.suffix.lower() != f'.{args.format}':
            output_file = str(output_path.with_suffix(f'.{args.format}'))
            print(f"Note: Changed output extension to match format: {output_file}")
    
    # Plot based on selected style
    print("\nGenerating plot...")
    if args.style == 'alternative':
        plot_motif_scores_alternative(motif_data, layer_key, args.aggregation, args.sigma, output_file)
    else:
        plot_motif_scores(motif_data, layer_key, args.aggregation, args.sigma, output_file)
    
    if not output_file:
        print("✓ Plot displayed")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot position-wise aggregated motif scores with optional smoothing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic plot with default settings (MAX aggregation, no smoothing)
  python plot_window_scores.py --window-scores-dir window_scores/ --layer 22
  
  # With Gaussian smoothing (sigma=3 for light smoothing)
  python plot_window_scores.py --window-scores-dir window_scores/ --layer 22 --sigma 3
  
  # Heavy smoothing for trend visualization
  python plot_window_scores.py --window-scores-dir window_scores/ --layer 22 --sigma 10 --output smooth_plot.png
  
  # Use MEAN aggregation instead of MAX
  python plot_window_scores.py --window-scores-dir window_scores/ --layer 22 --aggregation mean
  
  # Combine MEAN aggregation with smoothing
  python plot_window_scores.py --window-scores-dir window_scores/ --layer 22 --aggregation mean --sigma 5
  
  # Alternative style (high contrast, no grid/legend)
  python plot_window_scores.py --window-scores-dir window_scores/ --layer 22 --style alternative --output alternative_plot.png
  
  # Save as PDF instead of PNG
  python plot_window_scores.py --window-scores-dir window_scores/ --layer 22 --style alternative --format pdf --output plot.pdf
        """
    )
    
    ap.add_argument("--window-scores-dir", type=str, required=True,
                   help="Directory containing window score JSON files")
    ap.add_argument("--layer", type=int, required=True,
                   help="Layer number to plot (e.g., 22 for L22)")
    ap.add_argument("--aggregation", type=str, choices=['max', 'mean'], default='max',
                   help="Score aggregation method for overlapping windows (default: max)")
    ap.add_argument("--sigma", type=float, default=0,
                   help="Gaussian smoothing sigma (0 = no smoothing, 3-5 = light, 10+ = heavy). Default: 0")
    ap.add_argument("--style", type=str, choices=['default', 'alternative'], default='default',
                   help="Plot style: 'default' (with grid, legend) or 'alternative' (high contrast, no grid/legend, 0s at edges). Default: default")
    ap.add_argument("--format", type=str, choices=['png', 'pdf', 'svg'], default='png',
                   help="Output file format. Default: png")
    ap.add_argument("--output", type=str, default=None,
                   help="Output file path (e.g., plot.png). If not provided, displays plot.")
    
    args = ap.parse_args()
    main(args)

