"""Visualization utilities for TCAV analysis"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def plot_roc_pr_curves(
    metrics_by_layer: Dict[str, Dict],
    output_path: str,
    title: str = "ROC and PR Curves by Layer"
) -> None:
    """
    Plot ROC and Precision-Recall curves for all layers.
    
    Args:
        metrics_by_layer: Dict mapping layer names to metrics dicts
        output_path: Path to save figure
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(metrics_by_layer)))
    
    for (layer_name, metrics), color in zip(metrics_by_layer.items(), colors):
        if 'fpr' in metrics and 'tpr' in metrics:
            # ROC curve
            ax1.plot(
                metrics['fpr'], 
                metrics['tpr'],
                label=f"{layer_name} (AUC={metrics['auroc']:.3f})",
                color=color,
                linewidth=2
            )
        
        if 'precision' in metrics and 'recall' in metrics:
            # PR curve
            ax2.plot(
                metrics['recall'],
                metrics['precision'],
                label=f"{layer_name} (AUC={metrics['auprc']:.3f})",
                color=color,
                linewidth=2
            )
    
    # ROC plot styling
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # PR plot styling
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_random_cav_comparison(
    concept_auroc: float,
    random_aurocs: List[float],
    layer_name: str,
    output_path: str,
    alpha: float = 0.05
) -> None:
    """
    Plot concept CAV AUROC vs random CAV distribution.
    
    Core feature #4: Random CAV statistical comparison
    
    Args:
        concept_auroc: AUROC of concept CAV
        random_aurocs: List of AUROCs from random CAVs
        layer_name: Layer identifier
        output_path: Path to save figure
        alpha: Significance level
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of random CAVs
    ax.hist(
        random_aurocs, 
        bins=15, 
        alpha=0.6, 
        color='gray',
        edgecolor='black',
        label=f'Random CAVs (n={len(random_aurocs)})'
    )
    
    # Concept CAV line
    ax.axvline(
        concept_auroc, 
        color='red', 
        linewidth=3,
        linestyle='--',
        label=f'Concept CAV (AUROC={concept_auroc:.3f})'
    )
    
    # Statistics
    random_mean = np.mean(random_aurocs)
    random_std = np.std(random_aurocs)
    z_score = (concept_auroc - random_mean) / (random_std + 1e-10)
    
    # Significance threshold (95th percentile of randoms)
    threshold_95 = np.percentile(random_aurocs, 95)
    ax.axvline(
        threshold_95,
        color='orange',
        linewidth=2,
        linestyle=':',
        label=f'95th percentile ({threshold_95:.3f})'
    )
    
    # Add stats text
    stats_text = (
        f"Random CAVs: μ={random_mean:.3f}, σ={random_std:.3f}\n"
        f"Z-score: {z_score:.2f}\n"
        f"Significant: {'YES ✓' if concept_auroc > threshold_95 else 'NO ✗'}"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax.set_xlabel('AUROC', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(
        f'{layer_name}: Concept CAV vs Random CAVs',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_localization_heatmap(
    sequence: str,
    projection_scores: np.ndarray,
    annotated_span: Optional[Tuple[int, int]],
    accession: str,
    layer_name: str,
    output_path: str,
    window_size: int = 41
) -> None:
    """
    Plot localization heatmap for a single protein.
    
    Args:
        sequence: Protein sequence
        projection_scores: Projection score per position/window
        annotated_span: (start, end) of annotated motif (0-based)
        accession: Protein accession
        layer_name: Layer identifier
        output_path: Path to save figure
        window_size: Sliding window size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), 
                                     gridspec_kw={'height_ratios': [1, 3]})
    
    # Top panel: Projection scores
    positions = np.arange(len(projection_scores))
    ax1.plot(positions, projection_scores, linewidth=2, color='steelblue')
    ax1.fill_between(
        positions, projection_scores, alpha=0.3, color='steelblue'
    )
    
    # Mark annotated region
    if annotated_span:
        start, end = annotated_span
        ax1.axvspan(start, end, alpha=0.2, color='red', 
                    label='Annotated ZnF')
    
    # Mark top prediction
    top_idx = np.argmax(projection_scores)
    ax1.scatter(
        [top_idx], [projection_scores[top_idx]],
        color='red', s=100, zorder=5, marker='*',
        label=f'Top prediction (pos {top_idx})'
    )
    
    ax1.set_ylabel('CAV Projection', fontsize=11)
    ax1.set_xlim(0, len(sequence))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f'{accession} - {layer_name} Localization',
        fontsize=12,
        fontweight='bold'
    )
    
    # Bottom panel: Sequence with coloring
    # Color each position by projection score
    seq_array = np.array(list(sequence))
    
    # Create colormap for sequence
    norm_scores = projection_scores / (projection_scores.max() + 1e-10)
    
    # Plot as colored text (simplified - just show regions)
    cmap = plt.cm.Reds
    for i, (aa, score) in enumerate(zip(sequence, norm_scores)):
        color = cmap(score)
        ax2.text(
            i, 0, aa,
            ha='center', va='center',
            fontsize=6, fontfamily='monospace',
            color=color, weight='bold' if score > 0.7 else 'normal'
        )
    
    # Mark annotated span
    if annotated_span:
        start, end = annotated_span
        ax2.axvspan(start, end, alpha=0.15, color='red')
    
    ax2.set_xlim(-1, len(sequence))
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel('Sequence Position', fontsize=11)
    ax2.set_yticks([])
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_threshold_selection(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    selected_threshold: float,
    layer_name: str,
    output_path: str,
    method: str = "f1_max"
) -> None:
    """
    Plot threshold selection visualization.
    
    Core feature #3: Threshold registry visualization
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        selected_threshold: Chosen threshold
        layer_name: Layer identifier
        output_path: Path to save figure
        method: Threshold selection method
    """
    from sklearn.metrics import f1_score
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: ROC with threshold point
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    
    # Find closest threshold on curve
    idx = np.argmin(np.abs(thresholds_roc - selected_threshold))
    ax1.scatter(
        [fpr[idx]], [tpr[idx]],
        color='red', s=150, zorder=5, marker='*',
        label=f'Selected (t={selected_threshold:.3f})'
    )
    
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve with Selected Threshold', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Precision/Recall/F1 vs threshold
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    
    # Compute F1 for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    ax2.plot(thresholds_pr, precision[:-1], label='Precision', linewidth=2)
    ax2.plot(thresholds_pr, recall[:-1], label='Recall', linewidth=2)
    ax2.plot(thresholds_pr, f1_scores, label='F1 Score', linewidth=2, linestyle='--')
    
    ax2.axvline(
        selected_threshold,
        color='red',
        linewidth=2,
        linestyle=':',
        label=f'Selected threshold'
    )
    
    ax2.set_xlabel('Threshold', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title(
        f'Threshold Selection ({method})',
        fontsize=12,
        fontweight='bold'
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{layer_name} Threshold Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_layer_comparison(
    metrics_by_layer: Dict[str, Dict],
    metric_name: str,
    output_path: str,
    ylabel: str = "AUROC"
) -> None:
    """
    Bar plot comparing a metric across layers.
    
    Args:
        metrics_by_layer: Dict mapping layer names to metrics
        metric_name: Name of metric to plot (e.g., 'auroc')
        output_path: Path to save figure
        ylabel: Y-axis label
    """
    layers = list(metrics_by_layer.keys())
    values = [metrics_by_layer[l][metric_name] for l in layers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(layers, values, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_title(f'{ylabel} by Layer', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


