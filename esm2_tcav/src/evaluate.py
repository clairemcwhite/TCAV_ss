"""
Evaluation module with threshold registry and efficient visualization.

Core feature #3: Threshold registry
Core feature #6: Smart heatmap selection (top/bottom/random)
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    roc_auc_score, average_precision_score, f1_score
)
import scipy.stats as stats

from .utils.visualization import (
    plot_roc_pr_curves,
    plot_random_cav_comparison,
    plot_localization_heatmap,
    plot_threshold_selection,
    plot_layer_comparison
)
from .utils.preprocessing import preprocess_embeddings

logger = logging.getLogger(__name__)


def load_cav_artifacts(
    layer: int,
    cav_dir: str,
    version: str = "v1"
) -> Dict:
    """
    Load all CAV artifacts for a layer.
    
    Args:
        layer: Layer number
        cav_dir: CAV directory
        version: Artifact version
        
    Returns:
        Dictionary with all artifacts
    """
    import joblib
    
    cav_path = Path(cav_dir)
    prefix = f"L{layer}"
    
    # Load concept CAV
    concept_file = cav_path / f"{prefix}_concept_{version}.npy"
    concept_cav = np.load(concept_file)
    
    # Load random CAVs
    random_cavs = []
    for random_file in sorted(cav_path.glob(f"{prefix}_random_*_{version}.npy")):
        random_cavs.append(np.load(random_file))
    
    # Load scaler
    scaler_file = cav_path / f"{prefix}_scaler_{version}.pkl"
    scaler = joblib.load(scaler_file)
    
    # Load PCA if exists
    pca_file = cav_path / f"{prefix}_pca_{version}.pkl"
    pca = joblib.load(pca_file) if pca_file.exists() else None
    
    # Load report
    report_file = cav_path / f"{prefix}_report_{version}.json"
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    logger.info(
        f"Loaded L{layer} artifacts: "
        f"concept CAV + {len(random_cavs)} random CAVs"
    )
    
    return {
        'concept_cav': concept_cav,
        'random_cavs': random_cavs,
        'scaler': scaler,
        'pca': pca,
        'report': report
    }


def compute_projections(
    embeddings: np.ndarray,
    cav: np.ndarray,
    scaler: Any,
    pca: Optional[Any] = None
) -> np.ndarray:
    """
    Compute CAV projections for embeddings.
    
    Args:
        embeddings: Raw embeddings (n_samples, hidden_dim)
        cav: CAV vector (preprocessed_dim,)
        scaler: Fitted scaler
        pca: Optional fitted PCA
        
    Returns:
        Projection scores (n_samples,)
    """
    # Preprocess embeddings
    X_preprocessed = preprocess_embeddings(embeddings, scaler, pca)
    
    # Project onto CAV
    projections = X_preprocessed @ cav
    
    return projections


def evaluate_projection_performance(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict:
    """
    Compute classification metrics from projection scores.
    
    Args:
        y_true: True labels
        y_scores: Projection scores
        
    Returns:
        Dictionary of metrics
    """
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)
    
    # PR curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    
    metrics = {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds_roc': thresholds_roc.tolist(),
        'thresholds_pr': thresholds_pr.tolist()
    }
    
    return metrics


def select_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    method: str = "f1_max"
) -> Tuple[float, Dict]:
    """
    Select optimal threshold based on method.
    
    Core feature #3: Systematic threshold selection
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        method: Selection method
            - "f1_max": Maximize F1 score
            - "precision_at_recall_90": Precision when recall >= 0.90
            - "fpr_0.05": Threshold at 5% FPR
            
    Returns:
        Tuple of (threshold, metadata_dict)
    """
    from sklearn.metrics import f1_score
    
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    
    if method == "f1_max":
        # Maximize F1 score
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
            precision[:-1] + recall[:-1] + 1e-10
        )
        best_idx = np.argmax(f1_scores)
        threshold = thresholds_pr[best_idx]
        
        metadata = {
            'method': 'f1_max',
            'f1_score': float(f1_scores[best_idx]),
            'precision': float(precision[best_idx]),
            'recall': float(recall[best_idx])
        }
    
    elif method == "precision_at_recall_90":
        # Find threshold where recall >= 0.90
        valid_idx = recall[:-1] >= 0.90
        if valid_idx.any():
            idx = np.where(valid_idx)[0][0]
            threshold = thresholds_pr[idx]
        else:
            # Fallback to max F1
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
                precision[:-1] + recall[:-1] + 1e-10
            )
            idx = np.argmax(f1_scores)
            threshold = thresholds_pr[idx]
            logger.warning("Could not achieve recall >= 0.90, using max F1")
        
        metadata = {
            'method': 'precision_at_recall_90',
            'precision': float(precision[idx]),
            'recall': float(recall[idx])
        }
    
    elif method == "fpr_0.05":
        # Find threshold where FPR <= 0.05
        valid_idx = fpr <= 0.05
        if valid_idx.any():
            idx = np.where(valid_idx)[0][-1]  # Last index where FPR <= 0.05
            threshold = thresholds_roc[idx]
        else:
            # Fallback
            idx = 0
            threshold = thresholds_roc[idx]
            logger.warning("Could not achieve FPR <= 0.05, using strictest threshold")
        
        metadata = {
            'method': 'fpr_0.05',
            'fpr': float(fpr[idx]),
            'tpr': float(tpr[idx])
        }
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    logger.info(f"Selected threshold: {threshold:.4f} (method: {method})")
    
    return float(threshold), metadata


def compare_with_random_cavs(
    concept_auroc: float,
    random_aurocs: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Statistical comparison of concept CAV vs random CAVs.
    
    Core feature #4: Random CAV comparison
    
    Args:
        concept_auroc: AUROC of concept CAV
        random_aurocs: AUROCs of random CAVs
        alpha: Significance level
        
    Returns:
        Dictionary of comparison statistics
    """
    random_mean = np.mean(random_aurocs)
    random_std = np.std(random_aurocs)
    
    # Z-score
    z_score = (concept_auroc - random_mean) / (random_std + 1e-10)
    
    # P-value (one-sided test: concept > random)
    p_value = 1 - stats.norm.cdf(z_score)
    
    # 95th percentile threshold
    threshold_95 = np.percentile(random_aurocs, 95)
    
    is_significant = concept_auroc > threshold_95
    
    comparison = {
        'concept_auroc': float(concept_auroc),
        'random_auroc_mean': float(random_mean),
        'random_auroc_std': float(random_std),
        'random_auroc_min': float(np.min(random_aurocs)),
        'random_auroc_max': float(np.max(random_aurocs)),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'threshold_95': float(threshold_95),
        'is_significant': bool(is_significant),
        'alpha': alpha
    }
    
    logger.info(
        f"Random CAV comparison: "
        f"z={z_score:.2f}, p={p_value:.4f}, "
        f"significant={'YES' if is_significant else 'NO'}"
    )
    
    return comparison


def evaluate_layer(
    layer: int,
    embed_dir: str,
    cav_dir: str,
    output_dir: str,
    artifact_version: str = "v1",
    threshold_method: str = "f1_max"
) -> Dict:
    """
    Evaluate CAVs for one layer.
    
    Args:
        layer: Layer number
        embed_dir: Embedding directory
        cav_dir: CAV directory
        output_dir: Output directory
        artifact_version: CAV version
        threshold_method: Threshold selection method
        
    Returns:
        Dictionary of evaluation metrics
    """
    from .train_cav import load_embeddings_for_layer
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Layer {layer}")
    logger.info(f"{'='*60}")
    
    # Load embeddings
    X, y, _ = load_embeddings_for_layer(layer, embed_dir)
    
    # Load CAV artifacts
    artifacts = load_cav_artifacts(layer, cav_dir, artifact_version)
    
    # Compute concept CAV projections
    concept_projections = compute_projections(
        X,
        artifacts['concept_cav'],
        artifacts['scaler'],
        artifacts['pca']
    )
    
    # Compute metrics
    metrics = evaluate_projection_performance(y, concept_projections)
    
    # Select optimal threshold
    threshold, threshold_meta = select_optimal_threshold(
        y, concept_projections, method=threshold_method
    )
    
    metrics['threshold'] = threshold
    metrics['threshold_metadata'] = threshold_meta
    
    # Evaluate random CAVs
    random_projections_list = []
    for random_cav in artifacts['random_cavs']:
        random_proj = compute_projections(
            X, random_cav, artifacts['scaler'], artifacts['pca']
        )
        random_projections_list.append(random_proj)
    
    # Compute random AUROCs
    random_aurocs = [
        roc_auc_score(y, proj) for proj in random_projections_list
    ]
    
    # Statistical comparison
    comparison = compare_with_random_cavs(
        metrics['auroc'],
        random_aurocs
    )
    
    metrics['random_cav_comparison'] = comparison
    
    logger.info(
        f"L{layer} Performance: "
        f"AUROC={metrics['auroc']:.3f}, "
        f"AUPRC={metrics['auprc']:.3f}, "
        f"Threshold={threshold:.4f}"
    )
    
    return metrics


def save_evaluation_results(
    metrics_by_layer: Dict[str, Dict],
    output_dir: str
) -> None:
    """
    Save evaluation results to JSON.
    
    Args:
        metrics_by_layer: Dict mapping layer names to metrics
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full metrics
    metrics_file = output_path / "projection_eval.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_by_layer, f, indent=2)
    
    logger.info(f"✓ Saved evaluation metrics: {metrics_file}")


def save_threshold_registry(
    metrics_by_layer: Dict[str, Dict],
    output_path: str
) -> None:
    """
    Save threshold registry for use in detection.
    
    Core feature #3: Threshold registry
    
    Args:
        metrics_by_layer: Dict mapping layer names to metrics
        output_path: Path to save thresholds.json
    """
    thresholds = {}
    
    for layer_name, metrics in metrics_by_layer.items():
        thresholds[layer_name] = {
            'threshold': metrics['threshold'],
            'method': metrics['threshold_metadata']['method'],
            'auroc': metrics['auroc'],
            'auprc': metrics['auprc']
        }
        thresholds[layer_name].update(metrics['threshold_metadata'])
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    logger.info(f"✓ Saved threshold registry: {output_file}")


def create_evaluation_visualizations(
    metrics_by_layer: Dict[str, Dict],
    cav_dir: str,
    output_dir: str,
    artifact_version: str = "v1"
) -> None:
    """
    Create evaluation plots.
    
    Args:
        metrics_by_layer: Metrics for all layers
        cav_dir: CAV directory (for random CAV data)
        output_dir: Output directory
        artifact_version: CAV version
    """
    output_path = Path(output_dir) / "plots"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ROC/PR curves
    plot_roc_pr_curves(
        metrics_by_layer,
        str(output_path / "roc_pr_curves.png")
    )
    
    # Layer comparison (AUROC)
    plot_layer_comparison(
        metrics_by_layer,
        'auroc',
        str(output_path / "auroc_by_layer.png"),
        ylabel="AUROC"
    )
    
    # Random CAV comparisons (per layer)
    for layer_name, metrics in metrics_by_layer.items():
        layer_num = int(layer_name.replace('L', ''))
        
        # Load random AUROC from artifacts
        artifacts = load_cav_artifacts(layer_num, cav_dir, artifact_version)
        random_aurocs = artifacts['report']['random_cav_stats']['aurocs']
        
        plot_random_cav_comparison(
            metrics['auroc'],
            random_aurocs,
            layer_name,
            str(output_path / f"{layer_name}_random_comparison.png")
        )
        
        # Threshold selection plot
        # Note: Need y_true and y_scores - skip for now or load from embeddings
        # plot_threshold_selection(...)
    
    logger.info(f"✓ Created evaluation visualizations in {output_path}")

