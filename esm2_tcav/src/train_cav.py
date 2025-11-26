"""
CAV training module with versioned artifacts.

Core feature #7: Versioned preprocessing artifacts
"""

import numpy as np
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import hashlib

from .utils.preprocessing import create_scaler_and_pca

logger = logging.getLogger(__name__)


def load_embeddings_for_layer(
    layer: int,
    embed_dir: str,
    pos_file: str = None,
    neg_file: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load positive and negative embeddings for a layer.
    
    Args:
        layer: Layer number
        embed_dir: Directory containing embeddings
        pos_file: Optional custom positive file name
        neg_file: Optional custom negative file name
        
    Returns:
        Tuple of (X, y, sample_indices)
    """
    embed_path = Path(embed_dir)
    
    # Default file names
    if pos_file is None:
        pos_file = f"L{layer}_pos.npy"
    if neg_file is None:
        neg_file = f"L{layer}_neg.npy"
    
    # Load
    pos_emb = np.load(embed_path / pos_file)
    neg_emb = np.load(embed_path / neg_file)
    
    logger.info(
        f"Loaded L{layer}: {len(pos_emb)} positive, {len(neg_emb)} negative"
    )
    
    # Combine
    X = np.vstack([pos_emb, neg_emb])
    y = np.hstack([
        np.ones(len(pos_emb)),
        np.zeros(len(neg_emb))
    ])
    
    # Track sample indices
    sample_indices = np.arange(len(X))
    
    return X, y, sample_indices


def train_concept_cav(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    cv_folds: int = 5,
    random_seed: int = 42
) -> Tuple[LogisticRegression, Dict]:
    """
    Train concept CAV using logistic regression.
    
    Args:
        X_train: Training embeddings (preprocessed)
        y_train: Labels
        C: Regularization strength
        cv_folds: Number of CV folds
        random_seed: Random seed
        
    Returns:
        Tuple of (fitted_model, metrics_dict)
    """
    # Train model
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_seed,
        solver='lbfgs'
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # Fit on full training set
    clf.fit(X_train, y_train)
    
    # Metrics
    y_pred_proba = clf.predict_proba(X_train)[:, 1]
    train_auroc = roc_auc_score(y_train, y_pred_proba)
    train_auprc = average_precision_score(y_train, y_pred_proba)
    
    metrics = {
        'cv_auroc_mean': float(cv_scores.mean()),
        'cv_auroc_std': float(cv_scores.std()),
        'cv_auroc_scores': cv_scores.tolist(),
        'train_auroc': float(train_auroc),
        'train_auprc': float(train_auprc),
        'n_train': len(X_train),
        'n_positive': int(y_train.sum()),
        'n_negative': int((1 - y_train).sum())
    }
    
    logger.info(
        f"Concept CAV trained: "
        f"CV AUROC = {metrics['cv_auroc_mean']:.3f} ± {metrics['cv_auroc_std']:.3f}, "
        f"Train AUROC = {train_auroc:.3f}"
    )
    
    return clf, metrics




def extract_cav_vector(model: LogisticRegression) -> np.ndarray:
    """
    Extract and normalize CAV vector from trained model.
    
    Args:
        model: Fitted logistic regression model
        
    Returns:
        Normalized CAV vector
    """
    # Get weight vector
    cav = model.coef_[0]  # Shape: (n_features,)
    
    # Normalize to unit length
    cav_norm = cav / (np.linalg.norm(cav) + 1e-10)
    
    return cav_norm


def compute_artifact_hash(data: np.ndarray) -> str:
    """
    Compute hash of array for versioning/validation.
    
    Args:
        data: Numpy array
        
    Returns:
        SHA256 hash (first 12 chars)
    """
    hasher = hashlib.sha256()
    hasher.update(data.tobytes())
    return hasher.hexdigest()[:12]


def save_cav_artifacts(
    layer: int,
    output_dir: str,
    concept_cav: np.ndarray,
    scaler: Any,
    pca: Optional[Any],
    concept_metrics: Dict,
    config: Dict,
    artifact_version: str = "v1"
) -> None:
    """
    Save all CAV artifacts with versioning and manifest.
    
    Core feature #7: Versioned artifacts with manifest
    
    Args:
        layer: Layer number
        output_dir: Output directory
        concept_cav: Concept CAV vector
        scaler: Fitted scaler
        pca: Fitted PCA (or None)
        concept_metrics: Concept CAV metrics
        config: Training configuration
        artifact_version: Version string
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    layer_prefix = f"L{layer}"
    
    # Save concept CAV
    concept_file = output_path / f"{layer_prefix}_concept_{artifact_version}.npy"
    np.save(concept_file, concept_cav)
    
    # Save scaler
    scaler_file = output_path / f"{layer_prefix}_scaler_{artifact_version}.pkl"
    joblib.dump(scaler, scaler_file)
    
    # Save PCA if present
    pca_file = None
    if pca is not None:
        pca_file = output_path / f"{layer_prefix}_pca_{artifact_version}.pkl"
        joblib.dump(pca, pca_file)
    
    # Compile report
    report = {
        'layer': layer,
        'artifact_version': artifact_version,
        'timestamp': datetime.now().isoformat(),
        
        # Concept CAV metrics
        'concept_metrics': concept_metrics,
        
        # Training config
        'config': config,
        
        # Artifact hashes for validation
        'artifact_hashes': {
            'concept_cav': compute_artifact_hash(concept_cav),
            'scaler': compute_artifact_hash(scaler.mean_),
        }
    }
    
    if pca is not None:
        report['artifact_hashes']['pca'] = compute_artifact_hash(
            pca.components_
        )
    
    # Save report
    report_file = output_path / f"{layer_prefix}_report_{artifact_version}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Saved CAV artifacts for {layer_prefix}:")
    logger.info(f"  - Concept CAV: {concept_file}")
    logger.info(f"  - Scaler: {scaler_file}")
    if pca_file:
        logger.info(f"  - PCA: {pca_file}")
    logger.info(f"  - Report: {report_file}")
    
    # Create manifest
    create_manifest(layer, output_path, artifact_version)


def create_manifest(layer: int, output_dir: Path, version: str) -> None:
    """
    Create manifest file listing all artifacts for a layer.
    
    Args:
        layer: Layer number
        output_dir: Output directory
        version: Artifact version
    """
    prefix = f"L{layer}"
    
    # Find all artifact files
    artifact_files = {
        'concept_cav': f"{prefix}_concept_{version}.npy",
        'scaler': f"{prefix}_scaler_{version}.pkl",
        'pca': f"{prefix}_pca_{version}.pkl",
        'report': f"{prefix}_report_{version}.json"
    }
    
    # Check existence and compute sizes
    manifest = {
        'layer': layer,
        'version': version,
        'created': datetime.now().isoformat(),
        'files': {}
    }
    
    for key, filename in artifact_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            manifest['files'][key] = {
                'filename': filename,
                'size_bytes': filepath.stat().st_size,
                'exists': True
            }
    
    # Save manifest
    manifest_file = output_dir / f"{prefix}_manifest_{version}.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✓ Created manifest: {manifest_file}")


def train_layer_cavs(
    layer: int,
    embed_dir: str,
    output_dir: str,
    config: Dict,
    artifact_version: str = "v1"
) -> None:
    """
    Complete CAV training pipeline for one layer.
    
    Args:
        layer: Layer number
        embed_dir: Embedding directory
        output_dir: Output directory
        config: Configuration dictionary
        artifact_version: Version string
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training CAVs for Layer {layer}")
    logger.info(f"{'='*60}")
    
    # Load embeddings
    X, y, _ = load_embeddings_for_layer(layer, embed_dir)
    
    # Create preprocessing artifacts
    logger.info("Creating scaler and PCA...")
    scaler, pca, preprocess_meta = create_scaler_and_pca(
        X,
        use_pca=config.get('use_pca', True),
        pca_dim=config.get('pca_dim', 128)
    )
    
    # Preprocess
    X_scaled = scaler.transform(X)
    if pca is not None:
        X_scaled = pca.transform(X_scaled)
    
    # Train concept CAV
    logger.info("Training concept CAV...")
    concept_model, concept_metrics = train_concept_cav(
        X_scaled,
        y,
        C=config.get('regularization_C', 1.0),
        cv_folds=config.get('cv_folds', 5),
        random_seed=config.get('random_seed', 42)
    )
    
    concept_cav = extract_cav_vector(concept_model)
    
    # Add preprocessing metadata to metrics
    concept_metrics.update(preprocess_meta)
    
    # Save all artifacts
    save_cav_artifacts(
        layer=layer,
        output_dir=output_dir,
        concept_cav=concept_cav,
        scaler=scaler,
        pca=pca,
        concept_metrics=concept_metrics,
        config=config,
        artifact_version=artifact_version
    )
    
    logger.info(f"✓ Layer {layer} complete!\n")


