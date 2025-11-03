"""
CAV training module with random CAVs and versioned artifacts.

Core feature #4: Random CAV generation and comparison
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


def train_random_cavs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_random: int = 20,
    random_mode: str = "label_shuffle",
    C: float = 1.0,
    random_seed_base: int = 42
) -> Tuple[List[LogisticRegression], List[float]]:
    """
    Train random control CAVs.
    
    Core feature #4: Random CAV generation with configurable strategy
    
    Args:
        X_train: Training embeddings
        y_train: True labels
        n_random: Number of random CAVs to train
        random_mode: "label_shuffle", "feature_permute", or "gaussian_noise"
        C: Regularization strength
        random_seed_base: Base seed (each CAV gets seed+i)
        
    Returns:
        Tuple of (random_models, random_aurocs)
    """
    random_models = []
    random_aurocs = []
    
    logger.info(f"Training {n_random} random CAVs (mode: {random_mode})...")
    
    for i in range(n_random):
        seed = random_seed_base + i
        np.random.seed(seed)
        
        # Generate random labels/features based on mode
        if random_mode == "label_shuffle":
            # Shuffle labels (preserves class balance)
            y_random = np.random.permutation(y_train)
            X_random = X_train
            
        elif random_mode == "feature_permute":
            # Permute each feature independently
            X_random = np.random.permutation(X_train)
            y_random = y_train
            
        elif random_mode == "gaussian_noise":
            # Replace with Gaussian noise matching statistics
            X_random = np.random.randn(*X_train.shape)
            X_random = X_random * X_train.std(axis=0) + X_train.mean(axis=0)
            y_random = y_train
        else:
            raise ValueError(f"Unknown random_mode: {random_mode}")
        
        # Train random CAV
        clf_random = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=seed,
            solver='lbfgs'
        )
        clf_random.fit(X_random, y_random)
        
        # Evaluate on ORIGINAL data/labels
        y_pred_proba = clf_random.predict_proba(X_train)[:, 1]
        auroc = roc_auc_score(y_train, y_pred_proba)
        
        random_models.append(clf_random)
        random_aurocs.append(auroc)
    
    logger.info(
        f"Random CAVs AUROC: "
        f"{np.mean(random_aurocs):.3f} ± {np.std(random_aurocs):.3f} "
        f"(expected ~0.50)"
    )
    
    return random_models, random_aurocs


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
    random_cavs: List[np.ndarray],
    scaler: Any,
    pca: Optional[Any],
    concept_metrics: Dict,
    random_aurocs: List[float],
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
        random_cavs: List of random CAV vectors
        scaler: Fitted scaler
        pca: Fitted PCA (or None)
        concept_metrics: Concept CAV metrics
        random_aurocs: Random CAV AUROCs
        config: Training configuration
        artifact_version: Version string
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    layer_prefix = f"L{layer}"
    
    # Save concept CAV
    concept_file = output_path / f"{layer_prefix}_concept_{artifact_version}.npy"
    np.save(concept_file, concept_cav)
    
    # Save random CAVs
    for i, random_cav in enumerate(random_cavs):
        random_file = output_path / f"{layer_prefix}_random_{i:02d}_{artifact_version}.npy"
        np.save(random_file, random_cav)
    
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
        
        # Random CAV statistics
        'random_cav_stats': {
            'n_random_cavs': len(random_cavs),
            'auroc_mean': float(np.mean(random_aurocs)),
            'auroc_std': float(np.std(random_aurocs)),
            'auroc_min': float(np.min(random_aurocs)),
            'auroc_max': float(np.max(random_aurocs)),
            'aurocs': random_aurocs
        },
        
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
    logger.info(f"  - Random CAVs: {len(random_cavs)} files")
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
        'report': f"{prefix}_report_{version}.json",
        'random_cavs': []
    }
    
    # Find random CAV files
    for f in output_dir.glob(f"{prefix}_random_*_{version}.npy"):
        artifact_files['random_cavs'].append(f.name)
    
    # Check existence and compute sizes
    manifest = {
        'layer': layer,
        'version': version,
        'created': datetime.now().isoformat(),
        'files': {}
    }
    
    for key, filename in artifact_files.items():
        if key == 'random_cavs':
            continue
        filepath = output_dir / filename
        if filepath.exists():
            manifest['files'][key] = {
                'filename': filename,
                'size_bytes': filepath.stat().st_size,
                'exists': True
            }
    
    # Add random CAVs
    manifest['files']['random_cavs'] = []
    for filename in sorted(artifact_files['random_cavs']):
        filepath = output_dir / filename
        manifest['files']['random_cavs'].append({
            'filename': filename,
            'size_bytes': filepath.stat().st_size
        })
    
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
    
    # Train random CAVs
    logger.info("Training random CAVs...")
    random_models, random_aurocs = train_random_cavs(
        X_scaled,
        y,
        n_random=config.get('n_random_cavs', 20),
        random_mode=config.get('random_mode', 'label_shuffle'),
        C=config.get('regularization_C', 1.0),
        random_seed_base=config.get('random_seed_base', 42)
    )
    
    random_cavs = [extract_cav_vector(m) for m in random_models]
    
    # Add preprocessing metadata to metrics
    concept_metrics.update(preprocess_meta)
    
    # Save all artifacts
    save_cav_artifacts(
        layer=layer,
        output_dir=output_dir,
        concept_cav=concept_cav,
        random_cavs=random_cavs,
        scaler=scaler,
        pca=pca,
        concept_metrics=concept_metrics,
        random_aurocs=random_aurocs,
        config=config,
        artifact_version=artifact_version
    )
    
    logger.info(f"✓ Layer {layer} complete!\n")

