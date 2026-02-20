"""
CAV training module with versioned artifacts.
"""

import numpy as np
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import hashlib

from .utils.preprocessing import create_scaler_and_pca

logger = logging.getLogger(__name__)


def load_embeddings(
    embed_dir: str,
    pos_file: str = "pos.npy",
    neg_file: str = "neg.npy"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load positive and negative embeddings.

    Args:
        embed_dir: Directory containing embeddings
        pos_file: Positive embeddings filename
        neg_file: Negative embeddings filename

    Returns:
        Tuple of (X, y)
    """
    embed_path = Path(embed_dir)

    pos_emb = np.load(embed_path / pos_file)
    neg_emb = np.load(embed_path / neg_file)

    logger.info(f"Loaded {len(pos_emb)} positive, {len(neg_emb)} negative embeddings")

    X = np.vstack([pos_emb, neg_emb])
    y = np.hstack([np.ones(len(pos_emb)), np.zeros(len(neg_emb))])

    return X, y


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
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_seed,
        solver='lbfgs'
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')

    clf.fit(X_train, y_train)

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
        f"CAV trained: CV AUROC = {metrics['cv_auroc_mean']:.3f} ± {metrics['cv_auroc_std']:.3f}, "
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
    cav = model.coef_[0]
    cav_norm = cav / (np.linalg.norm(cav) + 1e-10)
    return cav_norm


def compute_artifact_hash(data: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(data.tobytes())
    return hasher.hexdigest()[:12]


def save_cav_artifacts(
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

    Args:
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

    concept_file = output_path / f"concept_{artifact_version}.npy"
    np.save(concept_file, concept_cav)

    scaler_file = output_path / f"scaler_{artifact_version}.pkl"
    joblib.dump(scaler, scaler_file)

    pca_file = None
    if pca is not None:
        pca_file = output_path / f"pca_{artifact_version}.pkl"
        joblib.dump(pca, pca_file)

    report = {
        'artifact_version': artifact_version,
        'timestamp': datetime.now().isoformat(),
        'concept_metrics': concept_metrics,
        'config': config,
        'artifact_hashes': {
            'concept_cav': compute_artifact_hash(concept_cav),
            'scaler': compute_artifact_hash(scaler.mean_),
        }
    }

    if pca is not None:
        report['artifact_hashes']['pca'] = compute_artifact_hash(pca.components_)

    report_file = output_path / f"report_{artifact_version}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved CAV artifacts to {output_path}")

    create_manifest(output_path, artifact_version)


def create_manifest(output_dir: Path, version: str) -> None:
    artifact_files = {
        'concept_cav': f"concept_{version}.npy",
        'scaler': f"scaler_{version}.pkl",
        'pca': f"pca_{version}.pkl",
        'report': f"report_{version}.json"
    }

    manifest = {
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

    manifest_file = output_dir / f"manifest_{version}.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)


def train_cav(
    embed_dir: str,
    output_dir: str,
    config: Dict,
    artifact_version: str = "v1",
    holdout_fraction: float = 0.0
) -> None:
    """
    Complete CAV training pipeline.

    Args:
        embed_dir: Directory containing pos.npy and neg.npy
        output_dir: Output directory for artifacts
        config: Configuration dictionary
        artifact_version: Version string
        holdout_fraction: Fraction of data to hold out for evaluation (0 = no holdout)
    """
    X, y = load_embeddings(embed_dir)

    holdout_metrics = None
    if holdout_fraction > 0.0:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y,
            test_size=holdout_fraction,
            stratify=y,
            random_state=config.get('random_seed', 42)
        )
        logger.info(
            f"Holdout split: {len(X_train)} train, {len(X_holdout)} holdout"
        )
    else:
        X_train, y_train = X, y
        X_holdout, y_holdout = None, None

    scaler, pca, preprocess_meta = create_scaler_and_pca(
        X_train,
        use_pca=config.get('use_pca', True),
        pca_dim=config.get('pca_dim', 128)
    )

    X_scaled = scaler.transform(X_train)
    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    concept_model, concept_metrics = train_concept_cav(
        X_scaled,
        y_train,
        C=config.get('regularization_C', 1.0),
        cv_folds=config.get('cv_folds', 5),
        random_seed=config.get('random_seed', 42)
    )

    if X_holdout is not None:
        X_holdout_scaled = scaler.transform(X_holdout)
        if pca is not None:
            X_holdout_scaled = pca.transform(X_holdout_scaled)
        cav_vec = extract_cav_vector(concept_model)
        holdout_scores = X_holdout_scaled @ cav_vec
        holdout_auroc = roc_auc_score(y_holdout, holdout_scores)
        holdout_auprc = average_precision_score(y_holdout, holdout_scores)
        holdout_metrics = {
            'holdout_auroc': float(holdout_auroc),
            'holdout_auprc': float(holdout_auprc),
            'n_holdout': len(X_holdout),
            'holdout_fraction': holdout_fraction
        }
        concept_metrics.update(holdout_metrics)
        logger.info(
            f"Holdout AUROC = {holdout_auroc:.3f}, AUPRC = {holdout_auprc:.3f}"
        )

    concept_cav = extract_cav_vector(concept_model)
    concept_metrics.update(preprocess_meta)

    save_cav_artifacts(
        output_dir=output_dir,
        concept_cav=concept_cav,
        scaler=scaler,
        pca=pca,
        concept_metrics=concept_metrics,
        config=config,
        artifact_version=artifact_version
    )
