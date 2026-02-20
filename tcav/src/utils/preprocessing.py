"""Preprocessing utilities for embeddings (standardization, PCA)"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_scaler_and_pca(
    X_train: np.ndarray,
    use_pca: bool = True,
    pca_dim: Optional[int] = 128
) -> Tuple[StandardScaler, Optional[PCA], dict]:
    """
    Create and fit scaler and PCA on training data.
    
    Core feature #7: Versioned preprocessing artifacts
    
    Args:
        X_train: Training data (n_samples, n_features)
        use_pca: Whether to apply PCA
        pca_dim: Target dimensionality for PCA
        
    Returns:
        Tuple of (scaler, pca, metadata_dict)
    """
    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    metadata = {
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'input_dim': X_train.shape[1]
    }
    
    # Fit PCA if requested
    pca = None
    if use_pca and pca_dim is not None:
        # Don't exceed input dimensionality
        actual_pca_dim = min(pca_dim, X_train.shape[1])
        
        pca = PCA(n_components=actual_pca_dim, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        
        metadata.update({
            'pca_applied': True,
            'pca_dim': actual_pca_dim,
            'pca_variance_explained': float(variance_explained),
            'output_dim': actual_pca_dim
        })
        
        logger.info(
            f"PCA: {X_train.shape[1]} → {actual_pca_dim} dims, "
            f"{variance_explained:.2%} variance explained"
        )
    else:
        metadata.update({
            'pca_applied': False,
            'output_dim': X_train.shape[1]
        })
    
    return scaler, pca, metadata


def standardize_features(
    X: np.ndarray,
    scaler: StandardScaler
) -> np.ndarray:
    """
    Apply fitted scaler to data.
    
    Args:
        X: Data to standardize
        scaler: Fitted StandardScaler
        
    Returns:
        Standardized data
    """
    return scaler.transform(X)


def apply_pca(
    X: np.ndarray,
    pca: PCA
) -> np.ndarray:
    """
    Apply fitted PCA to data.
    
    Args:
        X: Data to transform (should already be standardized)
        pca: Fitted PCA object
        
    Returns:
        PCA-transformed data
    """
    if pca is None:
        return X
    return pca.transform(X)


def preprocess_embeddings(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: Optional[PCA] = None
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to embeddings.
    
    Args:
        X: Raw embeddings
        scaler: Fitted scaler
        pca: Optional fitted PCA
        
    Returns:
        Preprocessed embeddings
    """
    X_scaled = standardize_features(X, scaler)
    
    if pca is not None:
        X_scaled = apply_pca(X_scaled, pca)
    
    return X_scaled


def validate_preprocessing_compatibility(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: Optional[PCA] = None
) -> None:
    """
    Validate that data dimensions match preprocessing artifacts.
    
    Args:
        X: Data to validate
        scaler: Scaler to check against
        pca: Optional PCA to check against
        
    Raises:
        ValueError: If dimensions don't match
    """
    expected_dim = len(scaler.mean_)
    actual_dim = X.shape[1]
    
    if actual_dim != expected_dim:
        raise ValueError(
            f"Dimension mismatch! "
            f"Scaler expects {expected_dim} features, "
            f"but data has {actual_dim}. "
            f"Did you load the wrong scaler/embeddings?"
        )
    
    if pca is not None:
        # After scaling, check PCA compatibility
        X_scaled = scaler.transform(X[:1])  # Just check first sample
        pca_input_dim = pca.components_.shape[1]
        
        if X_scaled.shape[1] != pca_input_dim:
            raise ValueError(
                f"PCA dimension mismatch! "
                f"PCA expects {pca_input_dim} features after scaling."
            )

