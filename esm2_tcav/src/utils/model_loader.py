"""ESM2 model loading utilities with registry support"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def load_model_registry(registry_path: str) -> Dict[str, Any]:
    """
    Load model registry from YAML file.
    
    Args:
        registry_path: Path to model_registry.yaml
        
    Returns:
        Dictionary of model configurations
    """
    with open(registry_path, 'r') as f:
        registry = yaml.safe_load(f)
    return registry['models']


def get_model_config(model_name: str, registry_path: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Core feature #8: Enhanced registry with hidden_size validation
    
    Args:
        model_name: Name of the model (e.g., 'esm2_t6_8M_UR50D')
        registry_path: Path to model_registry.yaml
        
    Returns:
        Model configuration dictionary
        
    Raises:
        KeyError: If model not found in registry
    """
    registry = load_model_registry(registry_path)
    
    if model_name not in registry:
        available = ', '.join(registry.keys())
        raise KeyError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {available}"
        )
    
    return registry[model_name]


def load_esm2_model(
    model_name: str, 
    registry_path: str,
    device: str = "cuda"
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """
    Load ESM2 model from local checkpoint with validation.
    
    Core feature #1: Config-driven layer extraction
    Core feature #8: Dimension validation
    
    Args:
        model_name: Name of the model in registry
        registry_path: Path to model_registry.yaml
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, tokenizer, config_dict)
    """
    # Get model config from registry
    config = get_model_config(model_name, registry_path)
    checkpoint_path = config['checkpoint_path']
    
    # Auto-detect device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"Loading {model_name} from {checkpoint_path}")
    
    # Check if it's HuggingFace format first
    import os
    is_huggingface_format = os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin"))
    
    # Load model
    try:
        if is_huggingface_format:
            # HuggingFace format - use transformers
            logger.info("Detected HuggingFace format, using transformers backend")
            try:
                from transformers import EsmModel, EsmTokenizer, AutoTokenizer, AutoModelForMaskedLM
            except ImportError:
                raise ImportError("transformers not found. Install with: pip install transformers")
            
            # Check if this is ESMplusplus or standard ESM2
            model_type = config.get('model_type', 'esm2')  # default to esm2 for backward compatibility
            
            if model_type == 'esmplusplus':
                # ESMplusplus requires AutoModelForMaskedLM with trust_remote_code
                logger.info("Loading ESMplusplus model with trust_remote_code=True")
                model = AutoModelForMaskedLM.from_pretrained(checkpoint_path, trust_remote_code=True)
                tokenizer = model.tokenizer  # Tokenizer is an attribute of the model
                # Get base model for hidden size
                actual_hidden_size = model.config.hidden_size
            else:
                # Standard ESM2 loading
                model = EsmModel.from_pretrained(checkpoint_path)
                
                # Try to load tokenizer, with fallback for custom models
                try:
                    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                except (ValueError, OSError) as e:
                    # ESM++ or other custom models may have incompatible tokenizers
                    # Fall back to standard ESM2 tokenizer which should be compatible
                    logger.warning(f"Could not load tokenizer from checkpoint: {e}")
                    logger.warning("Falling back to standard ESM2 tokenizer")
                    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", do_lower_case=False)
                    
                    # Fix padding_idx mismatch between ESM++ model and ESM2 tokenizer
                    if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'padding_idx'):
                        if model.embeddings.padding_idx is None and hasattr(tokenizer, 'pad_token_id'):
                            logger.warning(f"Fixing padding_idx mismatch: setting model.embeddings.padding_idx = {tokenizer.pad_token_id}")
                            model.embeddings.padding_idx = tokenizer.pad_token_id
                
                actual_hidden_size = model.config.hidden_size
        else:
            # Fair-esm format
            try:
                import esm
            except ImportError:
                raise ImportError("fair-esm not found. Install with: pip install fair-esm")
            
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(
                checkpoint_path
            )
            tokenizer = alphabet.get_batch_converter()
            actual_hidden_size = model.embed_dim
        
        # Move to device
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Validate hidden size matches registry
        expected_hidden_size = config['hidden_size']
        
        if actual_hidden_size != expected_hidden_size:
            raise ValueError(
                f"Hidden size mismatch! "
                f"Registry says {expected_hidden_size}, "
                f"but model has {actual_hidden_size}. "
                f"Check model_registry.yaml"
            )
        
        logger.info(
            f"✓ Model loaded successfully on {device}\n"
            f"  Backend: {'transformers' if is_huggingface_format else 'fair-esm'}\n"
            f"  Hidden size: {actual_hidden_size}\n"
            f"  Layers to extract: {config['layers_to_extract']}"
        )
        
        return model, tokenizer, config
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def get_layers_to_extract(model_name: str, registry_path: str) -> list:
    """
    Get list of layer indices to extract for given model.
    
    Core feature #1: Config-driven layers (no hardcoding!)
    
    Args:
        model_name: Name of the model
        registry_path: Path to registry
        
    Returns:
        List of layer indices
    """
    config = get_model_config(model_name, registry_path)
    return config['layers_to_extract']


def validate_layer_indices(layers: list, num_layers: int) -> None:
    """
    Validate that requested layers are within model bounds.
    
    Args:
        layers: List of layer indices
        num_layers: Total number of layers in model
        
    Raises:
        ValueError: If any layer index is out of bounds
    """
    for layer in layers:
        if layer < 0 or layer > num_layers:
            raise ValueError(
                f"Layer {layer} out of bounds! "
                f"Model has {num_layers} layers (0-indexed)."
            )


