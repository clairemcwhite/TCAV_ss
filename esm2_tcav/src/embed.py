"""
Embedding extraction module with BOS token indexing guardrails.

Core feature #2: Validate window spans with BOS offset
"""

import torch
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from .utils.data_loader import load_jsonl_data
from .utils.model_loader import (
    load_esm2_model,
    get_layers_to_extract,
    validate_layer_indices
)

logger = logging.getLogger(__name__)


class IndexingValidator:
    """
    Validates and corrects window indexing with BOS token offset.
    
    Core feature #2: Indexing guardrails
    """
    
    def __init__(self, log_corrections: bool = True, corrections_log_path: str = None):
        self.log_corrections = log_corrections
        self.corrections_log_path = corrections_log_path
        self.corrections = []
        
    def validate_window(
        self, 
        window_span: Tuple[int, int],
        sequence_length: int,
        bos_offset: int = 1,
        sample_id: str = "unknown"
    ) -> Tuple[Tuple[int, int], bool]:
        """
        Validate and correct window span accounting for BOS token.
        
        Args:
            window_span: (start, end) in 0-based half-open coordinates
            sequence_length: Length of actual protein sequence
            bos_offset: BOS token offset (ESM2 uses 1)
            sample_id: Identifier for logging
            
        Returns:
            Tuple of (corrected_span, was_corrected)
        """
        start, end = window_span
        
        # Check bounds AFTER BOS shift
        # ESM2 tokenizes as: [BOS, AA1, AA2, ..., AAn, EOS]
        # So window [0, 41) in sequence maps to tokens [1, 42) in model output
        
        max_valid_end = sequence_length
        
        was_corrected = False
        original_span = (start, end)
        
        # Validate start
        if start < 0:
            logger.warning(
                f"Sample {sample_id}: window start {start} < 0, clipping to 0"
            )
            start = 0
            was_corrected = True
        
        # Validate end
        if end > max_valid_end:
            logger.warning(
                f"Sample {sample_id}: window end {end} > seq_len {max_valid_end}, "
                f"clipping to {max_valid_end}"
            )
            end = max_valid_end
            was_corrected = True
        
        # Validate start < end
        if start >= end:
            logger.error(
                f"Sample {sample_id}: invalid window [{start}, {end}), "
                f"setting to full sequence"
            )
            start, end = 0, max_valid_end
            was_corrected = True
        
        corrected_span = (start, end)
        
        if was_corrected and self.log_corrections:
            correction = {
                'sample_id': sample_id,
                'original_span': original_span,
                'corrected_span': corrected_span,
                'sequence_length': sequence_length,
                'reason': 'bounds_violation'
            }
            self.corrections.append(correction)
        
        return corrected_span, was_corrected
    
    def save_corrections_log(self):
        """Save all corrections to JSONL file."""
        if self.corrections_log_path and self.corrections:
            Path(self.corrections_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.corrections_log_path, 'w') as f:
                for correction in self.corrections:
                    f.write(json.dumps(correction) + '\n')
            logger.info(
                f"Saved {len(self.corrections)} indexing corrections to "
                f"{self.corrections_log_path}"
            )


def extract_layer_embeddings(
    model: torch.nn.Module,
    tokenizer: Any,
    sequences: List[Tuple[str, str]],  # [(header, sequence), ...]
    layers: List[int],
    device: str = "cuda",
    batch_size: int = 8
) -> Tuple[Dict[int, np.ndarray], List[np.ndarray]]:
    """
    Extract embeddings from specified layers for a batch of sequences.
    
    Args:
        model: ESM2 model
        tokenizer: ESM2 tokenizer (fair-esm or HuggingFace)
        sequences: List of (header, sequence) tuples
        layers: Layer indices to extract
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (layer_embeddings_dict, attention_masks_list)
        - layer_embeddings_dict: maps layer -> embeddings array (n_seqs, seq_len, hidden_dim)
        - attention_masks_list: list of attention masks (n_seqs, seq_len)
    """
    model.eval()
    
    # Initialize storage
    layer_embeddings = {layer: [] for layer in layers}
    all_attention_masks = []
    
    # Detect tokenizer type
    is_huggingface = hasattr(tokenizer, 'encode')
    
    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        
        if is_huggingface:
            # HuggingFace transformers format
            batch_strs = [seq for _, seq in batch]
            
            # Tokenize sequences with consistent max_length
            # ESM-2 models typically use max_length=1024
            encoded = tokenizer(
                batch_strs,
                padding='max_length',  # Pad to max_length
                truncation=True,
                max_length=1024,  # Consistent length for all sequences
                return_tensors="pt",
                add_special_tokens=True
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Debug: Check shapes
            logger.info(f"Batch shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            
            # Store attention masks
            all_attention_masks.append(attention_mask.cpu().numpy())
            
            # Extract embeddings
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # outputs.hidden_states is a tuple of (embedding, layer1, layer2, ...)
                # Layer 0 is embeddings, so layer N is at index N+1
                for layer in layers:
                    if layer < len(outputs.hidden_states):
                        embeddings = outputs.hidden_states[layer].cpu().numpy()
                        # embeddings shape: (batch_size, seq_len, hidden_dim)
                        # All sequences now have same seq_len due to padding/truncation
                        layer_embeddings[layer].append(embeddings)
                    else:
                        logger.warning(f"Layer {layer} not available in model")
                        
        else:
            # Fair-esm format
            batch_labels, batch_strs, batch_tokens = tokenizer(batch)
            batch_tokens = batch_tokens.to(device)
            
            # Create dummy attention mask (all ones) for fair-esm
            batch_size_actual = batch_tokens.shape[0]
            seq_len = batch_tokens.shape[1]
            dummy_mask = np.ones((batch_size_actual, seq_len))
            all_attention_masks.append(dummy_mask)
            
            # Extract embeddings
            with torch.no_grad():
                results = model(
                    batch_tokens,
                    repr_layers=layers,
                    return_contacts=False
                )
            
            # Store per-layer embeddings
            for layer in layers:
                # results['representations'][layer]: (batch_size, seq_len, hidden_dim)
                embeddings = results['representations'][layer].cpu().numpy()
                layer_embeddings[layer].append(embeddings)
    
    # Concatenate batches
    for layer in layers:
        if layer_embeddings[layer]:
            # Debug: Check shapes before concatenation
            shapes = [emb.shape for emb in layer_embeddings[layer]]
            logger.info(f"Layer {layer} embedding shapes before concatenation: {shapes}")
            
            layer_embeddings[layer] = np.concatenate(layer_embeddings[layer], axis=0)
            logger.info(f"Layer {layer} final shape: {layer_embeddings[layer].shape}")
        else:
            logger.warning(f"No embeddings extracted for layer {layer}")
    
    # Concatenate attention masks
    if all_attention_masks:
        attention_masks = np.concatenate(all_attention_masks, axis=0)
    else:
        attention_masks = []
    
    return layer_embeddings, attention_masks


def pool_window_embeddings(
    embeddings: np.ndarray,
    window_span: Tuple[int, int],
    bos_offset: int = 1,
    attention_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Pool embeddings over a window, accounting for BOS token and padding.
    
    Args:
        embeddings: Full sequence embeddings (seq_len, hidden_dim)
        window_span: (start, end) in sequence coordinates (0-based half-open)
        bos_offset: BOS token offset (default 1 for ESM2)
        attention_mask: Optional attention mask (seq_len,) to ignore padding tokens
        
    Returns:
        Pooled embedding vector (hidden_dim,)
    """
    start, end = window_span
    
    # Shift indices for BOS token
    # Sequence position i maps to token position i+1
    token_start = start + bos_offset
    token_end = end + bos_offset
    
    # Ensure we don't go beyond sequence length
    seq_len = embeddings.shape[0]
    token_start = min(token_start, seq_len)
    token_end = min(token_end, seq_len)
    
    # Extract window embeddings
    window_emb = embeddings[token_start:token_end]  # (window_len, hidden_dim)
    
    if attention_mask is not None:
        # Get attention mask for the window
        window_mask = attention_mask[token_start:token_end]  # (window_len,)
        
        # Only pool over non-padding tokens
        if window_mask.sum() > 0:
            # Weighted average based on attention mask
            window_emb_weighted = window_emb * window_mask[:, np.newaxis]
            pooled = window_emb_weighted.sum(axis=0) / window_mask.sum()
        else:
            # Fallback: mean pooling if no valid tokens
            pooled = window_emb.mean(axis=0)
    else:
        # Mean pooling (original behavior)
        pooled = window_emb.mean(axis=0)
    
    return pooled


def process_dataset(
    jsonl_path: str,
    model: torch.nn.Module,
    tokenizer: Any,
    model_config: Dict[str, Any],
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 8,
    validate_indexing: bool = True,
    corrections_log_path: str = None
) -> None:
    """
    Process entire dataset and save layer-wise embeddings.
    
    Core feature #2: Indexing validation enabled by default
    
    Args:
        jsonl_path: Path to JSONL data file
        model: ESM2 model
        tokenizer: ESM2 tokenizer
        model_config: Model configuration from registry
        output_dir: Directory to save embeddings
        device: Device to run on
        batch_size: Batch size
        validate_indexing: Whether to validate window spans
        corrections_log_path: Where to log corrections
    """
    # Load data
    logger.info(f"Loading data from {jsonl_path}")
    data = load_jsonl_data(jsonl_path)
    logger.info(f"Loaded {len(data)} samples")
    
    # Setup indexing validator
    validator = None
    if validate_indexing:
        validator = IndexingValidator(
            log_corrections=True,
            corrections_log_path=corrections_log_path
        )
    
    # Get layers to extract
    layers = model_config['layers_to_extract']
    validate_layer_indices(layers, model_config['num_layers'])
    
    logger.info(f"Extracting embeddings from layers: {layers}")
    
    # Prepare sequences
    sequences = []
    metadata = []
    
    for sample in data:
        accession = sample['accession']
        sequence = sample['sequence']
        window_span = tuple(sample['window_span_0based_halfopen'])
        
        # Validate window span
        if validator:
            window_span, was_corrected = validator.validate_window(
                window_span,
                len(sequence),
                bos_offset=1,
                sample_id=accession
            )
            
            if was_corrected:
                # Update sample with corrected span
                sample['window_span_0based_halfopen'] = list(window_span)
                sample['_window_corrected'] = True
        
        sequences.append((accession, sequence))
        metadata.append({
            'accession': accession,
            'sequence_length': len(sequence),
            'window_span': window_span,
            'set': sample.get('set', 'unknown')
        })
    
    # Extract full-sequence embeddings for all layers
    logger.info("Extracting embeddings...")
    layer_embeddings, attention_masks = extract_layer_embeddings(
        model, tokenizer, sequences, layers, device, batch_size
    )
    
    # Pool over windows and save per layer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for layer in layers:
        logger.info(f"Processing layer {layer}...")
        
        pooled_embeddings = []
        
        for i, sample_meta in enumerate(tqdm(metadata, desc=f"Layer {layer}")):
            # Get full sequence embedding for this sample
            seq_emb = layer_embeddings[layer][i]  # (seq_len, hidden_dim)
            
            # Get attention mask for this sample (if available)
            sample_attention_mask = attention_masks[i] if len(attention_masks) > i else None
            
            # Pool over window
            window_span = sample_meta['window_span']
            pooled = pool_window_embeddings(
                seq_emb, window_span, bos_offset=1, attention_mask=sample_attention_mask
            )
            pooled_embeddings.append(pooled)
        
        # Convert to array
        pooled_embeddings = np.array(pooled_embeddings)  # (n_samples, hidden_dim)
        
        # Save embeddings
        emb_file = output_path / f"L{layer}_all.npy"
        np.save(emb_file, pooled_embeddings)
        logger.info(
            f"Saved {emb_file}: shape {pooled_embeddings.shape}"
        )
        
        # Save metadata
        meta_file = output_path / f"L{layer}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved {meta_file}")
    
    # Save indexing corrections if any
    if validator:
        validator.save_corrections_log()
        if validator.corrections:
            logger.warning(
                f"⚠ {len(validator.corrections)} windows were corrected. "
                f"Check {corrections_log_path}"
            )
        else:
            logger.info("✓ All window spans passed validation")


def split_embeddings_by_set(
    embedding_file: str,
    metadata_file: str,
    output_dir: str
) -> None:
    """
    Split embeddings into positive and negative sets.
    
    Args:
        embedding_file: Path to combined embeddings (L{X}_all.npy)
        metadata_file: Path to metadata (L{X}_meta.json)
        output_dir: Output directory
    """
    # Load
    embeddings = np.load(embedding_file)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Split by set
    pos_idx = [i for i, m in enumerate(metadata) if m['set'] == 'positive']
    neg_idx = [i for i, m in enumerate(metadata) if m['set'] == 'negative']
    
    pos_emb = embeddings[pos_idx]
    neg_emb = embeddings[neg_idx]
    
    # Save
    layer_name = Path(embedding_file).stem.replace('_all', '')
    
    pos_file = Path(output_dir) / f"{layer_name}_pos.npy"
    neg_file = Path(output_dir) / f"{layer_name}_neg.npy"
    
    np.save(pos_file, pos_emb)
    np.save(neg_file, neg_emb)
    
    logger.info(f"Split {layer_name}: {len(pos_idx)} pos, {len(neg_idx)} neg")

