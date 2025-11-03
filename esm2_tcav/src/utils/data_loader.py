"""Data loading utilities for JSONL and FASTA files"""

import json
from typing import List, Dict, Any, Tuple
from pathlib import Path


def load_jsonl_data(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        List of sample dictionaries
    """
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """
    Parse FASTA file into dictionary.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Dictionary mapping accession to sequence
    """
    sequences = {}
    current_acc = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_acc:
                    sequences[current_acc] = ''.join(current_seq)
                # Parse new header
                current_acc = line[1:].split()[0]  # First word after >
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_acc:
            sequences[current_acc] = ''.join(current_seq)
    
    return sequences


def validate_jsonl_schema(sample: Dict[str, Any]) -> bool:
    """
    Validate that JSONL sample has required fields.
    
    Args:
        sample: Sample dictionary from JSONL
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['accession', 'sequence', 'window_span_0based_halfopen']
    return all(field in sample for field in required_fields)


def get_train_test_split(
    data: List[Dict[str, Any]], 
    train_ratio: float = 0.8, 
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train and test sets with stratification.
    
    Args:
        data: List of samples
        train_ratio: Fraction for training
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
    """
    import random
    random.seed(random_seed)
    
    # Shuffle data
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]
    
    return train_data, test_data


