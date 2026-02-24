#!/usr/bin/env python3
"""
CAV-guided masked language model generation with ESMplusplus.

A masked region of a protein sequence is filled in through a single forward pass.
During that pass, the CAV direction is added to the hidden states at every
transformer layer for the masked positions only. The hope is the generated
sequence in the masked region adopts the character of the trained concept
(e.g. a transmembrane helix).

The mask region may extend beyond the natural length of the input sequence —
use this to append a novel C-terminal (or N-terminal) region of a specified
length to an existing protein.

Examples
--------
# Replace an internal region (residues 24-45) with TM-steered generation:
python specific_scripts/cav_guided_generation.py \\
    --sequence MKTIIALSYIFCLVFAQQQDSGV... \\
    --model /groups/clairemcwhite/models/ESMplusplus_large \\
    --cav-dir ./cavs/GO_0005887/ \\
    --mask-start 24 \\
    --mask-end 46 \\
    --cav-weight 0.1 \\
    --out generated.fasta

# Append a novel 23-residue C-terminal TM region to a 300 aa protein:
python specific_scripts/cav_guided_generation.py \\
    --sequence MKTIIALSYIFCLVFA...  \\
    --model /groups/clairemcwhite/models/ESMplusplus_large \\
    --cav-dir ./cavs/GO_0005887/ \\
    --mask-start 300 \\
    --mask-end 323 \\
    --cav-weight 0.1 \\
    --out generated_cterm_tm.fasta

# Read sequence from a FASTA file:
python specific_scripts/cav_guided_generation.py \\
    --sequence my_protein.fasta \\
    --model /groups/clairemcwhite/models/ESMplusplus_large \\
    --cav-dir ./cavs/GO_0005887/ \\
    --mask-start 24 \\
    --mask-end 46 \\
    --out generated.fasta

# Baseline (no steering) for comparison:
python specific_scripts/cav_guided_generation.py \\
    --sequence MKTIIALSYIFCLVFAQQQDSGV... \\
    --model /groups/clairemcwhite/models/ESMplusplus_large \\
    --cav-dir ./cavs/GO_0005887/ \\
    --mask-start 24 \\
    --mask-end 46 \\
    --cav-weight 0.0 \\
    --out baseline.fasta
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence I/O
# ---------------------------------------------------------------------------

def read_sequence(sequence_arg: str) -> tuple[str, str]:
    """
    Accept either a raw amino-acid string or a path to a FASTA file.

    Returns (sequence, name) where name is the FASTA header (without '>') or
    'input_sequence' for raw strings.
    """
    p = Path(sequence_arg)
    if p.exists() and p.suffix.lower() in ('.fasta', '.fa', '.faa', '.fas'):
        with open(p) as fh:
            lines = fh.read().splitlines()
        name = lines[0].lstrip('>').strip() if lines and lines[0].startswith('>') else 'input_sequence'
        seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
        return seq, name
    # Treat as a raw sequence string
    return sequence_arg.strip(), 'input_sequence'


# ---------------------------------------------------------------------------
# CAV loading and back-projection
# ---------------------------------------------------------------------------

def load_and_backproject_cav(cav_dir: str, version: str = 'v1') -> np.ndarray:
    """
    Load CAV artifacts and back-project the concept vector into the model's
    full hidden-state space.

    The CAV was trained in StandardScaler + PCA-compressed space (default 128
    dims).  To inject it into the residual stream (shape: hidden_dim), we
    need to invert both transforms:

        cav_fullspace = pca.components_.T @ cav * scaler.scale_

    If no PCA file is found the CAV is assumed to already be in scaler-space:

        cav_fullspace = cav * scaler.scale_

    The result is a float32 numpy array of shape (hidden_dim,).
    """
    cav_path    = Path(cav_dir) / f"concept_{version}.npy"
    scaler_path = Path(cav_dir) / f"scaler_{version}.pkl"
    pca_path    = Path(cav_dir) / f"pca_{version}.pkl"

    if not cav_path.exists():
        raise FileNotFoundError(f"CAV not found: {cav_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    cav    = np.load(cav_path).astype(np.float32)      # (n_pca_components,) or (hidden_dim,)
    scaler = joblib.load(scaler_path)

    if pca_path.exists():
        pca = joblib.load(pca_path)
        logger.info(f"PCA found: projecting CAV from {cav.shape[0]}-dim PCA space "
                    f"back to {pca.components_.shape[1]}-dim hidden space")
        # pca.components_: (n_components, hidden_dim)  →  .T: (hidden_dim, n_components)
        cav_fullspace = (pca.components_.T @ cav).astype(np.float32)  # (hidden_dim,)
    else:
        logger.info("No PCA file found — using CAV directly in scaler space")
        cav_fullspace = cav.astype(np.float32)

    # Un-standardize: the scaler divided by scale_, so multiply back
    cav_fullspace = cav_fullspace * scaler.scale_.astype(np.float32)

    logger.info(f"CAV back-projected to shape {cav_fullspace.shape}, "
                f"L2 norm = {np.linalg.norm(cav_fullspace):.4f}")
    return cav_fullspace


# ---------------------------------------------------------------------------
# Forward-hook factory
# ---------------------------------------------------------------------------

def make_layer_hook(cav_tensor: torch.Tensor,
                    mask_positions: list[int],
                    cav_weight: float):
    """
    Return a forward hook that adds cav_weight * cav_tensor to hidden states
    at mask_positions after each transformer layer.

    The hook expects the layer output to be a tuple whose first element is the
    hidden-state tensor (batch, seq_len, hidden_dim).  This matches both
    EsmLayer and the standard HuggingFace transformer layer convention.
    """
    def hook(module, input, output):
        # output is a tuple; output[0] is the hidden state
        if isinstance(output, tuple):
            hidden = output[0]
            hidden[:, mask_positions, :] += cav_weight * cav_tensor
            return (hidden,) + output[1:]
        else:
            # Fallback: output is a plain tensor
            output[:, mask_positions, :] += cav_weight * cav_tensor
            return output
    return hook


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate(
    sequence: str,
    model_path: str,
    cav_dir: str,
    mask_start: int,
    mask_end: int,
    cav_weight: float = 0.1,
    device: str = 'cuda',
    version: str = 'v1',
) -> str:
    """
    Generate residues in [mask_start, mask_end) guided by the trained CAV.

    If mask_end > len(sequence) the sequence is first extended with mask tokens
    to reach mask_end, allowing novel residues to be appended beyond the
    natural terminus.

    Returns the final generated sequence (plain string, no FASTA header).
    """
    if mask_start < 0:
        raise ValueError(f"mask_start must be >= 0, got {mask_start}")
    if mask_end <= mask_start:
        raise ValueError(f"mask_end ({mask_end}) must be > mask_start ({mask_start})")

    device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    if str(device) != device:
        logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load tokenizer and model
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer and model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.eval()
    model.to(device)

    # ------------------------------------------------------------------
    # 2. Extend sequence if mask region goes beyond the natural end
    # ------------------------------------------------------------------
    natural_len = len(sequence)
    if mask_end > natural_len:
        n_extra = mask_end - natural_len
        logger.info(
            f"mask_end ({mask_end}) > sequence length ({natural_len}): "
            f"appending {n_extra} mask tokens to extend sequence"
        )
        sequence = sequence + tokenizer.mask_token * n_extra

    # ------------------------------------------------------------------
    # 3. Tokenize and apply mask to [mask_start, mask_end)
    # ------------------------------------------------------------------
    # Replace the target region with mask tokens
    masked_seq = (
        sequence[:mask_start]
        + tokenizer.mask_token * (mask_end - mask_start)
        + sequence[mask_end:]
    )

    inputs = tokenizer(masked_seq, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']  # (1, seq_len+2) — BOS + tokens + EOS

    logger.info(
        f"Sequence length (after extension): {len(sequence)}, "
        f"masked region: [{mask_start}, {mask_end}), "
        f"mask length: {mask_end - mask_start}"
    )

    # ------------------------------------------------------------------
    # 4. Identify masked token positions in the tokenized tensor
    #    ESMplusplus adds a BOS token at index 0, so residue i → token i+1
    # ------------------------------------------------------------------
    mask_token_id = tokenizer.mask_token_id
    # Verify: positions that are actually mask tokens in the input
    mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0].tolist()

    if not mask_positions:
        raise RuntimeError("No mask tokens found in the tokenized input — "
                           "check that the tokenizer's mask_token is correctly set.")

    logger.info(f"Found {len(mask_positions)} mask token positions in token sequence")

    # ------------------------------------------------------------------
    # 5. Back-project CAV to full hidden-state space
    # ------------------------------------------------------------------
    cav_fullspace = load_and_backproject_cav(cav_dir, version=version)
    cav_tensor = torch.tensor(cav_fullspace, dtype=model.dtype).to(device)

    # ------------------------------------------------------------------
    # 6. Register forward hooks on all transformer layers
    # ------------------------------------------------------------------
    handles = []

    # Try the standard ESM HuggingFace path first: model.esm.encoder.layer
    # Fall back to model.encoder.layer for other architectures
    layer_list = None
    for attr_path in ('esm.encoder.layer', 'encoder.layer', 'bert.encoder.layer'):
        obj = model
        try:
            for attr in attr_path.split('.'):
                obj = getattr(obj, attr)
            layer_list = obj
            logger.info(f"Found transformer layers at model.{attr_path} "
                        f"({len(layer_list)} layers)")
            break
        except AttributeError:
            continue

    if layer_list is None:
        raise RuntimeError(
            "Could not locate transformer layer list. "
            "Expected model.esm.encoder.layer or model.encoder.layer. "
            "Please inspect the model architecture and adapt the hook registration."
        )

    hook_fn = make_layer_hook(cav_tensor, mask_positions, cav_weight)
    for layer in layer_list:
        handles.append(layer.register_forward_hook(hook_fn))

    logger.info(
        f"Registered CAV hooks on {len(handles)} layers "
        f"(cav_weight={cav_weight})"
    )

    # ------------------------------------------------------------------
    # 7. Single forward pass under no_grad
    # ------------------------------------------------------------------
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)
    finally:
        # Always remove hooks, even if forward pass fails
        for h in handles:
            h.remove()
        logger.info("Forward hooks removed")

    # ------------------------------------------------------------------
    # 8. Decode: argmax at each masked position
    # ------------------------------------------------------------------
    predicted_ids = logits[0].argmax(dim=-1)  # (seq_len,)

    # Start from original token ids; fill in predictions at masked positions
    output_ids = input_ids[0].clone()
    for pos in mask_positions:
        output_ids[pos] = predicted_ids[pos]

    generated_tokens = tokenizer.decode(output_ids, skip_special_tokens=True)
    # tokenizer.decode may add spaces for some tokenizers; strip them
    generated_seq = generated_tokens.replace(' ', '')

    return generated_seq


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CAV-guided masked generation with ESMplusplus (HuggingFace).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--sequence', required=True,
        help='Input protein sequence as a string, or path to a FASTA file.'
    )
    parser.add_argument(
        '--model', required=True,
        help='Path to ESMplusplus model directory (HuggingFace format).'
    )
    parser.add_argument(
        '--cav-dir', required=True,
        help='Directory containing trained CAV artifacts (concept_v1.npy, scaler_v1.pkl, ...).'
    )
    parser.add_argument(
        '--mask-start', type=int, required=True,
        help='0-based start of region to generate (inclusive).'
    )
    parser.add_argument(
        '--mask-end', type=int, required=True,
        help='0-based end of region to generate (exclusive). '
             'May exceed the sequence length to append novel residues.'
    )
    parser.add_argument(
        '--cav-weight', type=float, default=0.1,
        help='Scalar multiplier for CAV addition to hidden states (default: 0.1). '
             'Use 0.0 for unsteered baseline.'
    )
    parser.add_argument(
        '--device', default='cuda',
        help='Device to run on: cuda or cpu (default: cuda, falls back to cpu if unavailable).'
    )
    parser.add_argument(
        '--out', default=None,
        help='Output FASTA path. Defaults to stdout if not provided.'
    )
    parser.add_argument(
        '--version', default='v1',
        help='CAV artifact version (default: v1).'
    )
    args = parser.parse_args()

    # Read input sequence
    sequence, seq_name = read_sequence(args.sequence)
    logger.info(f"Input sequence: {seq_name} ({len(sequence)} aa)")

    # Run generation
    generated = generate(
        sequence=sequence,
        model_path=args.model,
        cav_dir=args.cav_dir,
        mask_start=args.mask_start,
        mask_end=args.mask_end,
        cav_weight=args.cav_weight,
        device=args.device,
        version=args.version,
    )

    # Build FASTA header
    header = (
        f">{seq_name} "
        f"mask={args.mask_start}-{args.mask_end} "
        f"cav_weight={args.cav_weight} "
        f"cav_dir={Path(args.cav_dir).name}"
    )
    fasta_output = f"{header}\n{generated}\n"

    # Write output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(fasta_output)
        logger.info(f"Generated sequence written to {out_path}")
    else:
        sys.stdout.write(fasta_output)

    # Log a summary of the generated region
    mask_len = args.mask_end - args.mask_start
    if len(generated) >= args.mask_end:
        generated_region = generated[args.mask_start:args.mask_end]
        logger.info(f"Generated region [{args.mask_start}:{args.mask_end}]: {generated_region}")
    logger.info(f"Total output sequence length: {len(generated)}")


if __name__ == '__main__':
    main()
