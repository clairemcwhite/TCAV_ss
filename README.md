# Protein Motif Detection using Concept Activation Vectors (TCAV)

## Overview

We present an automated approach for identifying and annotating motifs and domains in protein sequences, using pretrained Protein Language Models (PLMs) and Concept Activation Vectors (CAVs), adapted from interpretability research in computer vision. We treat motifs as conceptual entities and represent them through learned CAVs in PLM embedding space by training simple linear classifiers to distinguish motif-containing from non-motif sequences. 


## Key Features

- **Interpretable motif detection** using CAVs in protein language model embedding space
- **Multiple motif detection** in single sequences with precise localization
- **Layer-specific analysis** to identify optimal representation layers
- **Flexible PLM support** (ESM2, ESM-C)

## Repository Structure

```
├── esm2_tcav/              # Core TCAV implementation
│   ├── src/                # Source code for embedding, training, evaluation
│   ├── config.yaml         # Model and training configuration
│   └── models/             # Model registry and configurations
├── tcav_data/              # Training data (positive/negative examples)
├── test_data/              # Test sequences for evaluation
├── test_data_small/        # Small test set for quick validation
├── Fasta Files/            # Input protein sequences
├── Outputs/                # Detection results
├── batch_train_cavs.py     # Train CAVs for multiple motifs
├── batch_detect_all_motifs.py  # Detect motifs in protein sequences
└── run_test_detection_onthefly.py  # Evaluate on test dataset
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd TCAV

## Usage

### 1. Training CAVs

Train Concept Activation Vectors for protein motifs:

```bash
python batch_train_cavs.py \
  --data-dir ./tcav_data \
  --config esm2_tcav/config.yaml \
  --output-dir ./tcav_outputs_esmplusplus_all \
  --model ESMplusplus_large \
  --layers 20 25 30 35 \
  --batch-size 8
```

**Options:**
- `--data-dir`: Directory containing positive/negative training examples
- `--model`: PLM to use (ESMplusplus_large, ESM2_t33_650M_UR50D, ESMC_600M, etc.)
- `--layers`: Which transformer layers to extract representations from
- `--no-save-embeddings`: Don't save intermediate embeddings (saves disk space)

### 2. Detecting Motifs in Sequences

Detect motifs in your protein sequences:

```bash
python batch_detect_all_motifs.py \
  --fasta my_protein.fasta \
  --tcav-dir ./tcav_outputs_esmplusplus_all \
  --data-dir ./tcav_data \
  --model-name ESMplusplus_large \
  --layers 25 \
  --rank-by-layer 25 \
  --rank-by score \
  --topk 100 \
  --device cuda
```

**Options:**
- `--fasta`: Input FASTA file with protein sequences
- `--tcav-dir`: Directory containing trained CAVs
- `--motifs`: Specific motifs to detect (default: all available)
- `--layers`: Layers to use for detection
- `--rank-by-layer`: Which layer to use for ranking results
- `--rank-by`: Ranking method (`score` or `position`)
- `--topk`: Number of top predictions to return
- `--window-scores-dir`: Directory to save detailed window scores

**Example: Detect specific motifs**
```bash
python batch_detect_all_motifs.py \
  --fasta Q9NHV9.fasta \
  --motifs PF00017 PF00018 PF00130 PF00621 \
  --model-name ESMplusplus_large \
  --tcav-dir ./tcav_outputs_esmplusplus_all \
  --layers 25 \
  --rank-by-layer 25 \
  --rank-by score \
  --device cuda
```

### 3. Evaluating on Test Dataset

Evaluate detection performance on benchmark data:

```bash
python run_test_detection_onthefly.py \
  --test-data-dir ./test_data_small \
  --tcav-dir ./tcav_outputs_esmplusplus_all \
  --data-dir ./tcav_data \
  --model-name ESMplusplus_large \
  --layers 25 \
  --rank-by score \
  --device cuda
```

## Data Format

### Training Data (`tcav_data/`)
Each motif requires:
- `{motif_id}_pos.fasta`: Positive examples containing the motif
- `{motif_id}_pos.jsonl`: Annotations with motif locations
- `{motif_id}_neg.fasta`: Negative examples without the motif
- `{motif_id}_neg.jsonl`: Negative example metadata

### Test Data
- `{protein_id}.fasta`: Test protein sequences
- `{protein_id}.jsonl`: Ground truth motif annotations

## Models Supported

- **ESM++** (ESMplusplus_large) - Recommended for best performance
- **ESM2** (ESM2_t33_650M_UR50D, ESM2_t36_3B_UR50D)
- **ESM-C** (ESMC_300M, ESMC_600M)

Configure models in `esm2_tcav/models/model_registry.yaml`

## Output

Detection results include:
- Top-k motif predictions per sequence
- Confidence scores based on CAV inner products
- Precise sequence position ranges
- Layer-wise scores for analysis

## Citation

If you use this code, please cite our paper:
```
[Citation information to be added]
```

