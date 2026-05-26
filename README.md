# Protein Motif Detection using Concept Activation Vectors (CAV)

## Overview

We present an automated approach for identifying and annotating motifs and domains in protein sequences, using pretrained Protein Language Models (PLMs) and Concept Activation Vectors (CAVs), adapted from interpretability research in computer vision. We treat motifs as conceptual entities and represent them through learned CAVs in PLM embedding space by training simple linear classifiers to distinguish motif-containing from non-motif sequences.

## Key Features

- **Interpretable motif detection** using CAVs in protein language model embedding space
- **Multiple motif detection** in single sequences with precise localization
- **Layer-specific analysis** to identify optimal representation layers
- **Flexible PLM support** (ESM++, ESM2, ESM-C)

## Repository Structure

```
├── hf_embed_new.py          # Embedding script (included in repo)
├── train_cav_from_span.sh   # Main training wrapper script
├── config.yaml              # Configuration template (copy to config.local.yaml)
├── tcav/                    # Core TCAV implementation
│   └── src/                 # Source modules (attribution, detection, training, evaluation)
├── scripts/                 # Pipeline scripts
│   ├── prepare_embeddings.py        # Pool per-residue embeddings by span
│   ├── train_cav_from_embeddings.py # Train a CAV from positive/negative matrices
│   ├── query_proteins_by_cav.py     # Rank/scan a FASTA against a trained CAV
│   ├── jsonl_to_span.py             # Convert JSONL test data to span format
│   ├── evaluate_cav.py              # Evaluate CAV performance
│   ├── scan_sequence.py             # Sliding-window scan of a sequence
│   ├── run_attribution.py           # Attribution analysis
│   └── fit_global_pca.py            # Fit PCA on reference population
├── reference_population/    # Reference negative embeddings (see Setup)
├── examples/                # Example input files
├── test_data/               # Full benchmark dataset (60 Pfam motifs, JSONL format)
└── test_data_small/         # Small subset for quick validation
```

## Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd TCAV_ss
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate tcav_protein
```

### 2. Configure Paths

Copy the configuration template and edit with your paths:

```bash
cp config.yaml config.local.yaml
# Edit config.local.yaml and set your model path
```

**Required configuration:**
- `model`: Path to your ESM++, ESM2, or ESM-C model directory

**Example `config.local.yaml`:**
```yaml
model: /path/to/ESMplusplus_large
reference_neg: reference_population/neg_embeddings.npy
reference_pca: reference_population/global_pca.pkl

embedding:
  layer: -11
  max_length: 2048
  batch_size: 1
```

### 3. Build Reference Negative Population

The repository includes pre-computed reference negatives in `reference_population/`. If you need to create your own, see `reference_population/README.md` for instructions.

## Input Formats

### FASTA
Standard FASTA format. The accession in the header must match the accession used in the span file.

```
>sp|P47735|RLK5_ARATH Receptor-like protein kinase 5 ...
MLYCLILLLC...
```

### Span file (`.span`)
Tab-separated, one span per line:

```
<accession>	<start>	<end>
```

The accession must match the FASTA header (full `sp|...|...` form or just the bare ID). Start and end are 1-based, inclusive residue positions.

```
sp|P47735|RLK5_ARATH	404	591
```

### JSONL (test data format)

The `test_data/` directories contain JSONL files with ground truth annotations:

```json
{
  "accession": "P12345",
  "sequence": "MTKL...",
  "length": 388,
  "protein_name": "Example protein",
  "gene_name": "GENE1",
  "target_motif": "PF00001",
  "ground_truth_annotations": [
    {"motif_id": "PF00001", "start": 36, "end": 312, "name": "Example domain"}
  ]
}
```

Convert to FASTA/span format using:
```bash
python scripts/jsonl_to_span.py input.jsonl --format both
```

## Pipeline

The pipeline consists of five steps for each concept. The provided `train_cav_from_span.sh` script runs Steps 01–04 automatically.

### Step 01 — Embed input FASTA

Embed the protein(s) that define your concept. Both per-residue (AA) and sequence-level embeddings are needed for span pooling in Step 02.

```bash
python hf_embed_new.py \
    -f $INPUT_FASTA \
    -o ${INPUT_FASTA}.pkl \
    --get_aa_embeddings \
    --get_sequence_embedding \
    --strat mean \
    -l -11 \
    -m $MODEL_PATH \
    -b 1 \
    --max_length 2048
```

The `train_cav_from_span.sh` script handles this automatically using paths from your config file.

### Step 02 — Pool embeddings by span → positive matrix

Extract the region(s) defined in your span file and pool them into a positive embedding matrix.

```bash
python scripts/prepare_embeddings.py \
    --pkl   ${INPUT_FASTA}.pkl \
    --info  ${INPUT_FASTA}.pkl.seqnames \
    --spans $SPAN_FILE \
    --out   ${SPAN_FILE%.span}_pos.npy
```

### Step 03 — Train CAV

Train a linear classifier (CAV) separating the positive span embeddings from the reference negative population.

```bash
python scripts/train_cav_from_embeddings.py \
    --pos     ${SPAN_FILE%.span}_pos.npy \
    --neg     reference_population/neg_embeddings.npy \
    --out     $concept_dir \
    --cv-folds 5 \
    --pca-pkl reference_population/global_pca.pkl
```

### Step 04 — Embed search FASTA

Embed the sequences you want to rank. Only sequence-level embeddings are needed here.

```bash
python hf_embed_new.py \
    -f $SEARCH_FASTA \
    -o ${SEARCH_FASTA}.pkl \
    --get_sequence_embedding \
    --strat mean \
    -l -11 \
    -m $MODEL_PATH \
    -b 1 \
    --max_length 2048
```

> **Note:** For small search FASTAs you can skip pre-computing the pkl and pass `--embed-fasta` directly to the query script (Step 05); it will embed on the fly.

### Step 05 — Query search FASTA against CAV

Score and rank every sequence (or every sliding window) in the search set against the trained CAV.

```bash
python scripts/query_proteins_by_cav.py \
    --cav   $concept_dir \
    --pkl   ${SEARCH_FASTA}.pkl \
    --out   results.tsv \
    --k     -1 \
    --sliding-window \
    --spans $SPAN_FILE \
    --fasta $SEARCH_FASTA
```

**Options:**
- `--k -1`: Returns all results (or set to N for top-N)
- `--sliding-window`: Enable per-window scoring for motif localization
- `--spans` and `--fasta`: Annotate known positive regions in output

## Running the Example

The `examples/` directory contains a complete example for a receptor-like kinase domain:

| File | Description |
|------|-------------|
| `RLK5_ARATH.fasta` | Input protein sequence (RLK5, *A. thaliana*) |
| `RLK5_ARATH.fasta.span` | Span defining the kinase domain (residues 404–591) |

### Quick Start: Train CAV on Example

```bash
# Train CAV for kinase domain (runs Steps 01-04)
bash train_cav_from_span.sh examples/RLK5_ARATH.fasta.span
```

This will:
1. Embed the RLK5 protein sequence
2. Pool embeddings for the kinase domain region (residues 404–591)
3. Train a CAV against the reference negative population
4. Output results to `examples/RLK5_ARATH_cav/`

### Complete Workflow: Train and Query

Train the CAV and search for similar domains in a proteome:

```bash
# Step 1-4: Train CAV (automated)
bash train_cav_from_span.sh examples/RLK5_ARATH.fasta.span

# Step 5: Query against a search database
SEARCH_FASTA=path/to/your/proteome.fasta

# Embed search sequences
python hf_embed_new.py \
    -f $SEARCH_FASTA \
    -o ${SEARCH_FASTA}.pkl \
    --get_sequence_embedding \
    --strat mean \
    -l -11 \
    -m $(grep "^model:" config.local.yaml | cut -d' ' -f2) \
    -b 1 \
    --max_length 2048

# Query with the trained CAV
python scripts/query_proteins_by_cav.py \
    --cav examples/RLK5_ARATH_cav \
    --pkl ${SEARCH_FASTA}.pkl \
    --out kinase_predictions.tsv \
    --k -1 \
    --sliding-window
```

### Working with Test Data

The `test_data/` directory contains JSONL benchmark files. Convert them to usable formats:

```bash
# Convert single file to FASTA and span format
python scripts/jsonl_to_span.py test_data/PF00001/test_100.jsonl --format both

# This creates:
#   test_data/PF00001/test_100.span  (span annotations)
#   test_data/PF00001/test_100.fasta (sequences)

# Convert entire directory
python scripts/jsonl_to_span.py test_data/ --recursive --format both

# Then train a CAV
bash train_cav_from_span.sh test_data/PF00001/test_100.span
```

## Models Supported

- **ESM++** (`ESMplusplus_large`) — recommended for best performance
- **ESM2** (`ESM2_t33_650M_UR50D`, `ESM2_t36_3B_UR50D`)

Layer `-11` (25th transformer layer) is used throughout, following the arxiv paper.

## Citation

If you use this code, please cite our paper:
```
[Citation information to be added]
```
