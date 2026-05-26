#!/usr/bin/env bash
# Train a CAV from a span file of positive protein IDs.
# 
# Usage: bash train_cav_from_span.sh <SPAN_FILE> [OPTIONS]
# 
# Example: bash train_cav_from_span.sh examples/RLK5_ARATH.fasta.span
#
# Options:
#   --few-exemplars    Use 0 CV folds (for small datasets)
#   --only-download    Only retrieve FASTA, don't train
#   --scaler-pkl PATH  Path to pre-fitted scaler
#
# Configuration:
#   Create config.local.yaml with your paths, or edit config.yaml
#   Required setting: model (path to ESM++/ESM2 model)

SPAN_FILE=$1
FEW_EXEMPLARS=0
ONLY_DOWNLOAD=0
SCALER_PKL=""
args=("$@")
for i in "${!args[@]}"; do
    [ "${args[$i]}" = "--few-exemplars" ] && FEW_EXEMPLARS=1
    [ "${args[$i]}" = "--only-download" ] && ONLY_DOWNLOAD=1
    if [ "${args[$i]}" = "--scaler-pkl" ]; then
        SCALER_PKL="${args[$((i+1))]}"
    fi
done
CV_FOLDS=$([ "$FEW_EXEMPLARS" -eq 1 ] && echo 0 || echo 5)

# -------------
### Load Configuration
# -------------
set -euo pipefail

# Detect repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
tcav_dir="$SCRIPT_DIR"

# Load config (prefer config.local.yaml if it exists)
CONFIG_FILE="${tcav_dir}/config.yaml"
if [ -f "${tcav_dir}/config.local.yaml" ]; then
    CONFIG_FILE="${tcav_dir}/config.local.yaml"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found (looked for config.yaml or config.local.yaml)"
    exit 1
fi

# Simple YAML parser for our needs (reads key: value pairs)
function get_config() {
    local key=$1
    grep "^$key:" "$CONFIG_FILE" | sed 's/^[^:]*: *//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/"
}

# Load paths from config
model=$(get_config "model")
ref_neg=$(get_config "reference_neg")
ref_pca=$(get_config "reference_pca")

# Validate required config
if [ -z "$model" ] || [ "$model" = "/path/to/your/ESMplusplus_large" ]; then
    echo "Error: Please set 'model' path in $CONFIG_FILE"
    exit 1
fi

# Embedding script is now in the repo
embed_script="${tcav_dir}/hf_embed_new.py"

# Load embedding parameters
layer=$(get_config "layer" || echo "-11")
max_length=$(get_config "max_length" || echo "2048")
batch_size=$(get_config "batch_size" || echo "1")

# Derived paths
FASTA_FILE=${SPAN_FILE}.fasta
PKL_FILE=${FASTA_FILE}.pkl
POS_NPY=${SPAN_FILE}.pos.npy
CAV_DIR=${SPAN_FILE%.span}_cav

# -------------
### Step 1: Retrieve FASTA for positive training proteins
# -------------
echo "Step 1: Retrieving FASTA sequences..."
if [ -f "$FASTA_FILE" ]; then
    echo "  Skipping: $FASTA_FILE already exists"
else
    echo "  Note: If FASTA doesn't exist, you may need to create it from your data"
    echo "  For JSONL test data, use: python scripts/jsonl_to_span.py <jsonl_file> --format both"
    if [ ! -f "$FASTA_FILE" ]; then
        echo "  Error: FASTA file not found: $FASTA_FILE"
        exit 1
    fi
fi

[ "$ONLY_DOWNLOAD" -eq 1 ] && { echo "Done (--only-download)."; exit 0; }

# -------------
### Step 2: Embed FASTA
# -------------
echo "Step 2: Embedding sequences..."
if [ -f "$PKL_FILE" ]; then
    echo "  Skipping: $PKL_FILE already exists"
else
    echo "  Using model: $model"
    echo "  Layer: $layer, Max length: $max_length, Batch size: $batch_size"
    python $embed_script \
        -f $FASTA_FILE \
        -o $PKL_FILE \
        --get_aa_embeddings \
        --get_sequence_embedding \
        --strat mean \
        -l $layer \
        -m $model \
        -b $batch_size \
        --max_length $max_length
fi

# -------------
### Step 3: Pool embeddings by span → positive embedding matrix
# -------------
echo "Step 3: Pooling embeddings by span..."
python $tcav_dir/scripts/prepare_embeddings.py \
    --pkl  $PKL_FILE \
    --info ${PKL_FILE}.seqnames \
    --spans $SPAN_FILE \
    --out  $POS_NPY

# -------------
### Step 4: Train CAV
# -------------
echo "Step 4: Training CAV..."
echo "  Using reference negatives: $ref_neg"
echo "  CV folds: $CV_FOLDS"
python $tcav_dir/scripts/train_cav_from_embeddings.py \
    --pos  $POS_NPY \
    --neg  $ref_neg \
    --out  $CAV_DIR \
    --cv-folds $CV_FOLDS \
    ${SCALER_PKL:+--pca-pkl "$SCALER_PKL"}

# -------------
### Step 5: Clean up intermediate embedding files
# -------------
echo "Step 5: Cleaning up intermediate files..."
rm -f $PKL_FILE ${PKL_FILE}.seqnames $POS_NPY
rm -f ${CAV_DIR}/pos.npy ${CAV_DIR}/neg.npy

echo ""
echo "✓ CAV training complete!"
echo "  Output directory: $CAV_DIR"
echo "  CAV file: ${CAV_DIR}/concept_cav_v1.npy"
