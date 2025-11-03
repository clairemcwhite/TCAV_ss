#!/bin/bash
#SBATCH --job-name=znf_detect
#SBATCH --output=outputs/logs/detect_%j.out
#SBATCH --error=outputs/logs/detect_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Zinc Finger Detection
# Requires GPU for sliding window inference

# Usage: sbatch slurm/detect.slurm <input_fasta>

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Check if input file provided
if [ -z "$1" ]; then
    echo "ERROR: No input FASTA file provided"
    echo "Usage: sbatch slurm/detect.slurm <input.fasta>"
    exit 1
fi

INPUT_FASTA=$1

# Activate conda environment
source /groups/clairemcwhite/envs/core_pkgs/bin/activate

# Print environment info
echo "Python: $(which python)"
echo "Input FASTA: $INPUT_FASTA"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run detection
echo ""
echo "Running zinc finger detection..."
python scripts/run_detect.py --config config.yaml --input "$INPUT_FASTA"

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="


