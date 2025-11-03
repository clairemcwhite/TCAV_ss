#!/bin/bash
#SBATCH --job-name=esm2_embed
#SBATCH --output=outputs/logs/embed_%j.out
#SBATCH --error=outputs/logs/embed_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# TCAV ESM2 Embedding Extraction
# Requires GPU for ESM2 model inference

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
source /groups/clairemcwhite/envs/core_pkgs/bin/activate
# Or if it's a conda env: conda activate /groups/clairemcwhite/envs/core_pkgs

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run embedding extraction
echo ""
echo "Running embedding extraction..."
python scripts/run_embed.py --config config.yaml

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="


