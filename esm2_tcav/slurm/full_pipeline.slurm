#!/bin/bash
#SBATCH --job-name=tcav_pipeline
#SBATCH --output=outputs/logs/pipeline_%j.out
#SBATCH --error=outputs/logs/pipeline_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# Full TCAV Pipeline: Embed → Train → Evaluate
# Requires GPU for embedding extraction

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
# Option A: Shared environment (if fair-esm is installed there)
# source /groups/clairemcwhite/envs/core_pkgs/bin/activate

# Option B: Your own environment (recommended)
conda activate tcav

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run full pipeline
echo ""
echo "Running full TCAV pipeline..."
python scripts/run_pipeline.py --config config.yaml

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="

