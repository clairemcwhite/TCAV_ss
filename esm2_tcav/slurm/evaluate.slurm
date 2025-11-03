#!/bin/bash
#SBATCH --job-name=cav_eval
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# TCAV Evaluation
# CPU-only (metrics computation + visualization)

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
source /groups/clairemcwhite/envs/core_pkgs/bin/activate

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run evaluation
echo ""
echo "Evaluating CAVs..."
python scripts/run_evaluate.py --config config.yaml

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="


