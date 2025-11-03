#!/bin/bash
#SBATCH --job-name=cav_train
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# TCAV CAV Training
# CPU-only (logistic regression + PCA)

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
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run CAV training
echo ""
echo "Training CAVs..."
python scripts/run_train_cav.py --config config.yaml

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="


