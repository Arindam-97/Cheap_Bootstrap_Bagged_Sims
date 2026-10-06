#!/bin/bash

#SBATCH --account=ieor_lam
#SBATCH --job-name=simple_mean_inputs_3
#SBATCH --array=1-4
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=3G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module load anaconda
export CONDA_PKGS_DIRS=$HOME/.conda/pkgs
eval "$(conda shell.bash hook)"
conda activate cheapboot

INPUT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" inputs_3.txt)

echo "Running job with input: $INPUT"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

python run_simple_mean.py --name "$INPUT"