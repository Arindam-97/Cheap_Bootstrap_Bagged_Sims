#!/bin/bash

#SBATCH --account=ieor_lam
#SBATCH --job-name=Stoch_lin_prog_inputs_6
#SBATCH --array=1-12
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=800M
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module load anaconda
export CONDA_PKGS_DIRS=$HOME/.conda/pkgs
eval "$(conda shell.bash hook)"
conda activate cheapboot

INPUT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" inputs_6.txt)

echo "Running job with input: $INPUT"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

python run_main_DT.py --name "$INPUT"