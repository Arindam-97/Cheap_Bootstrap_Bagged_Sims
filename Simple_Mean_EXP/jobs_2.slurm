#!/bin/bash

#SBATCH --account=ieor_lam
#SBATCH --job-name=simple_mean_inputs_2
#SBATCH --array=1-12
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err


module load anaconda
export CONDA_PKGS_DIRS=$HOME/.conda/pkgs
eval "$(conda shell.bash hook)"
conda activate cheapboot
# Read the input file for the current array task

INPUT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" inputs_2.txt)

# Execute your command with the input
echo "Running job with input: $INPUT"
# Your actual command goes here, for example:
python run_simple_mean.py --name "$INPUT"
