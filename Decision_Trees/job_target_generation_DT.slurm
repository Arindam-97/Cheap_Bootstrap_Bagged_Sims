#!/bin/bash

#SBATCH --account=ieor_lam
#SBATCH --job-name=simple_target_gen_SLP
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module load anaconda
export CONDA_PKGS_DIRS=$HOME/.conda/pkgs
eval "$(conda shell.bash hook)"
conda activate cheapboot

# Execute your command with the input
# Your actual command goes here, for example:

python run_target_generation_DT.py
