#!/bin/bash
#
#SBATCH --account=ieor_lam
#SBATCH --job-name=simple_mean_inputs_1
#SBATCH --array=1-2  # 12 jobs in the array
#SBATCH --time=00-02:00  # Adjust as needed
#SBATCH -c2
#SBATCH --mem-per-cpu=5G         # Adjust as needed
# Load necessary modules
module load anaconda
# Read the input file for the current array task
INPUT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" inputs_test.txt)

# Execute your command with the input
echo "Running job with input: $INPUT"
# Your actual command goes here, for example:
python run_simple_mean.py --name "$INPUT"
