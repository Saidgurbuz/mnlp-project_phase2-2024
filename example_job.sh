#!/bin/bash -l
#SBATCH --chdir /scratch/izar/krsteski
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 64G
#SBATCH --time 8:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

cd /scratch/izar/krsteski/project-m2-2024-chatterbox

# Load the necessary modules
module load gcc/11.3.0
module load cuda/11.8.0

# Activate env, this should be set-up before
source ~/venvs/course_py-3.10/bin/activate


# Run
python3 evaluator.py
