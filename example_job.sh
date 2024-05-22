#!/bin/bash -l
#SBATCH --chdir /scratch/izar/USERNAME
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 64G
#SBATCH --time 15:00:00 - CHANGE THIS AS NEEDED
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552

# Pull from git repository if needed

cd /scratch/izar/USERNAME/project-m2-2024-chatterbox

# Load the necessary modules
module load gcc/11.3.0
module load cuda/11.8.0

# Activate env, this should be set-up before
source ~/venvs/course_py-3.10/bin/activate


# Run
python3 training/train_dpo.py
