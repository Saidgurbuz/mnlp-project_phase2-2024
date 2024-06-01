#!/bin/bash -l
#SBATCH --chdir /scratch/izar/krsteski
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 64G
#SBATCH --time 15:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

cd /scratch/izar/krsteski/lm-evaluation-harness

# Load the necessary modules
module load gcc/11.3.0
module load cuda/11.8.0

# Activate env, this should be set-up before
source ~/venvs/course_py-3.10/bin/activate

# Run
lm_eval --model hf     --model_args pretrained=microsoft/Phi-3-mini-4k-instruct,peft=StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL,trust_remote_code=True     --tasks hellaswag     --device cuda:0