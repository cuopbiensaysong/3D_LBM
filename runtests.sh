#!/bin/bash
#SBATCH --job-name=lbm       # Job name
#SBATCH --output=nvidia1.txt      # Output file
#SBATCH --error=e-nividi1a.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=80G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node


nvidia-smi