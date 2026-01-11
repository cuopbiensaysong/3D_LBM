#!/bin/bash
#SBATCH --job-name=lbm       # Job name
#SBATCH --output=slurm/out.txt      # Output file
#SBATCH --error=slurm/error_m3d.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=80G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

cd /home/user01/aiotlab/htien/3D_LBM/3D_training
python lbm_trainer.py 