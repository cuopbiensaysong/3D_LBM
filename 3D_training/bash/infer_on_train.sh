#!/bin/bash
#SBATCH --job-name=lbm       # Job name
#SBATCH --output=slurm/infer_on_train_out.txt      # Output file
#SBATCH --error=slurm/infer_on_train_error.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=80G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node


python infer_on_train.py --output_dir ./results/infer_on_train/3D_LBM_uniform_noise0.01_step200 --num_inference_steps 200 --ckpt_path ./results/3D_LBM_uniform_noise0.01/best_checkpoint.pth --config_path ./results/3D_LBM_uniform_noise0.01/config.yaml --num_outputs_per_sample 4
