#!/bin/bash
#SBATCH --job-name=lbm       # Job name
#SBATCH --output=slurm/out.txt      # Output file
#SBATCH --error=slurm/error_m3d.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=80G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node


python inference1.py --test_csv_path /home/user01/aiotlab/htien/5_folds_split_3D/fold_1_test.csv --output_dir /home/user01/aiotlab/htien/3D_LBM/3D_training/results/inferences/3D_LBM_uniform_noise0.005_step20 --num_inference_steps 20 --ckpt_path /home/user01/aiotlab/htien/3D_LBM/3D_training/results/3D_LBM_uniform_noise0.005/best_checkpoint.pth --config_path /home/user01/aiotlab/htien/3D_LBM/3D_training/results/3D_LBM_uniform_noise0.005/config.yaml --save_img_output True --num_outputs_per_sample 30 --compute_metrics True

