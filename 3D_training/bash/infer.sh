#!/bin/bash
#SBATCH --job-name=lbm       # Job name
#SBATCH --output=slurm/infer_out_1.txt      # Output file
#SBATCH --error=slurm/infer_error_1.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=80G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

python inference1.py --test_csv_path /home/user01/aiotlab/htien/5_folds_split_3D/fold_1_test.csv --output_dir ./results/inferences/3D_uniform_noise0.02_step20 --num_inference_steps 20 --ckpt_path ./results/3D_uniform_noise0.02/best_checkpoint.pth --config_path ./results/3D_uniform_noise0.02/config.yaml --save_img_output True --num_outputs_per_sample 30 --compute_metrics True


