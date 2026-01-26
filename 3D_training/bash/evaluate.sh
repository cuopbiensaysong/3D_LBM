#!/bin/bash
#SBATCH --job-name=lbm       # Job name
#SBATCH --output=slurm/evaluate_out.txt      # Output file
#SBATCH --error=slurm/evaluate_error.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=80G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

cd evaluate_classify

python evaluate.py --inference_dir /home/user01/aiotlab/htien/3D_LBM/3D_training/results/inferences/3D_LBM_uniform_noise0.01_step200 --ckpt_path ./Classification-models/cyclic_scratch_DenseNet121_train+val.pth

python evaluate.py --inference_dir /home/user01/aiotlab/htien/3D_LBM/3D_training/results/inferences/3D_LBM_uniform_noise0.005_step20 --ckpt_path ./Classification-models/cyclic_scratch_DenseNet121_train+val.pth


python evaluate.py --inference_dir /home/user01/aiotlab/htien/3D_LBM/3D_training/results/inferences/3D_LBM_uniform_noise0.005_step200 --ckpt_path ./Classification-models/cyclic_scratch_DenseNet121_train+val.pth


python evaluate.py --inference_dir /home/user01/aiotlab/htien/3D_LBM/3D_training/results/inferences/3D_LBM_uniform_noise0.01_step20 --ckpt_path ./Classification-models/cyclic_scratch_DenseNet121_train+val.pth


