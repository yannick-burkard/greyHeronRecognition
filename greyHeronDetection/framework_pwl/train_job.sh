#!/bin/bash

#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=16 ###8                 # Number of tasks (cores)
#SBATCH --mem-per-cpu=16g ###16g ###request more if necessary!
#SBATCH --time=4:00:00             # Time limit (hh:mm:ss)
#SBATCH --output=/cluster/project/eawag/p05001/civil_service/greyHeronDetection/framework_pwl/logs/job_outputs/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --gres=gpumem:24g

python train_wDataResample.py

