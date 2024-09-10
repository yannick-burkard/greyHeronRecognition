#!/bin/bash

#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=8 ###8                 # Number of tasks (cores)
#SBATCH --mem-per-cpu=4g ###16g ###request more if necessary!
#SBATCH --time=4:00:00             # Time limit (hh:mm:ss)
#SBATCH --output=/cluster/project/eawag/p05001/civil_service/greyHeronClassification/logs/job_outputs/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11g


python evaluate.py