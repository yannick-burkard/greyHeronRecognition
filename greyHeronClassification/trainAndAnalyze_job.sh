#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/project/eawag/p05001/civil_service/greyHeronClassification/logs/job_outputs/slurm-%j.out
#SBATCH --gpus-per-node=6 #6
#SBATCH --gres=gpumem:24g #24g

#usage of this script: sbatch traindAndAnalyze_job.sh

: ${n_epochs:=1}
: ${batch_size:=32}
: ${learning_rate:=1e-5}
: ${weight_decay:=1e-12}
: ${dropout_prob:=0.2}
: ${seed:=9437}
: ${model_name:='mobilenet'}
: ${which_weights:='all'} #'last' or 'all': unfrozen weights from last or all layers
: ${n_last_im:='none'} #takes number of last images for background removal; if 'none', then original images are taken
: ${day_night:='day'}
: ${im_size:=896}
: ${augs:='colorjitter'} #data augmentations, example is 'colorjitter'; needs to be separated with comma without spaces in between
: ${resample_trn:='undersample'} #resample method for training: 'none', 'undersample', 'oversample_smote', 'oversample_dyn', 'oversample_dyn_ratio_50', 'oversample_naive', 'log_oversample', 'log_oversample_2', 'no_resample'
: ${n_cams_regroup:=12} #number of regrouped cameras during log oversampling
: ${ls_cams:='SBU4'} #filtered cameras
: ${val_full:=0} #0 (false) or 1 (true); evaluate model on full validation set after training
: ${trn_val:=0} #0 (false) or 1 (true); train with both training and validation data merged
: ${which_val:='val'} #'val', 'tst' or 'none'; which dataset to use for validation
: ${split:='seasonal'} #'chronological' (first) or 'seasonal' (second); splitting method used

timestamp=$(date +"%Y%m%d_%H%M%S")

python trainAndAnalyze.py \
    -job_id=$SLURM_JOB_ID \
    -timestamp=$timestamp \
    -n_epochs=$n_epochs \
    -batch_size=$batch_size \
    -learning_rate=$learning_rate \
    -weight_decay=$weight_decay \
    -dropout_prob=$dropout_prob \
    -seed=$seed \
    -model_name=$model_name \
    -which_weights=$which_weights \
    -n_last_im=$n_last_im \
    -day_night=$day_night \
    -im_size=$im_size \
    -augs=$augs \
    -resample_trn=$resample_trn \
    -n_cams_regroup=$n_cams_regroup \
    -ls_cams=$ls_cams \
    -val_full=$val_full \
    -trn_val=$trn_val \
    -which_val=$which_val \
    -split=$split


echo "Job submitted with ID: $SLURM_JOB_ID"



