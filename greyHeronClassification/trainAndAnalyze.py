from train import trainAndEvaluate
from analysis.plotLearningCurves import plot_learning_curves
import sys
import os
import argparse
from utils.train_utils import save_dictionary

parser = argparse.ArgumentParser(description='training configurations')
parser.add_argument('-n_epochs', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-learning_rate', type=float, default=1e-5)
parser.add_argument('-weight_decay', type=float, default=1e-12)
parser.add_argument('-dropout_prob', type=float, default=0.2)
parser.add_argument('-seed', type=int, default=9437)
parser.add_argument('-model_name', type=str, default='mobilenet')
parser.add_argument('-which_weights', type=str, default='all')
parser.add_argument('-job_id', type=str, default='xxxxxxxx')
parser.add_argument('-timestamp', type=str, default='xxxxxxxx_xxxxxx')
parser.add_argument('-n_last_im', type=str, default='none')
parser.add_argument('-day_night', type=str, default='day')
parser.add_argument('-im_size', type=str, default=224)
parser.add_argument('-augs', type=str, default=[])
parser.add_argument('-resample_trn', type=str, default='undersample')
parser.add_argument('-n_cams_regroup', type=int, default=0)
parser.add_argument('-ls_cams', type=str, default='SBU4')
parser.add_argument('-val_full', type=int, default=0)
parser.add_argument('-trn_val', type=int, default=0)
parser.add_argument('-which_val', type=str, default='val')
parser.add_argument('-split', type=str, default='seasonal')
args = parser.parse_args()

if 'SLURM_JOB_ID' in os.environ:
    job_id = os.environ['SLURM_JOB_ID']
    print(f"Current SLURM Job ID: {job_id}")
else:
    job_id = 'none'
    print("Not running within a SLURM job.")

if 'SLURM_NTASKS' in os.environ:
    ntasks = int(os.environ['SLURM_NTASKS'])
    print(f"Number of tasks (ntasks): {ntasks}")
else:
    ntasks = 1
    print("SLURM_NTASKS environment variable set. ntasks set to 1.")

ls_cams_all = [
        'GBU1','GBU2','GBU3','GBU4',
        'KBU1','KBU2','KBU3','KBU4',
        'NEN1','NEN2','NEN3','NEN4',
        'PSU1','PSU2','PSU3',
        'SBU1','SBU2','SBU3',
        'SBU4',
        'SGN1','SGN2','SGN3','SGN4'
        ]
ls_cams_flag = args.ls_cams
if ls_cams_flag=='SBU4':
    ls_cams=['SBU4']
if ls_cams_flag=='all':
    ls_cams=ls_cams_all

config = {
        "job_output_ID": args.job_id,
        "time_stamp": args.timestamp,
        "torch_seed": args.seed,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout_prob": args.dropout_prob,
        "which_weights": args.which_weights,
        "num_workers": int(ntasks),
        "n_last_im": args.n_last_im,
        "day_night": args.day_night,
        "image_size": int(args.im_size),
        "augs": list(str(args.augs).split(',')),
        "resample_trn": args.resample_trn,
        "n_cams_regroup": args.n_cams_regroup,
        "ls_cams": ls_cams,
        "val_full": args.val_full,
        "trn_val": args.trn_val,
        "which_val": args.which_val,
        "split": args.split,
        #default configs, adjust if needed
        "parent_dir": '/cluster/project/eawag/p05001/repos/greyHeronRecognition/', #parent directory for Euler cluster
        "pretrained_network": 'mobilenet',
        "num_classes": 2
        }

print('=========================================================')
trainAndEvaluate(config)
print('training done')
plot_learning_curves(parent_dir=config['parent_dir'],time_stamp=config['time_stamp'])
print('plotting done')
print('=========================================================')



