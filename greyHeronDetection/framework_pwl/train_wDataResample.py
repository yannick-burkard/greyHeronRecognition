import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import pandas as pd
import time
import glob
import os
import json
import shutil
from datetime import datetime
from utils.train_utils import get_data_and_labels, undersample_data, save_dictionary, get_max_FCE, oversample_data, get_baseline_metrics, log_oversample_pos, get_num_gpus, get_num_tasks, log_oversample_pos_2
from analysis.plotLearningCurves import plot_learning_curves
import yaml
import subprocess
import shutil


"""
Script to train and validate model on specified dataset based on config dictionary containing following keys
    'lr0' (float): initial learning rate (YOLOv5 hyperparameter)
    'lrf' (float): final learning rate (YOLOv5 hyperparameter)
    'warmup_epochs' (int): number of warmup epochs (YOLOv5 hyperparameter)
    'time_stamp' (str): time stamp of training job
    'parent_dir' (str): parent directory
    'split' (str): first ('chronological') or second ('seasonal') split used
    'day_night' (str): day or night-time images, or both
    'n_last_im' (int): number of last images for background removal; if 'none', then original images are taken
    'which_val' (str): which dataset to use for validation ('trn', 'val', 'trn_val', 'tst')
    'conf_tsh_fixed' (float): fixed confidence threshold throughout training and validation; if set to 0, then confidence threshold is adjusted after every epoch yielding max F1-score , 
    'trn_val' (bool): train with both training and validation data merged
    'resample_trn' (str): resample method for training
    'n_cams_regroup' (int): number of regrouped cameras during log oversampling
    'ls_cams' (list): list of filtered cameras
    'epochs (int): number of training epochs
    'batch_size' (int): batch size
    'weight_decay' (float): weight decay for learning step
    'imgsz' (int): image resolution
    'optimizer' (str): specifies optimizer used for learning step ('SGD', 'Adam', 'AdamW')
    'freeze' (int): number of layers to freeze in YOLOv5 model
    'model' (str): name of pretrained model
    'mosaic_prob' (float): mosaic probability (YOLOv5 hyperparamter)
    'val_dataset' (bool): evaluate trained model on full validation set after training
    'val_megadetector' (bool): evaluate megadetector on validation set
    'reduced_dataset' (bool or int): train and evaluate on reduced dataset, if set to int n, takes n first elements before resampling, useful to make quick checks
    'workers' (int): number of workers used for data loading 
    'n_gpus' (int): number of gpus used
    'seed' (int): seed used (YOLOv5 hyperparameter)
"""


#define configurations

ls_cams_all = [
        'GBU1','GBU2','GBU3','GBU4',
        'KBU1','KBU2','KBU3','KBU4',
        'NEN1','NEN2','NEN3','NEN4',
        'PSU1','PSU2','PSU3',
        'SBU1','SBU2','SBU3',
        'SBU4',
        'SGN1','SGN2','SGN3','SGN4'
        ]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f'time_stamp is: {timestamp}')

config = {
    'lr0': 1e-2,
    'lrf': 1e-2,
    'warmup_epochs': 0,
    'time_stamp':timestamp,
    'parent_dir': '/cluster/project/eawag/p05001/repos/greyHeronRecognition/',
    'split': 'seasonal',
    'day_night': 'day',
    'n_last_im': 'none',
    'which_val': 'val',
    'conf_tsh_fixed': 0,
    'trn_val': False,
    'resample_trn': 'undersample',
    'n_cams_regroup': 12,
    'ls_cams':['SBU4'],
    'epochs': 1,
    'batch_size': 32,
    'weight_decay': 1e-12,
    'imgsz': 1280,
    'optimizer': 'SGD',
    'freeze': 12,
    'model': 'md_zenodo_v5b0.0.0.pt',
    'mosaic_prob': 1.0,
    'val_dataset': False,
    'val_megadetector': False,
    'reduced_dataset': False,
    'workers': get_num_tasks(),
    'n_gpus': get_num_gpus(),
    'seed': 0
    }
print(f'configs: {config}')

#initialize variables

lr0 = config['lr0']
lrf = config['lrf']
parent_dir = config['parent_dir']
split_key = config['split']
day_night = config['day_night']
n_last_im = config['n_last_im']
resample_trn = config['resample_trn']
ls_cams = config['ls_cams']

epochs = config['epochs']
batch_size = config['batch_size']
imgsz = config['imgsz']
optimizer = config['optimizer']
workers = config['workers']
freeze = config['freeze']
n_gpus = config['n_gpus']
model_path = f"{parent_dir}models/detection/{config['model']}"
mosaic_prob = config['mosaic_prob']
val_dataset = config['val_dataset']
reduced_dataset = config['reduced_dataset']
which_val = config['which_val']
weight_decay = config['weight_decay']
seed = config['seed']
val_megadetector = config['val_megadetector']
n_cams_regroup = config['n_cams_regroup']
warmup_epochs = config['warmup_epochs']
trn_val = config['trn_val']
conf_tsh_fixed = config['conf_tsh_fixed']

which_val = config['which_val']

if which_val == 'none':
    noval=True
else:
    noval=False

if split_key == 'seasonal':
    split = '_split2'
elif split_key == 'chronological':
    split = ''

#get data paths and labels

ls_cams_rest = [cam for cam in ls_cams_all if cam not in ls_cams]
path_trn = f'data/csv_files/dataSDSC_trn{split}.csv'
ls_images_trn_imb, ls_labelsClass_trn_imb = get_data_and_labels(csv_path=path_trn,
                                       ls_cams_filt=ls_cams,
                                       parent_dir=parent_dir,
                                       n_last_im=n_last_im,
                                       day_night=day_night)

if which_val == 'val_rest':
    ls_cams_val = ls_cams_rest
else:
    ls_cams_val = ls_cams

path_val = f'data/csv_files/dataSDSC_val{split}.csv'
path_tst = f'data/csv_files/dataSDSC_tst{split}.csv'

ls_images_val_imb, ls_labelsClass_val_imb = get_data_and_labels(
                                       csv_path=path_val,
                                       ls_cams_filt=ls_cams_val,
                                       parent_dir=parent_dir,
                                       n_last_im=n_last_im,
                                       day_night=day_night)

if trn_val:
    ls_images_trn_imb+=ls_images_val_imb
    ls_labelsClass_trn_imb+=ls_labelsClass_val_imb

if which_val=='tst':
    ls_images_tst_imb, ls_labelsClass_tst_imb = get_data_and_labels(
                                       csv_path=path_tst,
                                       ls_cams_filt=ls_cams_val,
                                       parent_dir=parent_dir,
                                       n_last_im=n_last_im,
                                       day_night=day_night)
    ls_images_val_imb=ls_images_tst_imb
    ls_labelsClass_val_imb=ls_labelsClass_tst_imb


#reduce dataset (full / imbalanced) for faster checks / debugging
if type(reduced_dataset)==int:
    n_samples_reduced = reduced_dataset
    ls_images_trn_imb = ls_images_trn_imb[:n_samples_reduced]
    ls_labelsClass_trn_imb = ls_labelsClass_trn_imb[:n_samples_reduced]
    ls_images_val_imb = ls_images_val_imb[:n_samples_reduced]
    ls_labelsClass_val_imb = ls_labelsClass_val_imb[:n_samples_reduced]

#data resampling

if resample_trn == 'log_oversample':
    ls_images_trn_tmp, ls_labelsClass_trn_tmp = log_oversample_pos(ls_images_trn_imb, ls_labelsClass_trn_imb,ls_cams)
    ls_images_trn, ls_labelsClass_trn = oversample_data(ls_images_trn_tmp, ls_labelsClass_trn_tmp)
if resample_trn == 'log_oversample_2':
    ls_images_trn_tmp, ls_labelsClass_trn_tmp = log_oversample_pos_2(ls_images_trn_imb, ls_labelsClass_trn_imb,ls_cams,n_cams_regroup)
    ls_images_trn, ls_labelsClass_trn = oversample_data(ls_images_trn_tmp, ls_labelsClass_trn_tmp)
elif resample_trn == 'oversample_naive':
    ls_images_trn, ls_labelsClass_trn = oversample_data(ls_images_trn_imb, ls_labelsClass_trn_imb)

elif resample_trn == 'undersample':
    ls_images_trn, ls_labelsClass_trn = undersample_data(ls_images_trn_imb, ls_labelsClass_trn_imb)
else:
    ls_images_trn = ls_images_trn_imb

ls_images_val, ls_labelsClass_val = undersample_data(ls_images_val_imb, ls_labelsClass_val_imb)

n_samples_trn = len(ls_images_trn)
n_pos_trn = ls_labelsClass_trn.count(1)
n_samples_val = len(ls_images_val)
n_pos_val = ls_labelsClass_val.count(1)

n_samples_trn_full = len(ls_images_trn_imb)
n_pos_trn_full = ls_labelsClass_trn_imb.count(1)
n_samples_val_full = len(ls_images_val_imb)
n_pos_val_full = ls_labelsClass_val_imb.count(1)

text_data_size = f'training with '+\
          f'n_samples {n_samples_trn} (n_pos: {n_pos_trn}) and '+\
          f'evaluating with n_samples {n_samples_val} (n_pos: {n_pos_val}), '+\
          f'resampling trn: {resample_trn});\n'+\
            f'full dataset trn has {n_samples_trn_full} samples ({n_pos_trn_full} pos) and '+\
            f'full dataset val has {n_samples_val_full} samples ({n_pos_val_full} pos) and '
print(text_data_size)

#----------------------------------------------------------------------------------------------------

#saving configs and data infos

output_dir = f'{parent_dir}greyHeronDetection/framework_pwl/runs/train/{timestamp}/'
output_dir_data = f'{output_dir}data/'
output_dir_configs = f'{output_dir}configs/'

trn_txt_path = f'{output_dir_data}trn_data.txt'
val_txt_path = f'{output_dir_data}val_data.txt'

dic_data = {'path': '',
            'train': trn_txt_path,
            'val': val_txt_path,
            'names': {0: 'grey_heron'}}

if not os.path.exists(output_dir_data):
    os.makedirs(output_dir_data)
if not os.path.exists(output_dir_configs):
    os.makedirs(output_dir_configs)

if 'SLURM_JOB_ID' in os.environ:
    job_id = os.environ.get('SLURM_JOB_ID')
else:
    job_id = 'none'
print('JOB ID:',job_id)
job_path = f'{parent_dir}greyHeronDetection/framework_pwl/runs/train/{timestamp}/job_id.txt'
with open(job_path, "w") as file:
    file.write(job_id)

save_dictionary(config, output_dir_configs, 'configurations.txt')

data_info_path = output_dir_data+"/data_info.txt"
with open(data_info_path, "w") as file:
    file.write(text_data_size)

with open(trn_txt_path, 'w') as file:
    for item in ls_images_trn:
        file.write("%s\n" % item)
with open(val_txt_path, 'w') as file:
    for item in ls_images_val:
        file.write("%s\n" % item)
data_yaml_path = f'{output_dir_data}data.yaml'
with open(data_yaml_path, 'w') as yaml_file:
    yaml.dump(dic_data, yaml_file)

#full dataset --> only works for real if noval and trn_val are false
    
trn_txt_path_full = f'{output_dir_data}trn_data_full.txt'
val_txt_path_full = f'{output_dir_data}val_data_full.txt'

dic_data_full = {'path': '',
            'train': trn_txt_path_full,
            'val': val_txt_path_full, #during validation, only data from this path will be taken
            'names': {0: 'grey_heron'}}

with open(trn_txt_path_full, 'w') as file:
    for item in ls_images_trn_imb:
        file.write("%s\n" % item)
with open(val_txt_path_full, 'w') as file:
    for item in ls_images_val_imb:
        file.write("%s\n" % item)
data_yaml_path_full = f'{output_dir_data}data_full.yaml'
with open(data_yaml_path_full, 'w') as yaml_file:
    yaml.dump(dic_data_full, yaml_file)
    
ls_images_rsp_trnAndVal = ls_images_trn + ls_images_val
trnAndVal_txt_path_rsp = f'{output_dir_data}trnAndVal_data_rsp.txt'
dic_data_rsp_trnAndVal = {'path': '',
            'train': '',
            'val': trnAndVal_txt_path_rsp, #during validation, only data from this path will be taken
            'names': {0: "animal", 1: "person", 2: "vehicle"}}
with open(trnAndVal_txt_path_rsp, 'w') as file:
    for item in ls_images_rsp_trnAndVal:
        file.write("%s\n" % item)
data_yaml_path_rsp_trnAndVal = f'{output_dir_data}data_rsp_trnAndVal.yaml'
with open(data_yaml_path_rsp_trnAndVal, 'w') as yaml_file:
    yaml.dump(dic_data_rsp_trnAndVal, yaml_file)
    
ls_images_full_trnAndVal = ls_images_trn_imb + ls_images_val_imb
trnAndVal_txt_path_full = f'{output_dir_data}trnAndVal_data_full.txt'
dic_data_full_trnAndVal = {'path': '',
            'train': '',
            'val': trnAndVal_txt_path_full, #during validation, only data from this path will be taken
            'names': {0: "animal", 1: "person", 2: "vehicle"}}
with open(trnAndVal_txt_path_full, 'w') as file:
    for item in ls_images_full_trnAndVal:
        file.write("%s\n" % item)
data_yaml_path_full_trnAndVal = f'{output_dir_data}data_full_trnAndVal.yaml'
with open(data_yaml_path_full_trnAndVal, 'w') as yaml_file:
    yaml.dump(dic_data_full_trnAndVal, yaml_file)

#--------------------------------------------------------------------------------
    
#YOLOv5 hyperparameter yaml file

hyp_dic = {
  'lr0': lr0,
  'lrf': lrf,
  'momentum': 0.937,
  'weight_decay': weight_decay,
  'warmup_epochs': warmup_epochs, ######
  'warmup_momentum': 0.8,
  'warmup_bias_lr': 0.1,
  'box': 0.05,
  'cls': 0.5,
  'cls_pw': 1.0,
  'obj': 1.0,
  'obj_pw': 1.0,
  'iou_t': 0.2,
  'anchor_t': 4.0,
  'fl_gamma': 0.0,
  'hsv_h': 0.015,
  'hsv_s': 0.7,
  'hsv_v': 0.4,
  'degrees': 0.0,
  'translate': 0.0, ###
  'scale': 0.0, ###
  'shear': 0.0,
  'perspective': 0.0,
  'flipud': 0.0,
  'fliplr': 0.5,
  'mosaic': mosaic_prob,
  'mixup': 0.0,
  'copy_paste': 0.0}
hyp_yaml_path = f'{output_dir_configs}hyp.yaml'
with open(hyp_yaml_path, 'w') as yaml_file:
    yaml.dump(hyp_dic, yaml_file)

#--------------------------------------------------------------------------------

#model training

project = 'runs/train'
name = f'{timestamp}/results'

yolo_path = f"{parent_dir}greyHeronDetection/yolov5"
device = ','.join(str(i) for i in range(n_gpus))

subprocess.run(['wandb', 'offline'])

if n_gpus > 0:
    train_command = [
        'python', 
        '-m', 'torch.distributed.run', 
        '--nproc_per_node', str(n_gpus),
        f"{yolo_path}/train_mod.py",
        '--batch', str(batch_size),
        '--data', f"{data_yaml_path}",
        '--weights', model_path,
        '--epochs', str(epochs),
        '--project', project,
        '--name', name,
        '--imgsz', str(imgsz),
        '--optimizer', optimizer,
        '--workers', str(workers),
        '--hyp', hyp_yaml_path,
        '--freeze', str(freeze),
        '--device', device,
        '--seed', str(seed),
        '--conf_tsh_fixed', str(conf_tsh_fixed)
    ]
    if noval:
        train_command.append('--noval')
    subprocess.run(train_command)

elif n_gpus==0:
    train_command = [
        'python',
        f"{yolo_path}/train_mod.py",
        '--batch', str(batch_size),
        '--data', f"{data_yaml_path}",
        '--weights', model_path,
        '--epochs', str(epochs),
        '--project', project,
        '--name', name,
        '--imgsz', str(imgsz),
        '--optimizer', optimizer,
        '--workers', str(workers),
        '--hyp', hyp_yaml_path,
        '--freeze', str(freeze),
        '--nosave',#, '--noval'
        '--seed', str(seed),
        '--conf_tsh_fixed', str(conf_tsh_fixed)
    ]
    if noval:
        train_command.append('--noval')
    subprocess.run(train_command)

print('Training finished')

#save results
analysis_dir = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{timestamp}/'
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

shutil.copy(job_path, analysis_dir)
shutil.copy(output_dir_configs+'configurations.txt', analysis_dir)
shutil.copy(data_info_path, analysis_dir)
shutil.copy(output_dir+'results/opt.yaml', analysis_dir)
if not noval:
    shutil.copy(output_dir+'results/saved_additional/dic_FitConfEp.json', analysis_dir)
    shutil.copy(output_dir+'results/saved_additional/classCM_losses_val.json', analysis_dir)
shutil.copy(output_dir+'results/saved_additional/classCM_losses_trn.json', analysis_dir)


#------------------------------------------------------------------------------------------------------------------------

#get and save baseline metrics (including Megadetector results)

if val_megadetector:

    print('evaluating Megadetector...')

    print('...on resampled dataset (used as validation)')

    save_path_MD_rsp = f'{analysis_dir}results_MD/trainAndVal_rsp/'

    if n_gpus>0:
        subprocess.run([
            'python', 
            '-m', 'torch.distributed.run', 
            '--nproc_per_node', str(n_gpus),
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size/8)),
            '--data', f"{data_yaml_path_rsp_trnAndVal}",
            '--weights', model_path,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(0.2),
            '--save-path', str(save_path_MD_rsp),
            '--device', device,
        ])

    elif n_gpus==0:
        subprocess.run([
            'python', 
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size/8)),
            '--data', f"{data_yaml_path_rsp_trnAndVal}",
            '--weights', model_path,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(0.2),
            '--save-path', str(save_path_MD_rsp),
        ])

    print('...on full dataset')

    save_path_MD_full = f'{analysis_dir}results_MD/traniAndVal_full/'

    if n_gpus>0:
        subprocess.run([
            'python', 
            '-m', 'torch.distributed.run', 
            '--nproc_per_node', str(n_gpus),
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size/8)),
            '--data', f"{data_yaml_path_full_trnAndVal}",
            '--weights', model_path,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(0.2),
            '--save-path', str(save_path_MD_full),
            '--device', device,
        ])

    elif n_gpus==0:
        subprocess.run([
            'python', 
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size/8)),
            '--data', f"{data_yaml_path_full_trnAndVal}",
            '--weights', model_path,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(0.2),
            '--save-path', str(save_path_MD_full),
        ])


    print('evaluation finished')

    MD_metrics_rsp_path = f'{save_path_MD_rsp}{trnAndVal_txt_path_rsp[109:-4]}/classification_metrics.json'
    MD_metrics_full_path = f'{save_path_MD_full}{trnAndVal_txt_path_full[109:-4]}/classification_metrics.json'
    with open(MD_metrics_rsp_path, 'r') as file:
        MD_metrics_rsp_dic = json.load(file)
    with open(MD_metrics_full_path, 'r') as file:
        MD_metrics_full_dic = json.load(file)

    MD_metrics_rsp = [MD_metrics_rsp_dic['accuracy'],
                    MD_metrics_rsp_dic['precision'],
                    MD_metrics_rsp_dic['recall'],
                    MD_metrics_rsp_dic['f1_score'],
                    MD_metrics_rsp_dic['specificity'],
                    MD_metrics_rsp_dic['balanced_accuracy']]

    MD_metrics_full = [MD_metrics_full_dic['accuracy'],
                    MD_metrics_full_dic['precision'],
                    MD_metrics_full_dic['recall'],
                    MD_metrics_full_dic['f1_score'],
                    MD_metrics_full_dic['specificity'],
                    MD_metrics_full_dic['balanced_accuracy']]


metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'balanced_accuracy']

if not val_megadetector:
    MD_metrics_rsp = [0]*len(metric_names)
    MD_metrics_full = [0]*len(metric_names)

if not trn_val:
    ls_labelsClass_complete = ls_labelsClass_trn_imb + ls_labelsClass_val_imb
if trn_val:
    ls_labelsClass_complete = ls_labelsClass_trn_imb
    

metrics_all_neg = get_baseline_metrics('all_neg',ls_labelsClass_complete,metric_names)
metrics_all_pos = get_baseline_metrics('all_pos',ls_labelsClass_complete,metric_names)
metrics_all_pos_balanced = ['0.5','0.5','1.0','0.667','0.0','0.5']
metrics_all_neg_balanced = ['0.5','0.0','0.0','0.0','1.0','0.5']
metrics_50_50 = get_baseline_metrics('50_50',ls_labelsClass_complete,metric_names)
metrics_class_pctg = get_baseline_metrics('class_pctg',ls_labelsClass_complete,metric_names)
pctg_pos = [ls_labelsClass_complete.count(1)/len(ls_labelsClass_complete)]*(len(metric_names))
pctg_neg = [ls_labelsClass_complete.count(0)/len(ls_labelsClass_complete)]*(len(metric_names))

baseline_metrics_dic = {'metric_names': metric_names, 
                            'metrics_all_neg': metrics_all_neg, 'metrics_all_pos': metrics_all_pos, 
                            'metrics_all_neg_balanced': metrics_all_neg_balanced, 'metrics_all_pos_balanced': metrics_all_pos_balanced,
                            'metrics_50_50': metrics_50_50, 'metrics_class_pctg': metrics_class_pctg,
                        'pctg_pos': pctg_pos, 'pctg_neg': pctg_neg, 
                        'MD_metrics_rsp': MD_metrics_rsp, 'MD_metrics_full': MD_metrics_full}

save_dictionary(baseline_metrics_dic, analysis_dir, 'baseline_metrics.json')


#------------------------------------------------------------------------------------------------------------------------

#plot learning curves (metrics and losses)

plot_learning_curves(time_stamp=timestamp,parent_dir=parent_dir)
print('plotting losses and classification metrics finished')

print('###############################################################')

#------------------------------------------------------------------------------------------------------------------------

#get and save maximum fitness and corresponding conf tsh & epoch
if not noval:
    dic_FCE_path = analysis_dir+'dic_FitConfEp.json'
    max_fitness, max_conf_tsh_eval, max_epoch = get_max_FCE(dic_FCE_path)
    dic_max_FCE = {'max_fitness': max_fitness, 'max_conf_tsh_stats': max_conf_tsh_eval, 'max_epoch': max_epoch}
    dic_max_FCE_path = analysis_dir+'dic_max_FitConfEp.json'
    with open(dic_max_FCE_path, 'w') as json_file:
        json.dump(dic_max_FCE, json_file)
    print(f'maximum fitness {max_fitness} with conf tsh {max_conf_tsh_eval} at epoch {max_epoch} of {epochs}')

print('###############################################################')

#------------------------------------------------------------------------------------------------------------------------

#evaluate trained model on validation set --> unimportant part

if val_dataset:

    print('evaluating best model...')

    print('...on resampled dataset')

    best_model = f'{parent_dir}greyHeronDetection/framework_pwl/runs/train/{timestamp}/results/weights/best.pt'
    save_path_rsp = f'{analysis_dir}results_eval_best/val_rsp/'
    if n_gpus>0:
        subprocess.run([
            'python', 
            '-m', 'torch.distributed.run', 
            '--nproc_per_node', str(n_gpus),
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size/4)),
            '--data', f"{data_yaml_path}",
            '--weights', best_model,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(max_conf_tsh_eval),
            '--save-path', str(save_path_rsp),
            '--device', device,
        ])

    elif n_gpus==0:
        subprocess.run([
            'python', 
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size/4)),
            '--data', f"{data_yaml_path}",
            '--weights', best_model,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(max_conf_tsh_eval),
            '--save-path', str(save_path_rsp),
        ])


    print('...on full dataset')
    save_path_full = f'{analysis_dir}results_eval_best/val_full/'


    if n_gpus>0:
        subprocess.run([
            'python', 
            '-m', 'torch.distributed.run', 
            '--nproc_per_node', str(n_gpus),
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size/4)),
            '--data', f"{data_yaml_path_full}",
            '--weights', best_model,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(max_conf_tsh_eval),
            '--save-path', str(save_path_full),
            '--device', device,
        ])

    elif n_gpus==0:
        subprocess.run([
            'python', 
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size/4)),
            '--data', f"{data_yaml_path_full}",
            '--weights', best_model,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(max_conf_tsh_eval),
            '--save-path', str(save_path_full),
        ])





print('end of code')

