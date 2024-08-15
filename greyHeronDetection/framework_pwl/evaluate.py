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


def evaluate_config(config_eval):
    """
    Function to evaluate model (from training timestamp or Megadetector) on specified dataset
    Args:
        config (dictionary): contains list of evaluation configurations
            'parent_dir' (str): parent directory
            'imgsz' (int): image resolution
            'ls_cams' (list): list of filtered cameras
            'batch_size' (int): batch size
            'time_stamp' (str): time stamp of training job originally or 'megadetector'
            'n_last_im' (int): number of last images for background removal; if 'none', then original images are taken
            'day_night' (str): day or night-time images, or both
            'resample' (str): resample method applied to dataset
            'split' (str): first ('chronological') or second ('seasonal') split used
            'workers' (int): number of workers used for data loading
            'best_last' (str): load best ('best') or last ('last') model during training
            'n_gpus' (int): number of gpus used
            'conf_tsh' (float or str): specifies confidence threshold; if set to 
                'get_config': extract confidence threshold saved for training configs
                'get_stats': extract confidence thereshold yielding maximum F1 score during training
                'best': determine new condifence thereshold yielding maximum F1 score during evaluation for specified model and dataset
            'n_cams_regroup' (int): number of regrouped cameras for log oversampling 2
            'which_set' (str): training ('trn'), validation ('val'), both ('trn_val') or test ('tst')
    """

    print('configs:',config_eval)

    #initialize variables

    timestamp_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'current time_stamp is: {timestamp_now}')

    parent_dir = config_eval['parent_dir']
    split_key = config_eval['split']
    n_last_im = config_eval['n_last_im']
    n_gpus = config_eval['n_gpus']
    conf_tsh = config_eval['conf_tsh']
    day_night = config_eval['day_night']
    ls_cams = config_eval['ls_cams']
    batch_size = config_eval['batch_size']
    imgsz = config_eval['imgsz']
    workers = config_eval['workers']
    time_stamp = config_eval['time_stamp'] #can be actual time_stamp or megadetector
    n_cams_regroup = config_eval['n_cams_regroup']
    which_set=config_eval['which_set']
    resample=config_eval['resample']
    best_last=config_eval['best_last']

    if split_key == 'seasonal':
        split = '_split2'
    elif split_key == 'chronological':
        split = ''

    #get conf tsh

    if conf_tsh=='get_stats': #conf tsh giving max F1 saved in .json file
        dic_path = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{time_stamp}/dic_max_FitConfEp.json'
        with open(dic_path, 'r') as file:
            dic = json.load(file)
        conf_tsh = float(dic['max_conf_tsh_stats'])
        print(f'conf_tsh value obtained from dic_max_FitConfEp.json: {conf_tsh}')
    
    if conf_tsh=='get_config': #specified in training configs
        dic_path = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{time_stamp}/configurations.txt'
        with open(dic_path, 'r') as file:
            dic = json.load(file)
        conf_tsh = float(dic['conf_tsh_fixed'])
        print(f'conf_tsh value obtained from configutations.json: {conf_tsh}')

     #get model

    if time_stamp != 'megadetector':
        n_classes = 1
        model_path=f'{parent_dir}greyHeronDetection/framework_pwl/runs/train/{time_stamp}/results/weights/{best_last}.pt'
    if time_stamp == 'megadetector':
        n_classes = 3
        model = 'md_zenodo_v5b0.0.0.pt'
        model_path=f"{parent_dir}greyHeronDetection/saved_models/{model}"

    if n_classes == 3:
        class_names = {0: "animal", 1: "person", 2: "vehicle"}
    elif n_classes == 1:
        class_names = {0: "grey_heron"}

    #data preparation

    if which_set=='trn':
        path_csv=f'dataPreprocessing/csv_files/dataSDSC_trn{split}.csv'
    if which_set=='val':
        path_csv=f'dataPreprocessing/csv_files/dataSDSC_val{split}.csv'
    if which_set=='tst':
        path_csv=f'dataPreprocessing/csv_files/dataSDSC_tst{split}.csv'

    if which_set!='trn_val':
        ls_images_imb, ls_labels_imb = get_data_and_labels(path_csv,ls_cams,parent_dir,n_last_im,day_night)
    if which_set=='trn_val':
        path_csv_1=f'dataPreprocessing/csv_files/dataSDSC_trn{split}.csv'
        path_csv_2=f'dataPreprocessing/csv_files/dataSDSC_val{split}.csv'
        ls_images_imb_1, ls_labels_imb_1 = get_data_and_labels(path_csv_1,ls_cams,parent_dir,n_last_im,day_night)
        ls_images_imb_2, ls_labels_imb_2 = get_data_and_labels(path_csv_2,ls_cams,parent_dir,n_last_im,day_night)
        ls_images_imb = ls_images_imb_1+ls_images_imb_2
        ls_labels_imb = ls_labels_imb_1+ls_labels_imb_2

    if resample=='undersample':
        print('undersampling')
        ls_images, ls_labels = undersample_data(ls_images_imb, ls_labels_imb)
    if resample=='oversample_naive':
        ls_images, ls_labels = oversample_data(ls_images_imb, ls_labels_imb)
    if resample == 'log_oversample':
        ls_images_tmp, ls_labels_tmp = log_oversample_pos(ls_images_imb, ls_labels_imb, ls_cams)
        ls_images, ls_labels = oversample_data(ls_images_tmp, ls_labels_tmp)
    if resample == 'log_oversample_2':
        ls_images_tmp, ls_labels_tmp = log_oversample_pos_2(ls_images_imb, ls_labels_imb, ls_cams, n_cams_regroup)
        ls_images, ls_labels = oversample_data(ls_images_tmp, ls_labels_tmp)
    if resample == 'no_resample':
            ls_images, ls_labels = ls_images_imb, ls_labels_imb

    text_data_info = f'evaluating on {len(ls_images)} samples (n_pos: {ls_labels.count(1)})'
    print(text_data_info)

    #saving data info to yaml file
        
    output_eval = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output_eval/{timestamp_now}/'
    if not os.path.exists(output_eval):
        os.makedirs(output_eval)
    output_eval_data = f'{output_eval}data/'
    if not os.path.exists(output_eval_data):
        os.makedirs(output_eval_data)

    save_dictionary(config_eval, output_eval, 'configurations_eval.txt')

    data_txt_path = f'{output_eval_data}data_eval.txt'
    data_dic = {'path': '',
                'train': '',
                'val': data_txt_path, #during validation, only data from this path will be taken
                'names': class_names}
    with open(data_txt_path, 'w') as file:
        for item in ls_images:
            file.write("%s\n" % item)
    data_yaml_path = f'{output_eval_data}data_eval.yaml'
    with open(data_yaml_path, 'w') as yaml_file:
        yaml.dump(data_dic, yaml_file)

    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ.get('SLURM_JOB_ID')
    else:
        job_id = 'none'
    print('JOB ID:',job_id)
    job_path = f'{output_eval}job_id.txt'
    with open(job_path, "w") as file:
        file.write(job_id)

    txt_data_path = f'{output_eval}data_info.txt'
    with open(txt_data_path, "w") as file:
        file.write(text_data_info)

    #evaluate model
        
    yolo_path = f"{parent_dir}greyHeronDetection/yolov5"
    device = ','.join(str(i) for i in range(n_gpus))

    save_results_path = f'{output_eval}results/'

    if n_gpus>0:
        subprocess.run([
            'python', 
            '-m', 'torch.distributed.run', 
            '--nproc_per_node', str(n_gpus),
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size)),
            '--data', f"{data_yaml_path}",
            '--weights', model_path,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(conf_tsh),
            '--save-path', str(save_results_path),
            '--device', device,
        ])

    elif n_gpus==0:
        subprocess.run([
            'python', 
            f"{yolo_path}/val_mod.py",
            '--batch-size', str(int(batch_size)),
            '--data', f"{data_yaml_path}",
            '--weights', model_path,
            '--imgsz', str(imgsz),
            '--workers', str(workers),
            '--conf-tsh-eval', str(conf_tsh),
            '--save-path', str(save_results_path),
        ])




#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

ls_cams_all = [
        'GBU1','GBU2','GBU3','GBU4',
        'KBU1','KBU2','KBU3','KBU4',
        'NEN1','NEN2','NEN3','NEN4',
        'PSU1','PSU2','PSU3',
        'SBU1','SBU2','SBU3',
        'SBU4',
        'SGN1','SGN2','SGN3','SGN4'
        ]
ls_cams_nonSBU4 = ls_cams_rest = [cam for cam in ls_cams_all if cam not in ['SBU4']]
ordered_ls_cams = ['SBU4', 'NEN1', 'SGN3', 'SGN4', 'SGN1', 'SBU3', 'NEN3', 'NEN2', 'SGN2', 'NEN4', 'SBU2', 'SBU1', 'PSU3', 'KBU2', 'PSU1', 'PSU2', 'KBU4', 'KBU3', 'GBU4', 'KBU1', 'GBU3', 'GBU2', 'GBU1']
n_cams_regrouped = 12
ls_cams_single=ordered_ls_cams[:-12]
ls_cams_regrouped=ordered_ls_cams[-12:]

###important trainings
#SBU4 US best (trn & val): 20240615_170317
#SBU4 OS (trn & val): 
#SBU4 OS (trn + val): 20240615_204402
#All cams log OS 2 (trn & val): 20240607_104246
#All cams log OS 2 (trn + val): 20240615_215852
#All cams OS (trn & val): 20240615_220500
#All cams OS (trn + val): 20240616_134817
#SBU4OS (trn+val): 20240615_204402

"""
#Example usage:
config_eval = {
    'time_stamp': '20240616_134817',
    'conf_tsh': 'get_config',
    'parent_dir': '/cluster/project/eawag/p05001/civil_service/',
    'split': 'seasonal',
    'day_night': 'day',
    'n_last_im': 'none',
    'which_set': 'tst',  
    'resample': 'no_resample',
    'ls_cams':['SBU4'],
    'best_last': 'last',
    'imgsz': 1280,
    'workers': get_num_tasks(),
    'n_gpus': get_num_gpus(),
    'batch_size': 8,
    'n_cams_regroup': 12
    }
evaluate_config(config_eval=config_eval)
"""