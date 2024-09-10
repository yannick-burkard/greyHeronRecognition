import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torchvision.models as models
import time
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import json
import sys
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score


def get_data_and_labels(csv_path,ls_cams_filt,parent_dir,n_last_im,day_night):
    """
     Function to extract sample paths and labels from input csv file
    Args:
        csv_path (str): path leading to csv file listing dataset
        ls_cams_filt (list): list of cameras to be filtered
        parent_dir (str): parent directory
        n_last_im (int): number of last images for background removal; if 'none', then original images are taken
        day_night (str): day or night images, or both
    Returns:
        ls_images (list): list of image paths
        ls_labels (list): list of image labels
    """
    csv_path = parent_dir+csv_path
    df_data = pd.read_csv(csv_path)
    df_data_filt = df_data[df_data['camera'].isin(ls_cams_filt)]
    if day_night == 'day':
        df_data_filt = df_data_filt[df_data_filt['infrared'] == False]
    if day_night == 'night':
        df_data_filt = df_data_filt[df_data_filt['infrared'] == True]
    ls_images_SIAM = df_data_filt['file_path'].tolist()
    ls_dates = df_data_filt['time_stamp'].tolist()
    ls_labels = df_data_filt['class_grey_herons'].tolist()
    ls_image_datesPathsLabels = list(zip(ls_dates,ls_images_SIAM,ls_labels))
    sorted_image_datesPathsLabels = sorted(ls_image_datesPathsLabels, key=lambda x: x[0])
    ls_images_SIAM = [file_path for _, file_path, _ in sorted_image_datesPathsLabels]
    ls_images = [parent_dir+'data/dataset/'+ls_images_SIAM[i][35:] for i in range(len(ls_images_SIAM))]
    ls_labels = [file_label for _, _, file_label in sorted_image_datesPathsLabels]
    if len(n_last_im)<4:
        print(f"Using images with bkg removed using last {n_last_im}")
        prefix='dataDriveMichele_noBkg/dataDriveMichele_noBkg448_'
        ls_images=[f'{parent_dir}data/dataset/{prefix}{n_last_im}/SBU4/{image[-22:]}' for image in ls_images[(n_last_im+2):]]
        ls_labels = ls_labels[(n_last_im+2):]
    else:
        print(f"Using images with no bkg removed")
    return ls_images, ls_labels





def undersample_data(ls_paths,ls_labels):
    """
    Function to undersample majority class from two classes (0 and 1) to create a balanced dataset
    Args:
        ls_paths (list): list of image paths
        ls_labels (list): list of image labels
    Returns:
        paths_undersampled (list): list of undersampled image paths
        labels_undersampled (list): list of undersampled label paths
    """
    

    min_label = np.argmin(np.array([ls_labels.count(0),ls_labels.count(1)]))
    maj_label = np.argmax(np.array([ls_labels.count(0),ls_labels.count(1)]))

    n_samples_maj = ls_labels.count(maj_label)
    n_samples_min = ls_labels.count(min_label)
    diff_maj_min = n_samples_maj-n_samples_min

    zipped_data = list(zip(ls_paths,ls_labels))
    filtered_maj = [(path, label) for path, label in zipped_data if label == maj_label]
    filtered_min = [(path, label) for path, label in zipped_data if label == min_label]

    random.seed(-1)
    reduced_maj = random.sample(filtered_maj,k=int(n_samples_min))
    data_undersampled = reduced_maj + filtered_min

    paths_undersampled = [path for path, _ in data_undersampled]
    labels_undersampled = [label for _, label in data_undersampled]

    return paths_undersampled, labels_undersampled



def oversample_data(ls_paths,ls_labels):
    """
    Function to undersample majority class from two classes (0 and 1) to create a balanced dataset
    Args:
        ls_paths (list): list of image paths
        ls_labels (list): list of image labels
    Returns:
        paths_oversampled (list): list of oversampled image paths
        labels_oversampled (list): list of oversampled label paths
    """

    min_label = np.argmin(np.array([ls_labels.count(0),ls_labels.count(1)]))
    maj_label = np.argmax(np.array([ls_labels.count(0),ls_labels.count(1)]))

    n_samples_maj = ls_labels.count(maj_label)
    n_samples_min = ls_labels.count(min_label)
    diff_maj_min = n_samples_maj-n_samples_min
    diff_maj_min_module = n_samples_maj % n_samples_min

    zipped_data = list(zip(ls_paths,ls_labels))
    filtered_maj = [(path, label) for path, label in zipped_data if label == maj_label]
    filtered_min = [(path, label) for path, label in zipped_data if label == min_label]

    oversampled_min = copy.deepcopy(filtered_min)
    while len(oversampled_min)<diff_maj_min:
        oversampled_min+=filtered_min
    random.seed(-1)
    oversampled_min+=random.sample(filtered_min,k=diff_maj_min_module)

    data_oversampled = filtered_maj + oversampled_min

    paths_oversampled = [path for path, _ in data_oversampled]
    labels_oversampled = [label for _, label in data_oversampled]

    return paths_oversampled, labels_oversampled



def save_dictionary(data, directory, filename):
    """save data as dictionary on path directory/filename"""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)



def get_max_FCE(dic_path):
    """
    Function to obtain maximum fitness variables
    Args:
        dic_path (str): path specifying dictionary containing results for a specific training procedure
    Returns:
        max_fitness (float): maximum fitness
        max_conf_tsh (float): confidence threshold associated with maximum fitness
        max_epoch+1 (int): epoch number associated with maximum fitness
    """
   
    ls_fitness, ls_conf_tsh, ls_epoch = [], [], []
    with open(dic_path, "r") as file:
        for line in file:
            dic = json.loads(line)
            ls_fitness.append(dic['fitness'])
            ls_conf_tsh.append(dic['conf_tsh_stats'])
            ls_epoch.append(dic['epoch'])
    
    i = np.argmax(np.array(ls_fitness))
    max_fitness = ls_fitness[i]
    max_conf_tsh = ls_conf_tsh[i]
    max_epoch = ls_epoch[i]
            
    return max_fitness, max_conf_tsh, max_epoch+1



def get_baseline_metrics(which_baseline,ls_labels,ls_metric_names):
    """
    Function to get metrics based on baseline model:
        Args:
            which_baseline (str): which baseline model to use, options are
                'all_neg' (all negative), 'all_pos' (all positive), '50_50' (random 50/50 guessing), 'class_pctg' (random class pctg guessing)
            ls_labels (list): list of image labels
            metric_name (list): list of metric names used
        Returns:
            ls_metric_values (list): list of metric values
     """

    np.random.seed(27469237)

    if which_baseline=='all_neg':
        predictions = torch.tensor(np.zeros((len(ls_labels))))
    if which_baseline=='all_pos':
        predictions = torch.tensor(np.ones((len(ls_labels))))
    if which_baseline=='50_50':
        predictions = torch.tensor(np.random.choice([0, 1], size=len(ls_labels), p=[0.5, 0.5]))
    if which_baseline=='class_pctg':
        p_pos = ((list(ls_labels)).count(1))/(len(ls_labels))
        predictions = torch.tensor(np.random.choice([0, 1], size=len(ls_labels), p=[1-p_pos, p_pos]))

    ls_metric_values = []
    for i, metric in enumerate(ls_metric_names):
        if metric == 'accuracy':
            ls_metric_values.append(accuracy_score(ls_labels,predictions))
        if metric == 'precision':
            ls_metric_values.append(precision_score(ls_labels,predictions))
        if metric == 'recall':
            ls_metric_values.append(recall_score(ls_labels,predictions))
        if metric == 'f1_score':
            ls_metric_values.append(f1_score(ls_labels,predictions))
        if metric == 'specificity':
            ls_metric_values.append(recall_score(ls_labels,predictions,pos_label=0))
        if metric == 'balanced_accuracy':
            ls_metric_values.append(balanced_accuracy_score(ls_labels,predictions))
    
    return ls_metric_values

def log_oversample_pos(ls_paths,ls_labels,ls_cams):
    """
    Function to apply logarithmic undersampling accross different cameras using equation n_i' = n_i*log(1+n_max/n_i)
    Args:
        ls_paths (list): list of image paths
        ls_labels (list): list of labels
        ls_cams (list): list of cameras
    Returns:
        paths_total (list): list of resutling image paths
        labels_total (list): list of resulting label paths
    """

    dic_samples_per_cam = {}
    max_samples = 0
    max_cam = ''

    for cam in ls_cams:
        paths_labels_cam = [(path,label) for path,label in zip(ls_paths,ls_labels) if (path[-17:-13]==cam and label==1)]
        n_samples_cam = len(paths_labels_cam)
        if n_samples_cam>max_samples:
            max_samples=n_samples_cam
            max_cam=cam
        dic_samples_per_cam[cam] = paths_labels_cam

    paths_labels_pos_new = []

    for cam in ls_cams:

        paths_labels_cam = dic_samples_per_cam[cam]
        if cam == max_cam:
            paths_labels_pos_new+=paths_labels_cam
            continue
        n_samples_cam = len(paths_labels_cam)
        if n_samples_cam == 0:
            continue
        n_samples_cam_pr = int(n_samples_cam*np.log(1+max_samples/n_samples_cam))

        diff_samples_cam = n_samples_cam_pr - n_samples_cam
        diff_samples_cam_module = n_samples_cam_pr % n_samples_cam

        os_paths_labels_cam = copy.deepcopy(paths_labels_cam)

        while len(os_paths_labels_cam)<diff_samples_cam:
            os_paths_labels_cam+=paths_labels_cam
        
        random.seed(-1)
        os_paths_labels_cam+=random.sample(paths_labels_cam,k=diff_samples_cam_module)

        paths_labels_pos_new+=os_paths_labels_cam
    
    paths_labels_neg = [(path,label) for path,label in zip(ls_paths,ls_labels) if (label==0)]
    paths_labels_total = paths_labels_neg + paths_labels_pos_new

    paths_total = [path for path, _ in paths_labels_total]
    labels_total = [label for _, label in paths_labels_total]

    return paths_total, labels_total

def contract_array_str(arr,n_last,name):
    """Function to replace last n_last elements of an array arr with a string name"""
    arr_ret = arr[:(-n_last)]
    arr_ret=np.append(arr_ret,name)
    return arr_ret

def log_oversample_pos_2(ls_paths,ls_labels,ls_cams,n_cams_regroup):
    """
    Function to apply logarithmic undersampling accross different cameras using equation n_i' = n_i*log(1+n_max/n_i)/log(2) and camera regrouping --> used for paper
    Args:
        ls_paths (list): list of image paths
        ls_labels (list): list of labels
        ls_cams (list): list of cameras
        n_cams_regroup (int): number of cameras to be regrouped (with fewest positives)
    Returns:
        paths_total (list): list of resutling image paths
        labels_total (list): list of resulting label paths
    """

    dic_samples_per_cam = {}
    max_samples = 0
    max_cam = ''

    ordered_ls_cams = ['SBU4', 'NEN1', 'SGN3', 'SGN4', 'SGN1', 'SBU3', 'NEN3', 'NEN2', 'SGN2', 'NEN4', 'SBU2', 'SBU1', 'PSU3', 'KBU2', 'PSU1', 'PSU2', 'KBU4', 'KBU3', 'GBU4', 'KBU1', 'GBU3', 'GBU2', 'GBU1']
    name_regroup = 'others'
    ordered_ls_cams = contract_array_str(ordered_ls_cams,n_cams_regroup,name_regroup)

    for cam in ls_cams:
        paths_labels_cam = [(path,label) for path,label in zip(ls_paths,ls_labels) if (path[-17:-13]==cam and label==1)]
        n_samples_cam = len(paths_labels_cam)
        if n_samples_cam>max_samples:
            max_samples=n_samples_cam
            max_cam=cam
        if 'others' not in dic_samples_per_cam.keys() and cam not in ordered_ls_cams:
            dic_samples_per_cam[name_regroup] = paths_labels_cam
        elif 'others' in dic_samples_per_cam.keys() and cam not in ordered_ls_cams:
            dic_samples_per_cam[name_regroup]+=paths_labels_cam
        else:
            dic_samples_per_cam[cam] = paths_labels_cam

    paths_labels_pos_new = []

    for cam in dic_samples_per_cam.keys():

        paths_labels_cam = dic_samples_per_cam[cam]
        if cam == max_cam:
            paths_labels_pos_new+=paths_labels_cam
            continue
        n_samples_cam = len(paths_labels_cam)
        if n_samples_cam == 0:
            continue
        n_samples_cam_pr = int(n_samples_cam*np.log(1+max_samples/n_samples_cam)/np.log(2))

        diff_samples_cam = n_samples_cam_pr - n_samples_cam
        diff_samples_cam_module = n_samples_cam_pr % n_samples_cam

        os_paths_labels_cam = copy.deepcopy(paths_labels_cam)

        while len(os_paths_labels_cam)<diff_samples_cam:
            os_paths_labels_cam+=paths_labels_cam

        random.seed(-1)
        os_paths_labels_cam+=random.sample(paths_labels_cam,k=diff_samples_cam_module)

        paths_labels_pos_new+=os_paths_labels_cam
        print(f'for cam {cam} we had {len(paths_labels_cam)} and now {len(os_paths_labels_cam)}, so {int(len(os_paths_labels_cam)/len(paths_labels_cam))} copies per sample!')
    
    paths_labels_neg = [(path,label) for path,label in zip(ls_paths,ls_labels) if (label==0)]
    paths_labels_total = paths_labels_neg + paths_labels_pos_new

    paths_total = [path for path, _ in paths_labels_total]
    labels_total = [label for _, label in paths_labels_total]

    return paths_total, labels_total

def get_num_gpus():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        num_gpus = 0
    return num_gpus

def get_num_tasks():
    if 'SLURM_NTASKS' in os.environ:
        num_tasks = int(os.environ['SLURM_NTASKS'])
    else:
        num_tasks = 1
    return num_tasks





