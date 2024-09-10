import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch import nn
from torchmetrics import Accuracy, F1Score, Specificity, Precision, Recall, ConfusionMatrix
import pandas as pd
import time
import sys
import glob
#from torchsummary import summary
from sklearn.utils.class_weight import compute_class_weight
import os
import csv
from torchvision.transforms import v2
import copy
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from model.model_class import modelClass
from data_loader.data_loader import dataLoader
from utils.train_utils import train_one_epoch, evaluate_one_epoch, get_data_and_labels, oversample_data, undersample_data, get_baseline_metrics, load_data_torch, save_dictionary, log_oversample_pos, get_num_gpus, get_num_tasks
from analysis.analysis_utils import plotAndSaveCM

def evaluate_config(config_eval):
    """
    Function to evaluate model (from training timestamp) on specified dataset
    Args:
        config (dictionary): contains list of evaluation configurations, for options see train_job.sh
            'parent_dir (str): parent directory
            'image_size' (int): image resolution
            'ls_cams_filt' (list): list of filtered cameras
            'num_classes' (int): number of classes
            'pretrained_network' (str): pretrained model
            'batch_size' (int): batch size
            'which_weights' (str): unfrozen weights from last or all layers
            'time_stamp' (str): time stamp of training job originally
            'n_last_im' (int): number of last images for background removal; if 'none', then original images are taken
            'day_night' (str): day or night-time images, or both
            'resample' (str): resample method applied to dataset
            'split' (str): first ('chronological') or second ('seasonal') split used
            'num_workers' (int): number of workers used for data loading
            'best_last' (str): load best ('best') or last ('last') model during training
            'which_set' (str): training ('trn'), validation ('val'), both ('trn_val') or test ('tst')
    """

    timestamp_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    parent_dir = config_eval['parent_dir']
    split = config_eval['split']
    n_last_im = config_eval['n_last_im']
    day_night = config_eval['day_night']
    ls_cams_filt = config_eval['ls_cams_filt']
    which_set = config_eval['which_set'] #if test, no resampling
    resample = config_eval['resample']
    batch_size = config_eval['batch_size']
    image_size = config_eval['image_size']
    num_workers = config_eval['num_workers']
    time_stamp = config_eval['time_stamp']
    pretrained_network = config_eval['pretrained_network']
    which_weights = config_eval['which_weights']
    num_classes = config_eval['num_classes']
    best_last = config_eval['best_last']

    #load model
    file_path = glob.glob(f'{parent_dir}greyHeronClassification/logs/checkpoints/{best_last}_model_weightsAndMetrics*_{time_stamp}.pth')[0]
    dic_best = torch.load(file_path,map_location=torch.device('cpu'))
    best_model_dic = dic_best['model']
    new_state_dict = {k.replace('module.', ''): v for k, v in best_model_dic.items()}
    best_model = modelClass(
        pretrained_network=pretrained_network, 
        num_classes=num_classes, 
        which_weights=which_weights,
    )
    best_model.load_state_dict(new_state_dict)

    #load data
    data_loader = load_data_torch(
        split=split,
        ls_cams_filt=ls_cams_filt,
        parent_dir=parent_dir,
        n_last_im=n_last_im,
        day_night=day_night,
        resample=resample,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        add_transforms=[],
        mode=which_set
    )

    ls_images = data_loader.dataset.img_paths
    ls_labels = data_loader.dataset.labels
    n_samples = len(ls_images)
    n_pos = ls_labels.count(1)

    #---------------sending model and data to gpu (cuda)-------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for gpu_id in range(num_gpus):
            gpu_properties = torch.cuda.get_device_properties(gpu_id)
            gpu_size = gpu_properties.total_memory / (1024 ** 3)
            print(f"GPU {gpu_id} - Name: {gpu_properties.name}, Memory Size: {gpu_size:.2f} GB")
    else:
        print("No GPU available. Please ensure you have a compatible GPU and have installed the appropriate drivers.")
        torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        best_model = nn.DataParallel(best_model)
    best_model.to(device)
    """for metric in metrics:
        metric.to(torch.device(device))"""
    
    #-------------------------model evaluation------------------------------
    
    text_data_size = f'evaluating on '+\
          f'n_samples {n_samples} (n_pos: {n_pos})'
    print(text_data_size)

    ls_probabilities_np = np.array([])
    ls_paths_np = np.array([])
    ls_labels_np = np.array([])
    ls_predictions_np = np.array([])

    best_model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, vbatch in enumerate(data_loader):
            best_model.eval()
            vimages, vlabels, paths = vbatch
            if torch.cuda.is_available():
                vimages = vimages.to(device)
                vlabels = vlabels.to(device)
            voutputs = best_model(vimages)
            vprobabilities = torch.softmax(voutputs, dim=1)
            _, vpredictions = torch.max(vprobabilities, dim=1)

            probabilities_np = vprobabilities.cpu().numpy()
            probabilities_np_1 = probabilities_np[:,1]
            paths_np = np.array(paths)
            labels_np = vlabels.cpu().numpy()
            predictions_np = vpredictions.cpu().numpy()
            ls_probabilities_np = np.append(ls_probabilities_np, probabilities_np_1)
            ls_paths_np = np.append(ls_paths_np, np.squeeze(paths_np))
            ls_labels_np = np.append(ls_labels_np, np.squeeze(labels_np))
            ls_predictions_np = np.append(ls_predictions_np, np.squeeze(predictions_np))

    #compute metrics
    f1_full = f1_score(ls_labels_np,ls_predictions_np)
    acc_full = accuracy_score(ls_labels_np,ls_predictions_np)
    rec_full = recall_score(ls_labels_np,ls_predictions_np)
    prec_full = precision_score(ls_labels_np,ls_predictions_np)
    spec_full = recall_score(ls_labels_np,ls_predictions_np,pos_label=0)
    bal_acc_full = (rec_full+spec_full)/2
    CM_full = confusion_matrix(ls_labels_np,ls_predictions_np)
    CM_norm = confusion_matrix(ls_labels_np,ls_predictions_np,normalize='true')


    print('-----------------------')
    print(f'f1: {f1_full}')
    print(f'acc: {acc_full}')
    print(f'rec: {rec_full}')
    print(f'prec: {prec_full}')
    print(f'spec: {spec_full}')
    print(f'bal_acc: {bal_acc_full}')
    print(f'CM: {CM_full}')
    print('-----------------------')

    dic_metrics = {'f1_score': f1_full, 'acc': acc_full, 'rec': rec_full, 'prec': prec_full, 'spec': spec_full, 'bal_acc': bal_acc_full, 'CM': CM_full.tolist(), 'CM_norm': CM_norm.tolist()}

    #save results
    save_path=f'{parent_dir}greyHeronClassification/analysis/output_eval/{timestamp_now}/'
    os.makedirs(save_path, exist_ok=True)
    plotAndSaveCM(CM_norm,save_path+'confusion_matrix_norm.png','Normalized Confusion Matrix') 
    save_dictionary(config_eval, save_path, 'configurations_eval.txt')
    save_dictionary(dic_metrics, save_path, 'metrics_eval.txt')
    file_path = save_path+"data_info.txt"
    with open(file_path, "w") as file:
        file.write(text_data_size)

"""
#example usage:
config_eval = {
    'parent_dir':  '/cluster/project/eawag/p05001/repos/greyHeronRecognition/',
    'split': '_split2',
    'n_last_im': 'none',
    'day_night': 'day',
    'ls_cams_filt': ['SBU4'],
    'which_set': 'tst', #'trn, 'val', 'tst', 'trn_val'    
    'resample': 'no_resample',
    'batch_size': 8,
    'image_size': 896,
    'num_workers': get_num_gpus(),
    'time_stamp': '20240904_122224',
    'best_last': 'last',
    'pretrained_network': 'mobilenet',
    'which_weights': 'all',
    'num_classes': 2,
    'split': 'seasonal'
}
evaluate_config(config_eval=config_eval)
"""