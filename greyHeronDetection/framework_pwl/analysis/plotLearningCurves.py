import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import sys
import json
import glob
sys.path.append('..')
from analysis.analysis_utils import get_CM_loss_dic, get_metrics, plot_and_save, plot_and_save_avg

def plot_learning_curves(parent_dir,time_stamp):
    """
    Function to plot learing curves (loss and metrics) after a training procedure:
    Args:
        parent_dir (str): parent dictory
        time_stamp (str): time_stamp for training procedure
    """
    
    #load configs and initialize variables
    config_path=f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{time_stamp}/configurations.txt'
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)

    parent_dir = config['parent_dir']
    day_night = config['day_night']
    resample_trn = config['resample_trn']
    batch_size = config['batch_size']
    imgsz = config['imgsz']
    optimizer = config['optimizer']
    freeze = config['freeze']
    mosaic_prob = config['mosaic_prob']
    which_val = config['which_val']
    seed = config['seed']
    lr0 = config['lr0']
    which_val = config['which_val']

    if which_val == 'none':
        noval=True
    else:
        noval=False

    text_config = f'freeze: {freeze}, bs: {batch_size}, d/n: {day_night}, is: {imgsz}, rst: {resample_trn}, mp: {mosaic_prob}, opt: {optimizer}, wv: {which_val}, s: {seed}, lr0: {lr0}'

    #-----------------------------------------------------------------------

    print('---------------------------------------------')
    print(f'plotting learning curves for time_stamp {time_stamp}')
    print('---------------------------------------------')

    #define save directory
    plots_path = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{time_stamp}/plots'
    if not os.path.exists(plots_path):
            os.makedirs(plots_path)
    
    #obtain confusion matrix entries for each epoch (trn and val data), and compute corresponding metrics

    dicsCMloss_trn_path = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{time_stamp}/classCM_losses_trn.json'
    TP_trn, FP_trn, TN_trn, FN_trn, lbox_trn, lobj_trn, n_ep_trn = get_CM_loss_dic(dicsCMloss_trn_path)
    acc_trn, prec_trn, rec_trn, f1_trn, spec_trn, bal_acc_trn = get_metrics(TP_trn, FP_trn, TN_trn, FN_trn)

    if not noval:
        dicsCMloss_val_path = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{time_stamp}/classCM_losses_val.json'
        TP_val, FP_val, TN_val, FN_val, lbox_val, lobj_val, n_ep_val = get_CM_loss_dic(dicsCMloss_val_path)
        assert(n_ep_trn==n_ep_val)
        acc_val, prec_val, rec_val, f1_val, spec_val, bal_acc_val = get_metrics(TP_val, FP_val, TN_val, FN_val)
    if noval:
        acc_val, prec_val, rec_val, f1_val, spec_val, bal_acc_val, lbox_val, lobj_val, n_ep_val = [], [], [], [], [], [], [], [], []
         
    baseline_metrics_path = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{time_stamp}/baseline_metrics.json'
    with open(baseline_metrics_path, 'r') as file:
        baseline_metrics_dic = json.load(file)
    
    #plot and save metric and loss curves
        
    plot_and_save(n_ep_trn,metric_ls_trn=acc_trn, metric_ls_val=acc_val, metric_name='accuracy', text_config=text_config, min_max='max', save_dir=plots_path, time_stamp=time_stamp, baseline_dic=baseline_metrics_dic,noval=noval)
    plot_and_save(n_ep_trn,metric_ls_trn=prec_trn, metric_ls_val=prec_val, metric_name='precision', text_config=text_config, min_max='max', save_dir=plots_path, time_stamp=time_stamp, baseline_dic=baseline_metrics_dic,noval=noval)
    plot_and_save(n_ep_trn,metric_ls_trn=rec_trn, metric_ls_val=rec_val, metric_name='recall', text_config=text_config, min_max='max', save_dir=plots_path, time_stamp=time_stamp, baseline_dic=baseline_metrics_dic,noval=noval)
    plot_and_save(n_ep_trn,metric_ls_trn=f1_trn, metric_ls_val=f1_val, metric_name='f1_score', text_config=text_config, min_max='max', save_dir=plots_path, time_stamp=time_stamp, baseline_dic=baseline_metrics_dic,noval=noval)
    plot_and_save(n_ep_trn,metric_ls_trn=spec_trn, metric_ls_val=spec_val, metric_name='specificity', text_config=text_config, min_max='max', save_dir=plots_path, time_stamp=time_stamp, baseline_dic=baseline_metrics_dic,noval=noval)
    plot_and_save(n_ep_trn,metric_ls_trn=bal_acc_trn, metric_ls_val=bal_acc_val, metric_name='balanced_accuracy', text_config=text_config, min_max='max', save_dir=plots_path, time_stamp=time_stamp, baseline_dic=baseline_metrics_dic,noval=noval)

    plot_and_save(n_ep_trn,metric_ls_trn=lbox_trn, metric_ls_val=lbox_val, metric_name='box_loss', text_config=text_config, min_max='min', save_dir=plots_path, time_stamp=time_stamp, baseline_dic=baseline_metrics_dic,noval=noval)
    plot_and_save(n_ep_trn,metric_ls_trn=lobj_trn, metric_ls_val=lobj_val, metric_name='obj_loss', text_config=text_config, min_max='min', save_dir=plots_path, time_stamp=time_stamp, baseline_dic=baseline_metrics_dic,noval=noval)










