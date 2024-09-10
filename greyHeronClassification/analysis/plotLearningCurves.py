import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import sys
import json
import glob

current_path = os.getcwd()
if current_path[-8:]=='analysis':
     sys.path.append('..')
from analysis.analysis_utils import get_metric_list, get_loss_steps_batch, get_metrics_per_batch_and_ep

def plot_learning_curves(parent_dir,time_stamp):
    """
    Function to plot learing curves (loss and metrics) after a training procedure:
    Args:
        parent_dir (str): parent dictory
        time_stamp (str): time_stamp for training procedure
    """
    
    #load configs and initialize variables
    config_path=f'{parent_dir}greyHeronClassification/analysis/output/{time_stamp}/configurations.txt'
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    parent_dir=config['parent_dir']+''
    bs = config['batch_size']
    lr = config['learning_rate']
    wd = config['weight_decay']
    dp = config['dropout_prob']
    ww = config['which_weights']
    seed = config['torch_seed']
    mn = config['pretrained_network']
    li = config['n_last_im']
    dn = config['day_night']
    im_size = config['image_size']
    rs_trn = config['resample_trn']
    aug_trn = config['augs']
    text_config = f'model {mn} w/ {ww} weights, bs:{bs}, lr:{lr}, wd:{wd}, dp:{dp}, s:{seed}, d/n:{dn}, li:{li}, is: {im_size}, rst: {rs_trn}, aug: {aug_trn}'


    #define wheter to include validation results or not
    which_val = config['which_val']
    if which_val=='none':
        noval=True
    else:
        noval=False

    print('---------------------------------------------')
    print(f'plotting learning curves for time_stamp {time_stamp}')

    results_path = parent_dir + 'greyHeronClassification/analysis/output/' + time_stamp
    if not os.path.exists(results_path):
            os.makedirs(results_path)

    #load dictonaries
    dics_metrics = torch.load(f'{parent_dir}greyHeronClassification/logs/metrics/model_metrics_allEps_{time_stamp}.pth',map_location=torch.device('cpu'))

    epochs_ls = np.arange(0,len(dics_metrics))+1
    metric_names = ['loss','pos_class_loss','neg_class_loss','accuracy','precision','recall','f1_score','specificity','balanced_accuracy']

    #load metrics for best model (max f1 score)
    if not noval:
        file_path = f'{parent_dir}greyHeronClassification/logs/checkpoints/best_model_weightsAndMetrics_{time_stamp}.pth'
        dic_best = torch.load(file_path,map_location=torch.device('cpu'))
        index_f1= dic_best['metric_names'].index('f1_score')
        f1__best_model = dic_best['val_metrics'][index_f1][1]
        index_acc = dic_best['metric_names'].index('accuracy')
        acc_best_model = dic_best['val_metrics'][index_acc]
        index_rec= dic_best['metric_names'].index('recall')
        rec_best_model = dic_best['val_metrics'][index_rec][1]
        index_prec = dic_best['metric_names'].index('precision')
        prec_best_model = dic_best['val_metrics'][index_prec][1]
        index_spec = dic_best['metric_names'].index('recall')
        spec_best_model = dic_best['val_metrics'][index_spec][0]
        
    #generate plots
    for metric in metric_names:
        print(f'plotting {metric}')

        if metric == 'loss':
            steps_train = get_loss_steps_batch(dics_metrics,'steps_train','trn')
            loss_batch_trn = get_loss_steps_batch(dics_metrics,'loss_batch','trn')
            plt.plot(steps_train, loss_batch_trn,label='trn (per batch)',color='tab:blue',alpha=0.2)

        if metric == 'pos_class_loss':
            steps_train = get_loss_steps_batch(dics_metrics,'steps_train','trn')
            pos_loss_batch_trn_N = get_loss_steps_batch(dics_metrics,'pos_class_loss_batch','trn')
            pos_loss_batch_trn = [pos_loss_batch_trn_N[i] for i in range(len(pos_loss_batch_trn_N)) if pos_loss_batch_trn_N[i] is not None]
            steps_train = [steps_train[i] for i in range(len(steps_train)) if pos_loss_batch_trn_N[i] is not None]
            plt.plot(steps_train, pos_loss_batch_trn,label='trn (per batch)',color='tab:blue',alpha=0.2)

        if metric == 'neg_class_loss':
            
            steps_train = get_loss_steps_batch(dics_metrics,'steps_train','trn')
            neg_loss_batch_trn_N = get_loss_steps_batch(dics_metrics,'neg_class_loss_batch','trn')
            neg_loss_batch_trn = [neg_loss_batch_trn_N[i] for i in range(len(neg_loss_batch_trn_N)) if neg_loss_batch_trn_N[i] is not None]
            steps_train = [steps_train[i] for i in range(len(steps_train)) if neg_loss_batch_trn_N[i] is not None]
            plt.plot(steps_train, neg_loss_batch_trn,label='trn (per batch)',color='tab:blue',alpha=0.2)

        if metric in ['accuracy','precision','recall','f1_score','specificity','balanced_accuracy']:
            metric_per_batch_trn, steps_per_batch_trn = get_metrics_per_batch_and_ep(dics_metrics,metric,'trn')
            plt.plot(steps_per_batch_trn,metric_per_batch_trn,label='trn (per batch)',color='tab:blue',alpha=0.4)
            if not noval:
                metric_per_batch_val, steps_per_batch_val = get_metrics_per_batch_and_ep(dics_metrics,metric,'val')
                plt.plot(steps_per_batch_val,metric_per_batch_val,label='val (per batch)',color='tab:orange',alpha=0.4)


        metric_ls_trn = get_metric_list(dics_metrics,metric,'trn')
        if not noval:
            metric_ls_val = get_metric_list(dics_metrics,metric,'val')
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.15)
        plt.scatter(epochs_ls, metric_ls_trn,label='trn',color='tab:blue',marker='x')
        if not noval:
            plt.scatter(epochs_ls, metric_ls_val,label='val',color='tab:orange',marker='x')
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('Epoch Number')
        plt.ylabel(metric)
        plt.title(metric)
        plt.figtext(0.01, 0.01, text_config, ha='left', va='bottom', fontsize=5)
        if not noval:
            if metric=='loss':
                min_loss = np.round(float(min(metric_ls_val)),2)
                plt.figtext(0.99, 0.99, f'min val loss: {min_loss}', ha='right', va='top', fontsize=5)
            if metric=='f1_score':
                max_f1score = np.round(float(max(metric_ls_val)),2)
                plt.figtext(0.99, 0.99, f'max val f1 score: {max_f1score}', ha='right', va='top', fontsize=5)
            if metric=='precison':
                plt.figtext(0.99, 0.99, f'val precision best model: {prec_best_model}', ha='right', va='top', fontsize=5)
            if metric=='accuracy':
                plt.figtext(0.99, 0.99, f'val acc best model: {acc_best_model}', ha='right', va='top', fontsize=5)
            if metric=='recall':
                plt.figtext(0.99, 0.99, f'val recall best model: {rec_best_model}', ha='right', va='top', fontsize=5)
            if metric=='specificity':
                plt.figtext(0.99, 0.99, f'val specificity best model: {spec_best_model}', ha='right', va='top', fontsize=5)
        plt.savefig(f'{results_path}/plot_{metric}VsEpoch_{time_stamp}')
        plt.close()

    print('---------------------------------------------')
