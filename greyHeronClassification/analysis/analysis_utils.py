import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_metric_list(dics,metric_name,mode):
    """
    Function to extract metrics and loss after each training epoch
    Args:
        dics (list): list of dictionaries containing metric and loss values
        metric_name (str): string specifying name of metric or loss used
        mode (str): training ('trn'), validation ('val') or baseline ('baseline')
    Returns:
        metric_ls (list): list with metric or loss values after each epoch
    """
    
    metric_ls = []
    for dic in dics:
        if metric_name == 'balanced_accuracy':
            if mode == 'trn' or mode == 'val':
                index = dic['metric_names'].index('recall')
                metric_ls.append((dic[mode+'_metrics'][index][0]+dic[mode+'_metrics'][index][1])/2)
            if mode[:8] == 'baseline':
                which_guess = mode[9:]
                index = dic['metric_names'].index('recall')
                metric_ls.append((dic['metrics_'+which_guess][index][0]+dic['metrics_'+which_guess][index][0])/2)
        elif metric_name == 'specificity':
            if mode == 'trn' or mode == 'val':
                index = dic['metric_names'].index('recall')
                metric_ls.append(dic[mode+'_metrics'][index][0])
            if mode[:8] == 'baseline':
                which_guess = mode[9:]
                index = dic['metric_names'].index('recall')
                metric_ls.append(dic['metrics_'+which_guess][index][0])
        else:
            if metric_name == 'loss':
                metric = dic[mode+'_'+metric_name]
                metric_ls.append(metric)
            elif metric_name == 'pos_class_loss':
                metric = dic[mode+'_'+metric_name[4:]][1]
                metric_ls.append(metric)
            elif metric_name == 'neg_class_loss':
                metric = dic[mode+'_'+metric_name[4:]][0]
                metric_ls.append(metric)
            elif mode[:8] == 'baseline':
                which_guess = mode[9:]
                index = dic['metric_names'].index(metric_name)
                metric = dic['metrics_'+which_guess][index]
                if type(metric) == list:
                    metric_ls.append(metric[-1])
                else:
                    metric_ls.append(metric)
            elif metric_name == 'pctg_pos' or metric_name == 'pctg_neg':
                index = dic['metric_names'].index('confusion_matrix')
                CM = np.array(dic[mode+'_metrics'][index])
                tot_samples = np.sum(CM)
                print(f"sum is {tot_samples}")
                if metric_name == 'pctg_neg':
                    class_samples = np.sum(CM[:,0])
                if metric_name == 'pctg_pos':
                    class_samples = np.sum(CM[:,1])
                class_pctg = class_samples/tot_samples
                metric_ls.append(class_pctg)
            else:
                index = dic['metric_names'].index(metric_name)
                metric = dic[mode+'_metrics'][index]
                if type(metric) == list and metric_name != 'specificity':
                    metric_ls.append(metric[-1])
                else:
                    metric_ls.append(metric)
    return metric_ls

def get_loss_steps_batch(dics,loss_steps_str,mode):
    """
    Function to extract loss per batch after each training step, or steps as epoch fractions
    Args:
        dics (list): list of dictionaries containing metric values and losses
        loss_steps_str (str): string specifying wheter steps or loss are being extracted, and which classes included in loss
        mode (str): training ('trn') or validation ('val')
    Returns:
        loss_batch_ls (list): list with losses or epoch fractions per batch after each training step
    """
    loss_batch_ls = []
    for i, dic in enumerate(dics):
        if i == 0:
            if loss_steps_str == 'loss_batch_0':
                loss_batch_ls += dic[mode+'_'+loss_steps_str]
                continue
            elif loss_steps_str == 'pos_class_loss_batch_0':
                ls_metric = dic[mode+'_'+loss_steps_str[4:]][1]
                loss_batch_ls += ls_metric
                continue
            elif loss_steps_str == 'neg_class_loss_batch_0':
                ls_metric = dic[mode+'_'+loss_steps_str[4:]][0]
                loss_batch_ls += ls_metric
                continue
        if i == 1:
            if loss_steps_str == 'loss_batch_step0':
                loss_batch_ls += dic[mode+'_'+loss_steps_str]
            elif loss_steps_str == 'pos_class_loss_batch_step0':
                ls_metric = dic[mode+'_'+loss_steps_str[4:]][1]
                loss_batch_ls += ls_metric
            elif loss_steps_str == 'neg_class_loss_batch_step0':
                ls_metric = dic[mode+'_'+loss_steps_str[4:]][0]
                loss_batch_ls += ls_metric
        if loss_steps_str == 'loss_batch':
            ls_metric = dic[mode+'_'+loss_steps_str]
            loss_batch_ls += ls_metric
        elif loss_steps_str == 'pos_class_loss_batch':
            ls_metric = dic[mode+'_'+loss_steps_str[4:]][1]
            loss_batch_ls += ls_metric
        elif loss_steps_str == 'neg_class_loss_batch':
            ls_metric = dic[mode+'_'+loss_steps_str[4:]][0]
            loss_batch_ls += ls_metric
        elif loss_steps_str == 'steps_train':
            ls_metric = list(dic[loss_steps_str])
            loss_batch_ls  += ls_metric
        elif loss_steps_str[:4] == 'pctg':
            index = dic['metric_names'].index('confusion_matrix')
            for metric_vals in dic[mode+'_metrics_batch']:
                CM = metric_vals[index].numpy()
                tot_samples = np.sum(CM)
                if loss_steps_str == 'pctg_neg_batch':
                    class_samples = np.sum(CM[:,0])
                if loss_steps_str == 'pctg_pos_batch':
                    class_samples = np.sum(CM[:,1])
                if loss_steps_str == 'pctg_neg_curr_batch':
                    class_samples = np.sum(CM[0,:])
                if loss_steps_str == 'pctg_pos_curr_batch':
                    class_samples = np.sum(CM[1,:])
                class_pctg = class_samples/tot_samples
                loss_batch_ls.append(class_pctg)
    return loss_batch_ls


def get_metrics_per_batch_and_ep(dics,metric_name,mode):
    """
    Function to extract metrics and epoch fraction after each metric computation step
    Args:
        dics (list): list of dictionaries containing metric values and losses
        metric_name (str): string specifying name of metric used
        mode (str): training ('trn') or validation ('val')
    Returns:
        metric_batch_ls (list): list with losses per batch after each metric computation step
        n_eps_batch (list): list with epoch fractions corresponding to each metric computation step
    """
    metric_batch_ls = []
    n_eps_batch = []
    for i, dic in enumerate(dics[:]):
        index = dic['metric_names'].index('confusion_matrix')
        CM_trn_all = np.array(dic['trn_metrics'][index])
        tot_samp_trn = np.sum(CM_trn_all)
        for j, metric_vals in enumerate(dic[mode+'_metrics_batch']):
            CM_trn_j = dic['trn_metrics_batch'][j][index].numpy()
            CM = metric_vals[index].numpy()
            current_metric = compute_metric_CM(CM,metric_name)
            frac_ep = i+np.sum(CM_trn_j)/tot_samp_trn
            n_eps_batch.append(frac_ep)
            metric_batch_ls.append(current_metric)
    return metric_batch_ls, n_eps_batch


def compute_metric_CM(CM,metric_name):
    """
    Function to compute metric based on confusion matrix
    Args:
        CM (numpy.array): array corresponing to confusion matrix
        metric_name (str): name of metric being computed
    Returns:
        metric_val (float): computed metric value
    """
    eps=1e-8 #prevent division by zero
    TN, TP = CM[0,0], CM[1,1]
    FP, FN = CM[0,1], CM[1,0]
    metric_val = 0
    if metric_name=='accuracy':
        metric_val = (TP+TN)/(TP+FP+TN+FN+eps)
    if metric_name=='precision':
        metric_val = TP/(TP+FP+eps)
    if metric_name=='recall':
        metric_val = TP/(TP+FN+eps)
    if metric_name=='f1_score':
        prec = TP/(TP+FP+eps)
        rec = TP/(TP+FN+eps)
        metric_val = 2*(prec*rec)/(prec+rec+eps)
    if metric_name=='specificity':
        metric_val = TN/(TN+FP+eps)
    if metric_name=='balanced_accuracy':
        rec = TP/(TP+FN+eps)
        spec = TN/(TN+FP+eps)
        metric_val = (rec+spec)/2
    return metric_val




