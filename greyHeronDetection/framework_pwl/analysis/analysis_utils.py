import json
import matplotlib.pyplot as plt
import numpy as np




def get_CM_loss_dic(dic_path):
    """
    Function to load confusion matrix entries for each epoch
    Args:
        dic_path (str): path of .json file containg confusion matrix for all epochs after a training procedure
    Returns:
        np.array(TP) (numpy.array): array containing object true positive values
        np.array(FP) (numpy.array): array containing object false positive values
        np.array(TN) (numpy.array): array containing object true negative values
        np.array(FN) (numpy.array): array containing object false negative values
        np.array(lbox) (numpy.array): array containing box loss values
        np.array(lobj) (numpy.array): array containing object loss values
        n_ep (list): list containing ordered epoch numbers
    """

    TP, FP, TN, FN, lbox, lobj = [], [], [], [], [], []
    n_ep = 0
    n_ep_ls = []
    with open(dic_path, "r") as file:
        for line in file:
            n_ep+=1
            dic = json.loads(line)
            TP.append(dic['TP'])
            FP.append(dic['FP'])
            TN.append(dic['TN'])
            FN.append(dic['FN'])
            lbox.append(dic['box_loss'])
            lobj.append(dic['obj_loss'])
            if 'epoch' in dic.keys():
                n_ep_ls.append(dic['epoch'])
    
    if len(n_ep_ls)!=0:
        n_ep=n_ep_ls
    print('n_ep',n_ep)
    print('n_ep_ls',n_ep_ls)

    return np.array(TP), np.array(FP), np.array(TN), np.array(FN), np.array(lbox), np.array(lobj), n_ep

def get_metrics(TP, FP, TN, FN):
    """
    Function to compute metric values
    Args (all of type numpy.array):
        TP, FP, TN, FN: true and false positives, true and false negatives
    Returns (all of type numpy.array):
        acc, prec, rec, f1, spec, bal_acc: accuracy, precision, recall, F1-score, specificity and balanced accuracy values
    """
    eps=1e-8
    acc = (TP+TN)/(TP+FP+TN+FN+eps)
    prec = TP/(TP+FP+eps)
    rec = TP/(TP+FN+eps)
    f1 = 2*(prec*rec)/(prec+rec+eps)
    spec = TN/(TN+FP+eps)
    bal_acc = (rec+spec)/2
    return acc, prec, rec, f1, spec, bal_acc

def plot_and_save(epochs_ls, metric_ls_trn, metric_ls_val, metric_name, text_config, min_max, save_dir, time_stamp, baseline_dic, include_baseline = False, noval=False):
    """
    Function to plot and save metric or loss curves
    Args:
        epochs_ls (list or int): list containing order epoch numbers, or int specifying total number of epochs
        metric_ls_trn (list): list containing metric (or loss) values for training data
        metric_ls_val (list): list containing metric (or loss) values for validation data
        metric_name (str): name of metric
        text_config (str): text containing relevant configuration, is displayed on plot
        min_max (str): print minmum ('min') or maximum ('maximum') metric (or loss) value
        save_dir (str): directory  in which plot file is saved
        time_stamp (str): timestamp corresponding to training procedure
        baseline_dic (dictionary): dictionary containing baseline metrics
        include_baseline (bool): whether to include baseline results or not
        noval (bool): validation data not included
    """

    print(f'plotting {metric_name}')

    #extract two max baseline metrics that are < 1
    bl_keys = baseline_dic.keys()
    if metric_name in baseline_dic['metric_names']:
        idx = baseline_dic['metric_names'].index(metric_name)
        max_bl_val_1, max_bl_name_1 = 0, ''
        max_bl_val_2, max_bl_name_2 = 0, ''
        for key in bl_keys:
            if key != 'metric_names':
                current_val = float(baseline_dic[key][idx])
                current_name = key
                if current_val > max_bl_val_1 and current_val<1:
                    max_bl_val_2, max_bl_name_2 = max_bl_val_1, max_bl_name_1
                    max_bl_val_1, max_bl_name_1 = current_val, current_name
                elif current_val > max_bl_val_2 and current_val<1:
                    max_bl_val_2, max_bl_name_2 = current_val, current_name
    
    if type(epochs_ls)==int:
        epochs_ls = np.arange(0,len(metric_ls_trn))

    #plot
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.15)
    plt.plot(epochs_ls, metric_ls_trn,label='trn',color='tab:blue')
    if not noval:
        plt.plot(epochs_ls, metric_ls_val,label='val',color='tab:orange')
        if min_max=='min':
            min_value = np.round(float(min(metric_ls_val)),2)
            plt.figtext(0.91, 0.91, f'min : {min_value}', ha='right', va='top', fontsize=6)
        if min_max=='max':
            max_value = np.round(float(max(metric_ls_val)),2)
            plt.figtext(0.91, 0.91, f'max : {max_value}', ha='right', va='top', fontsize=6)
    if include_baseline and metric_name in baseline_dic['metric_names']:
        plt.axhline(y=max_bl_val_1, color='fuchsia', linestyle='--', label=max_bl_name_1)
        plt.axhline(y=max_bl_val_2, color='indigo', linestyle='--', label=max_bl_name_2)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch Number')
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.figtext(0.01, 0.01, text_config, ha='left', va='bottom', fontsize=6)
    plt.savefig(f'{save_dir}/plot_{metric_name}VsEpoch_{time_stamp}.png')
    plt.close()

    print('---------------------------------------------')






















