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

from model.model_class import modelClass
from data_loader.data_loader import dataLoader
from utils.train_utils import train_one_epoch, evaluate_one_epoch, get_data_and_labels, oversample_data, undersample_data, get_baseline_metrics, load_data_torch, save_dictionary, log_oversample_pos


def trainAndEvaluate(config):
    """
    Function to train and validate model on specified dataset
    Args:
        config (dictionary): contains list of evaluation configurations, for options see trainAndAnalyze_job.sh
            'torch_seed' (int): seed used for pytorch
            'num_workers' (int): number of workers used for data loading
            'parent_dir (str): parent directory
            'image_size' (int): image resolution
            'ls_cams' (list): list of filtered cameras
            'num_classes' (int): number of classes
            'pretrained_network' (str): pretrained model
            'learning_rate' (float): leraning rate
            'batch_size' (int): batch size
            'weight_decay' (float): weight decay for learning step
            'dropout_prob' (float): dropout probability
            'which_weights' (str): unfrozen weights from last or all layers
            'n_epochs (int): number of training epochs
            'time_stamp' (str): time stamp of training job
            'n_last_im' (int): number of last images for background removal; if 'none', then original images are taken
            'day_night' (str): day or night-time images, or both
            'augs' (str): transformations for data augmentation
            'resample_trn' (str): resample method for training
            'n_cams_regroup' (int): number of regrouped cameras during log oversampling
            'val_full' (bool): evaluate model on full validation set after training
            'trn_val' (bool): train with both training anf validation data merged
            'which_val' (str): which dataset to use for validation
            'split' (str): first ('chronological') or second ('seasonal') split used
    """

    seed=config['torch_seed']
    num_workers = config['num_workers']
    parent_dir=config['parent_dir']
    image_size = config['image_size']
    ls_cams_filt = config['ls_cams']
    num_classes = config['num_classes']
    pretrained_network = config['pretrained_network']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    weight_decay = config['weight_decay']
    dropout_prob = config['dropout_prob']
    which_weights = config['which_weights']
    num_epochs = config['n_epochs']
    timestamp = config['time_stamp']
    n_last_im = config['n_last_im']
    day_night = config['day_night']
    augs = config['augs']
    resample_trn = config['resample_trn']
    n_cams_regroup = config['n_cams_regroup']
    val_full = bool(config['val_full'])
    trn_val = bool(config['trn_val'])
    which_val = config['which_val']
    split = config['split']

    if which_val=='none':
        noval=True
    else:
        noval=False

    print('configurations: ',config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(seed)

    #------------------------------------------------------------------------------------------------------------

    #define metrics being used
    accuracy = Accuracy(task="binary")
    precision = Precision(task="multiclass", num_classes=2, average=None)
    recall = Recall(task="multiclass", num_classes=2, average=None)
    f1_score = F1Score(task="multiclass", num_classes=2, average=None)
    confusion_matrix = ConfusionMatrix(task="binary", num_classes=2)
    metric_names = ["accuracy", "precision","recall", "f1_score", "confusion_matrix"]
    metrics = [accuracy,precision,recall,f1_score,confusion_matrix]

    #path to metics an checkpoints
    model_metrics_path = parent_dir+'greyHeronClassification/logs/metrics/'
    model_checkpoint_path = parent_dir+'greyHeronClassification/logs/checkpoints/'

    #add tranfsormations for augmenation
    add_transforms_trn = []
    if 'colorjitter' in augs:
        print('colorjitter being used as augmentation')
        add_transforms_trn+=[transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.9)]

    #------------------------------------------------------------------------------------------------------------

    #load data

    if not trn_val:
        mode_trn='trn'
    elif trn_val:
        mode_trn='trn_val'
    train_data_loader = load_data_torch(
        split=split,
        ls_cams_filt=ls_cams_filt,
        parent_dir=parent_dir,
        n_last_im=n_last_im,
        day_night=day_night,
        resample=resample_trn,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        add_transforms=add_transforms_trn,
        n_cams_regroup=n_cams_regroup,
        mode=mode_trn
    )
    ls_images_trn = train_data_loader.dataset.img_paths
    ls_labels_trn = train_data_loader.dataset.labels
    n_samples_trn = len(ls_labels_trn)
    n_pos_trn = ls_labels_trn.count(1)
    
    if not noval:
        add_transforms_val=[]
        if which_val=='val':
            mode_val='val'
        if which_val=='tst':
            mode_val='tst'
        val_data_loader = load_data_torch(
            split=split,
            ls_cams_filt=ls_cams_filt,
            parent_dir=parent_dir,
            n_last_im=n_last_im,
            day_night=day_night,
            resample='undersample',
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            add_transforms=add_transforms_val,
            mode=mode_val
        )
        ls_images_val = val_data_loader.dataset.img_paths
        ls_labels_val = val_data_loader.dataset.labels
        n_samples_val = len(ls_labels_val)
        print(f'number of val samples {n_samples_val}')
        n_pos_val = ls_labels_val.count(1)

        val_data_loader_full = load_data_torch(
            split=split,
            ls_cams_filt=ls_cams_filt,
            parent_dir=parent_dir,
            n_last_im=n_last_im,
            day_night=day_night,
            resample='no_resample',
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            add_transforms=add_transforms_val,
            mode=mode_val
        )
    elif noval:
        val_data_loader=[]
        ls_images_val =[]
        ls_labels_val = []
        n_samples_val = len(ls_labels_val)
        print(f'number of val samples {n_samples_val}')
        n_pos_val = 0

    #------------------------------------------------------------------------------------------------------------

    #create model
    model = modelClass(
        pretrained_network=pretrained_network, 
        num_classes=num_classes, 
        which_weights=which_weights, #all or last
    )
   
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            old_p = module.p
            module.p = dropout_prob
            print(f"dropout probability in layer {name} changed from {old_p} to {module.p}")
    
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #-------------------------computing and saving baseline metrics-------------------------------
    if split=='chronological':
        split_name=''
    elif split=='seasonal':
        split_name='_split2'

    if not trn_val:
        path_trn=f'data/csv_files/dataSDSC_trn{split_name}.csv'
        ls_images_trn_imb, ls_labels_trn_imb = get_data_and_labels(path_trn,ls_cams_filt,parent_dir,n_last_im,day_night)
    elif trn_val:
        path_trn_1=f'data/csv_files/dataSDSC_trn{split_name}.csv'
        path_trn_2=f'data/csv_files/dataSDSC_val{split_name}.csv'
        ls_images_trn_imb_1, ls_labels_trn_imb_1 = get_data_and_labels(path_trn_1,ls_cams_filt,parent_dir,n_last_im,day_night)
        ls_images_trn_imb_2, ls_labels_trn_imb_2 = get_data_and_labels(path_trn_2,ls_cams_filt,parent_dir,n_last_im,day_night)
        ls_images_trn_imb = ls_images_trn_imb_1+ls_images_trn_imb_2
        ls_labels_trn_imb = ls_labels_trn_imb_1+ls_labels_trn_imb_2
    if not noval:
        if which_val == 'val':
            path_val=f'data/csv_files/dataSDSC_val{split_name}.csv'
        if which_val == 'tst':
            path_val=f'data/csv_files/dataSDSC_tst{split_name}.csv'
        ls_images_val_imb, ls_labels_val_imb = get_data_and_labels(path_val,ls_cams_filt,parent_dir,n_last_im,day_night)
    elif noval:
        ls_images_val_imb, ls_labels_val_imb = [], []

    ls_images_complete = ls_images_trn_imb + ls_images_val_imb
    ls_labels_complete = ls_labels_trn_imb + ls_labels_val_imb
    metrics_all_neg = get_baseline_metrics('all_neg',ls_labels_complete,metrics)
    metrics_all_pos = get_baseline_metrics('all_pos',ls_labels_complete,metrics)
    metrics_50_50 = get_baseline_metrics('50_50',ls_labels_complete,metrics)
    metrics_class_pctg = get_baseline_metrics('class_pctg',ls_labels_complete,metrics)
    pctg_pos = ls_labels_complete.count(1)/len(ls_labels_complete)
    pctg_neg = ls_labels_complete.count(0)/len(ls_labels_complete)

    baseline_metrics_dic = [{'metric_names': metric_names, 
                             'metrics_all_neg': metrics_all_neg, 'metrics_all_pos': metrics_all_pos, 
                             'metrics_50_50': metrics_50_50, 'metrics_class_pctg': metrics_class_pctg,
                            'pctg_pos': pctg_pos, 'pctg_neg': pctg_neg}]

    #---------------sending model, data and metrics to gpu (cuda)-------------------

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for gpu_id in range(num_gpus):
            gpu_properties = torch.cuda.get_device_properties(gpu_id)
            gpu_size = gpu_properties.total_memory / (1024 ** 3)  # Convert bytes to gigabytes
            print(f"GPU {gpu_id} - Name: {gpu_properties.name}, Memory Size: {gpu_size:.2f} GB")
    else:
        print("No GPU available. Please ensure you have a compatible GPU and have installed the appropriate drivers.")
        torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    for metric in metrics:
        metric.to(torch.device(device))

    #----------------------------------training----------------------------------

    if resample_trn[:14]!='oversample_dyn':
        evaluate_one_epoch_trn = evaluate_one_epoch
        train_one_epoch_trn = train_one_epoch
        train_data_loader_main=train_data_loader
        train_data_loader_second=None

    text_data_size = f'training with '+\
          f'n_samples {n_samples_trn} (n_pos: {n_pos_trn}) and '+\
          f'evaluating with n_samples {n_samples_val} (n_pos: {n_pos_val}), '+\
          f'resampling trn: {resample_trn})'
    print(text_data_size)


    from datetime import datetime
    print(f"time stamp is {timestamp}")
    best_vloss = 1e6
    best_vf1score = -1
    saved_list = []


    print('---------------------------------------------------')

    #uncomment to save loss and metrics at epoch 0

    """
    print("EPOCH 0")
    avg_loss, avg_class_loss, metric_vals, \
        ls_avg_loss, ls_avg_class_loss, \
            ls_high_loss_0, ls_wrong_class_0, ls_vmetric_vals_trn_0,  \
                 ls_zipped_pathLabPredProb_full_trn0 = evaluate_one_epoch_trn(
            validation_loader=train_data_loader_main,
            validation_loader_second=train_data_loader_second,
            loss_fn=loss_fn, 
            model=model,
            metrics=metrics,
            device=device
        )
    avg_vloss, avg_class_vloss, vmetric_vals, \
        ls_avg_vloss, ls_avg_class_vloss, ls_vmetric_vals_val_0, \
            ls_high_loss_0, ls_wrong_vclass_0, \
                 ls_zipped_pathLabPredProb_full_val0 = evaluate_one_epoch(
            validation_loader=val_data_loader,
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            device=device
        )
    saved_list.append({'trn_loss': avg_loss, 'val_loss': avg_vloss, 
                       'trn_class_loss': avg_class_loss, 'val_class_loss': avg_class_vloss, 
                       'metric_names': metric_names, 
                       'trn_metrics': metric_vals, 'val_metrics': vmetric_vals, 
                       'epoch_number': 0,
                       'trn_loss_batch_0': ls_avg_loss,
                       'trn_class_loss_batch_0': ls_avg_class_loss,})
    """
    
    ('---------------------------------------------------')

    t_ep = time.time()
    #start training loop
    for epoch_number in range(num_epochs):
        torch.cuda.empty_cache()
        print('EPOCH {}:'.format(epoch_number + 1))
        model.train(True)
        avg_loss, avg_class_loss, metric_vals, \
            ls_avg_loss, ls_avg_class_loss, ls_metric_vals, ls_vmetric_vals, \
                ls_avg_loss_step0, ls_avg_class_loss_step0, \
                     ls_high_loss, ls_wrong_class, \
                         ls_zipped_pathLabPredProb_trn = train_one_epoch_trn(
            epoch_index=epoch_number,
            num_epochs=num_epochs,
            training_loader=train_data_loader_main,
            optimizer=optimizer, 
            loss_fn=loss_fn, 
            model=model,
            metrics=metrics,
            device=device,
            validation_loader=val_data_loader
        )
        if not noval:
            avg_vloss, avg_class_vloss, vmetric_vals, \
                ls_avg_vloss, ls_avg_class_vloss, ls_vmetric_vals_unused, \
                    ls_high_vloss, ls_wrong_vclass, \
                        ls_zipped_pathLabPredProb_val = evaluate_one_epoch(
                validation_loader=val_data_loader,
                model=model,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device
            )
            
        if noval:
            avg_vloss, avg_class_vloss, vmetric_vals, ls_high_vloss, ls_wrong_vclass, ls_zipped_pathLabPredProb_val = 'noval', 'noval', 'noval', 'noval', 'noval', 'noval'
        del_t_ep=time.time()-t_ep
        print('Losses: trn {}, val {}'.format(avg_loss, avg_vloss, epoch_number+1))

        dic_current_save = {'trn_loss': avg_loss, 'val_loss': avg_vloss, 
                    'trn_class_loss': avg_class_loss, 'val_class_loss': avg_class_vloss, 
                    'metric_names': metric_names, 
                    'trn_metrics': metric_vals, 'val_metrics': vmetric_vals, 
                    'epoch_number': epoch_number+1,
                    'trn_loss_batch': ls_avg_loss,
                    'trn_class_loss_batch': ls_avg_class_loss,
                    'trn_metrics_batch': ls_metric_vals, 'val_metrics_batch': ls_vmetric_vals,
                    'trn_loss_batch_step0': ls_avg_loss_step0,
                    'trn_class_loss_batch_step0': ls_avg_class_loss_step0,
                    'steps_train': np.linspace(epoch_number,epoch_number+1,num=len(ls_avg_loss),endpoint=False),
                    't_ep': del_t_ep,
                    'ls_high_loss': ls_high_loss,
                    'ls_high_vloss': ls_high_vloss,
                    'ls_wrong_class': ls_wrong_class,
                    'ls_wrong_vclass': ls_wrong_vclass}
        saved_list.append(dic_current_save)            
        
        if not noval:
            index_vf1score = dic_current_save['metric_names'].index('f1_score')
            current_vf1score_pos = dic_current_save['val_metrics'][index_vf1score][1]

            #taking best model with max f1
            if current_vf1score_pos > best_vf1score:
                best_vf1score = current_vf1score_pos
                best_vf1score_epoch_number = epoch_number+1
                best_f1_wrong_class = ls_wrong_class
                best_f1_wrong_vclass = ls_wrong_vclass
                best_vloss = avg_vloss
                best_model_path = model_checkpoint_path+'best_model_weightsAndMetrics_'+str(timestamp)+".pth"
                best_model_dic = copy.deepcopy(model).state_dict()
                best_avg_loss = avg_loss
                best_avg_vloss = avg_vloss
                best_avg_class_loss = avg_class_loss
                best_avg_class_vloss = avg_class_vloss
                best_metric_names = metric_names
                best_metric_vals = metric_vals
                best_vmetric_vals = vmetric_vals
                best_epoch_number = epoch_number+1
                best_pathLabPredProb_trn = ls_zipped_pathLabPredProb_trn
                best_pathLabPredProb_val = ls_zipped_pathLabPredProb_val

        print(f'total_time per epoch: {del_t_ep}')
        t_ep = time.time()
        print('---------------------------------------------------')
    #finish training loop    

    #--------------------------------------------------------------------------------------------------

    #evaluating best model on full validation data
        
    if val_full:
        if not noval:
            best_model = modelClass(
                pretrained_network=pretrained_network, 
                num_classes=num_classes, 
                which_weights=which_weights, #all or last
            )
            new_state_dict = {k.replace('module.', ''): v for k, v in best_model_dic.items()}
            #print(new_state_dict.keys())
            best_model.load_state_dict(new_state_dict)
            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs!")
                best_model = nn.DataParallel(best_model)
            best_model.to(device)
            avg_vloss_, avg_class_vloss_, vmetric_vals_full, \
                    ls_avg_vloss_, ls_avg_class_vloss_, ls_vmetric_vals_unused, \
                        ls_high_vloss_, ls_wrong_vclass_, \
                            ls_zipped_pathLabPredProb_val_full = evaluate_one_epoch(
                    validation_loader=val_data_loader_full,
                    model=best_model,
                    loss_fn=loss_fn,
                    metrics=metrics,
                    device=device
                )
            dic_best_metrics={
                'metric_names': metric_names,
                'best_metrics_val_undersampled': best_vmetric_vals,
                'best_metrics_val_full': vmetric_vals_full
            }
        elif noval:
            vmetric_vals_full = 'noval'

    else:
        vmetric_vals_full = 'no_val_full'
        vmetric_vals_full = 'no_val_full'
        ls_zipped_pathLabPredProb_val_full = 'no_val_full'
    
    #--------------------------------------------------------------------------------------------------

    #saving checkpoints & dictionaries
    if not noval:
        dic_best = {'model': best_model_dic, 
                    'trn_loss': best_avg_loss, 'val_loss': best_avg_vloss, 
                    'trn_class_loss': best_avg_class_loss, 'val_class_loss': best_avg_class_vloss, 
                    'metric_names': best_metric_names, 
                    'trn_metrics': best_metric_vals, 'val_metrics': best_vmetric_vals, 
                    'val_metrics_full': vmetric_vals_full,
                    'epoch_number': best_epoch_number+1, 
                    'trn_wrong_class': best_f1_wrong_class, 'val_wrong_class': best_f1_wrong_vclass,
                    'zipped_pathLabProbPred_trn': best_pathLabPredProb_trn, 
                    'zipped_pathLabProbPred_val': best_pathLabPredProb_val,
                    'zipped_pathLabProbPred_val_full': ls_zipped_pathLabPredProb_val_full}
        torch.save(dic_best, best_model_path) #best model save
        
    dic_last = {'model': model.state_dict(), 
                'trn_loss': avg_loss, 'val_loss': avg_vloss, 
                'trn_class_loss': avg_class_loss, 'val_class_loss': avg_class_vloss, 
                'metric_names': metric_names, 
                'trn_metrics': metric_vals, 'val_metrics': vmetric_vals, 
                'epoch_number': epoch_number+1, 
                'trn_wrong_class': ls_wrong_class, 'val_wrong_class': ls_wrong_vclass,
                'zipped_pathLabPredProb_trn': ls_zipped_pathLabPredProb_trn, 
                'zipped_pathLabPredProb_val': ls_zipped_pathLabPredProb_val}
    torch.save(dic_last, model_checkpoint_path+'last_model_weightsAndMetrics_'+str(timestamp)+".pth") #last model save
    
    torch.save(saved_list, model_metrics_path+'model_metrics_allEps_{}'.format(timestamp)+'.pth')
    torch.save(baseline_metrics_dic, model_metrics_path+'model_metrics_baseline_{}'.format(timestamp)+'.pth')

    print(f'time_stamp: {timestamp}')
    if not noval:
        print(f'model performance on val dataset for metrics {best_metric_names}: '+\
            f'{best_vmetric_vals}, and loss {best_avg_vloss} on epoch {best_epoch_number}')

    print('######################## TRAINING DONE ############################')

    #-------------------------------------------------------------------------------------------------------
    
    #saving info about configs and data

    dir_output = config['parent_dir']+"greyHeronClassification/analysis/output/"+config['time_stamp']
    save_dictionary(config, dir_output, 'configurations.txt')
    os.makedirs(dir_output, exist_ok=True)
    file_path = dir_output+"/data_info.txt"
    with open(file_path, "w") as file:
        # Write the string to the file
        file.write(text_data_size)
    
    ls_files_trn = [image[-22:] for image in ls_images_trn]
    zipped_lists_trn = list(zip(ls_images_trn, ls_files_trn, ls_labels_trn))
    # Specify the file name
    csv_file_name_trn = f'{dir_output}/ls_trn.csv'
    # Write to CSV file
    with open(csv_file_name_trn, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(['Full Path', 'File Name', 'Label'])
        # Write data
        csv_writer.writerows(zipped_lists_trn)
    
    ls_files_val = [image[-22:] for image in ls_images_val]
    zipped_lists_val = list(zip(ls_images_val, ls_files_val, ls_labels_val))
    # Specify the file name
    csv_file_name_val = f'{dir_output}/ls_val.csv'
    # Write to CSV file
    with open(csv_file_name_val, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(['Full Path', 'File Name', 'Label'])
        # Write data
        csv_writer.writerows(zipped_lists_val)
    
    if val_full and not noval:
        save_dictionary(dic_best_metrics, dir_output, 'best_metrics.txt')


