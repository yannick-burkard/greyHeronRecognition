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
from model.model_class import modelClass
import copy
import os
import json
import sys
from imblearn.over_sampling import SMOTE
sys.path.append('..')
from data_loader.data_loader import dataLoader
from data_loader.custom_dataset import CustomDataset

def train_one_epoch(
        epoch_index,
        num_epochs, 
        training_loader,
        optimizer, 
        loss_fn, 
        model,
        metrics,
        device,
        validation_loader=None,
        loss_threshold=1.,
    ):
    """
    Function to train one full epoch (while keeping track of various metrics and losses)
    Args:
        epoch_index (int): index of the current epoch
        num_epochs (int): total number of traning epochs 
        training_loader (pytorch object): DataLoader used for training
        optimizer (pytorch object): optimizer used for training
        loss_fn (pytorch object): loss function used for training
        model (pytorch object): model being trained
        metrics (list): list metrics to track performance
        device (str): device (CPU/GPU) on which training is run
        validation_loader (pytorch object): DataLoader used for validation
        loss_threshold (float): threshold for considering high loss samples
    Returns (quantities computed on training_loader unless otherwise specified):
        avg_loss (float): average loss after entire epoch
        [avg_loss_neg, avg_loss_pos] ([float, float]): average loss for negative and positive class
        metric_vals (list): list of metric values after entire epoch
        ls_avg_loss (list): list of average losses per batch after corresponding training step (non-cummulative)
        [ls_avg_loss_neg, ls_avg_loss_pos] ([list, list]): list of average losses for batch after the corresponding training step for only negative and positive class (non-cummulative)
        ls_metric_vals (list): list of metric values for all metric computation steps (cummulative)
        ls_vmetric_vals (list): list of metric values for all metric computation steps on validation_loader (non-cummulative)
        ls_avg_loss_step0 (list): list of average losses per batch for untrained model
        [ls_avg_loss_neg_step0, ls_avg_loss_pos_step0] ([list, list]): list of average losses per batch for untrained model for untrained model, for only negative and positive class
        ls_zipped_high_loss (list): list of (zipped) losses, paths, and labels for samples with high loss
        ls_wrong_class (list): list of (ziped) paths, labels, probability scores for missclassified samples
        ls_zipped_pathLabPredProb_full (list): list of (zipped) paths, labels, predicitons and probaiblity scores for all samples
    """

    #initialize variables and lists

    running_loss = 0.
    running_loss_neg = 0.
    running_loss_pos = 0.
    nsamples=0
    nsamples_neg = 0
    nsamples_pos = 0
    for metric in metrics:
        metric.reset()
    t0 = time.time()
    i0 = 0

    ls_avg_loss = []
    ls_avg_loss_neg = []
    ls_avg_loss_pos = [] 
    ls_metric_vals = []
    ls_vmetric_vals = []
    
    ls_avg_loss_step0 = []
    ls_avg_loss_neg_step0 = []
    ls_avg_loss_pos_step0 = []

    ls_losses=np.array([])
    ls_paths=np.array([])
    ls_labels=np.array([])

    ls_wrong_class = [] #(path,true_label,probability)

    ls_probabilities_full = np.array([])
    ls_predictions_full = np.array([])
    ls_labels_full = np.array([])
    ls_paths_full = np.array([])

    #define untrained model
    if epoch_index == 0:
        untrained_model = modelClass(pretrained_network='mobilenet', num_classes=2)
        untrained_model.to(device)
        
    #metrics per batch (training and validation loader)
    metrics_batch = []
    for metric in metrics:
        metric_batch = metric.clone()
        metric_batch.reset()
        metrics_batch.append(metric_batch)
    vmetrics_batch = []
    for metric in metrics:
        vmetric_batch = metric.clone()
        vmetric_batch.reset()
        vmetrics_batch.append(vmetric_batch)
    
    #start training step loop
    for i, batch in enumerate(training_loader):

        images, labels, paths = batch
        
        #sending to gpu
        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)
    
        #computing outputs, loss and metrics in eval mode
        model.eval()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(probabilities, dim=1)
        for metric in metrics:
            metric.update(predictions, labels)
        nsamples+=len(images)
        model.train(True)

        #appending paths, labels, probabilities, predictions
        probabilities_np_1 = probabilities.cpu().detach().numpy()[:,1]
        paths_np = np.array(paths)
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        ls_probabilities_full = np.concatenate((ls_probabilities_full,probabilities_np_1))
        ls_predictions_full = np.concatenate((ls_predictions_full,predictions_np))
        ls_paths_full = np.concatenate((ls_paths_full,paths_np))
        ls_labels_full = np.concatenate((ls_labels_full,labels_np))
        
        #extract images with high loss (> loss_threshold)
        loss_np = loss.cpu().detach().numpy()
        paths_np = np.array(paths)
        labels_np = labels.cpu().numpy()
        indices_filt = np.where(loss_np>loss_threshold)
        ls_losses = np.concatenate((ls_losses, loss_np[indices_filt]))
        ls_paths = np.concatenate((ls_paths, paths_np[indices_filt]))
        ls_labels = np.concatenate((ls_labels, labels_np[indices_filt]))

        #training step
        optimizer.zero_grad()
        outputs_trn = model(images)
        loss_trn = loss_fn(outputs_trn, labels).sum()
        loss_trn.backward()
        optimizer.step()
        running_loss += loss.sum().item()

        #computing loss per class
        labels_neg, outputs_neg, labels_pos, outputs_pos = split_pos_neg_batch(labels,outputs)
        current_loss_neg = None
        current_loss_pos = None
        if labels_neg != []:
            current_loss_neg = loss_fn(outputs_neg, labels_neg).sum().item()
            running_loss_neg += current_loss_neg
        nsamples_neg += len(labels_neg)
        if labels_pos != []:
            current_loss_pos = loss_fn(outputs_pos, labels_pos).sum().item()
            running_loss_pos += current_loss_pos
        nsamples_pos += len(labels_pos)

        #extracting missclassified images
        probabilities_np = probabilities.cpu().detach().numpy()
        predictions_np = predictions.cpu().numpy()
        mismatch_ind = np.where(predictions_np != labels_np)[0]
        wrong_paths_np = paths_np[mismatch_ind]
        wrong_labels_np = labels_np[mismatch_ind]
        wrong_probabilities_np = probabilities_np[mismatch_ind]
        pathLabelProb = list(zip(wrong_paths_np,wrong_labels_np,wrong_probabilities_np))
        ls_wrong_class+=pathLabelProb

        #appending losses / metrics per batch
        ls_avg_loss.append(loss.sum().item()/len(labels))
        batch_loss_neg = None
        batch_loss_pos = None
        if current_loss_pos != None:
            batch_loss_pos = current_loss_pos/len(labels_pos)
        if current_loss_neg != None:
            batch_loss_neg = current_loss_neg/len(labels_neg)
        ls_avg_loss_neg.append(batch_loss_neg)
        ls_avg_loss_pos.append(batch_loss_pos)
        for metric_batch in metrics_batch:
            metric_batch.update(predictions, labels)

        #metrics computation step: done after n training steps (now n = int(num_epochs*len(training_loader)/60))
        if i%(int(num_epochs*len(training_loader)/60))==0:
            ls_metric_batch = []
            for metric_batch in metrics_batch:
                ls_metric_batch.append(metric_batch.clone().compute())
            ls_metric_vals.append(ls_metric_batch)
            
            with torch.no_grad():
                ls_vmetric_batch = []
                for vmetric_batch in vmetrics_batch:
                    vmetric_batch.reset()
                for j, vbatch in enumerate(validation_loader):
                    model.eval()
                    vimages, vlabels, paths = vbatch
                    if torch.cuda.is_available():
                        vimages = vimages.to(device)
                        vlabels = vlabels.to(device)
                    voutputs = model(vimages)
                    vprobabilities = torch.softmax(voutputs, dim=1)
                    _, vpredictions = torch.max(vprobabilities, dim=1)
                    for vmetric_batch in vmetrics_batch:
                        vmetric_batch.update(vpredictions, vlabels)
            model.train()

            if len(validation_loader)!=0:
                ls_vmetric_batch = []
                for vmetric_batch in vmetrics_batch:
                    ls_vmetric_batch.append(vmetric_batch.clone().compute())
                ls_vmetric_vals.append(ls_vmetric_batch)
            else:
                ls_vmetric_vals.append('noval')

        #computing loss and loss per class for each batch 
        if i==0:
            model_step0 = copy.deepcopy(model)
        loss_step0 = loss_fn(model_step0(images),labels).sum().item()
        ls_avg_loss_step0.append(loss_step0/len(labels))
        labels_neg, images_neg, labels_pos, images_pos = split_pos_neg_batch(labels,images)
        if labels_neg != []:
            loss_neg_step0 = loss_fn(model_step0(images_neg),labels_neg).sum().item()
            ls_avg_loss_neg_step0.append(loss_neg_step0/len(labels_neg))
        if labels_pos != []:
            loss_pos_step0 = loss_fn(model_step0(images_pos),labels_pos).sum().item()
            ls_avg_loss_pos_step0.append(loss_pos_step0/len(labels_pos))

        if i!=0 and (i % 100 == 0 or i == len(training_loader)-1):
            print(f"time taken for last {i-i0} batches is {time.time()-t0}s")
            print(f"processing batch {i+1} with {len(images)} samples in epoch {epoch_index+1}")
            i0 = i
            t0 = time.time()
    #finish training step loop

    eps=1e-8
    avg_loss = running_loss / (nsamples+eps)
    avg_loss_neg = running_loss_neg / (nsamples_neg+eps)
    avg_loss_pos = running_loss_pos / (nsamples_pos+eps)

    print(f'trn loss: {avg_loss}, class loss: {[avg_loss_neg, avg_loss_pos]}')
    print(f'tot n samples: {nsamples}, class n samples: {[nsamples_neg, nsamples_pos]}')

    metric_vals = []
    for metric in metrics:
        metric_vals.append(metric.compute().tolist())

    ls_zipped_high_loss = list(zip(ls_losses,ls_paths,ls_labels))
    ls_zipped_pathLabPredProb_full = list(zip(ls_paths_full,ls_labels_full,ls_predictions_full,ls_probabilities_full))

    return avg_loss, [avg_loss_neg, avg_loss_pos], metric_vals, \
                ls_avg_loss, [ls_avg_loss_neg, ls_avg_loss_pos], ls_metric_vals, ls_vmetric_vals, \
                    ls_avg_loss_step0, [ls_avg_loss_neg_step0, ls_avg_loss_pos_step0], \
                        ls_zipped_high_loss, ls_wrong_class, \
                            ls_zipped_pathLabPredProb_full



def evaluate_one_epoch(
        validation_loader,
        model,
        loss_fn,
        metrics,
        device,
        loss_threshold=1.,
    ):

    """
    Function to evaluate one full epoch (while keeping track of various metrics and losses)
    Args:
        loss_fn (pytorch object): loss function used for training
        model (pytorch object): model being trained
        metrics (list): list metrics to track performance
        device (str): device (CPU/GPU) on which training is run
        validation_loader (pytorch object): DataLoader used for validation
        loss_threshold (float): threshold for considering high loss samples
    Returns (quantities computed on validation_loader unless otherwise specified):
        avg_vloss (float): average loss after entire epoch
        [avg_vloss_neg, avg_vloss_pos] ([float, float]): average loss for negative and positive class
        metric_vals (list): list of metric values after entire epoch
        ls_avg_vloss (list): list of average losses per batch after corresponding training step (non-cummulative)
        [ls_avg_vloss_neg, ls_avg_vloss_pos] ([float, float]): list of average losses for batch after the corresponding training step for only negative and positive class (non-cummulative)
        ls_metric_vals (list): list of metric values after each batch loop (cummulative)
        ls_zipped_high_loss (list): list of (zipped) losses, paths, and labels for samples with high loss
        ls_wrong_class (list): list of (ziped) paths, labels, probability scores for missclassified samples
        ls_zipped_pathLabPredProb_full (list): list of (zipped) paths, labels, predicitons and probaiblity scores for all samples
    """

    #initialize variables and lists

    running_vloss = 0.0
    running_vloss_neg = 0.
    running_vloss_pos = 0.

    model.eval()
    vnsamples=0
    vnsamples_neg=0
    vnsamples_pos=0

    ls_avg_vloss = []
    ls_avg_vloss_neg = []
    ls_avg_vloss_pos = [] 
    ls_metric_vals = []

    ls_losses=np.array([])
    ls_paths=np.array([])
    ls_labels=np.array([])

    ls_wrong_class = []

    ls_probabilities_full = np.array([])
    ls_predictions_full = np.array([])
    ls_labels_full = np.array([])
    ls_paths_full = np.array([])

    for metric in metrics:
        metric.reset()
        metric.to(device)
    
    #metrics per batch
    metrics_batch = []
    for metric in metrics:
        metric_batch = metric.clone()
        metric_batch.reset()
        metrics_batch.append(metric_batch)

    with torch.no_grad():
        #start validation loop
        for i, vbatch in enumerate(validation_loader):

            vimages, vlabels, paths = vbatch

            if torch.cuda.is_available():
                vimages = vimages.to(device)
                vlabels = vlabels.to(device)

            #evaluation step (outputs, loss and metrics)
            voutputs = model(vimages)
            vloss = loss_fn(voutputs, vlabels)
            vprobabilities = torch.softmax(voutputs, dim=1)
            _, vpredictions = torch.max(vprobabilities, dim=1)
            running_vloss += vloss.sum().item()
            for metric in metrics:
                metric.update(vpredictions, vlabels)
            vnsamples+=len(vimages)

            #appending paths, labels, probabilities, predictions
            probabilities_np_1 = vprobabilities.cpu().numpy()[:,1]
            paths_np = np.array(paths)
            labels_np = vlabels.cpu().numpy()
            predictions_np = vpredictions.cpu().numpy()
            ls_probabilities_full = np.concatenate((ls_probabilities_full,probabilities_np_1))
            ls_predictions_full = np.concatenate((ls_predictions_full,predictions_np))
            ls_paths_full = np.concatenate((ls_paths_full,paths_np))
            ls_labels_full = np.concatenate((ls_labels_full,labels_np))

            #extract images with high loss (> loss_threshold)
            loss_np = vloss.cpu().numpy()
            paths_np = np.array(paths)
            labels_np = vlabels.cpu().numpy()
            indices_filt = np.where(loss_np>loss_threshold)
            ls_losses = np.concatenate((ls_losses, loss_np[indices_filt]))
            ls_paths = np.concatenate((ls_paths, paths_np[indices_filt]))
            ls_labels = np.concatenate((ls_labels, labels_np[indices_filt]))

            #computing loss per class
            vlabels_neg, voutputs_neg, vlabels_pos, voutputs_pos = split_pos_neg_batch(vlabels,voutputs)
            current_vloss_neg = None
            current_vloss_pos = None
            if vlabels_neg != []:
                current_vloss_neg = loss_fn(voutputs_neg, vlabels_neg).sum().item()
                running_vloss_neg += current_vloss_neg
            vnsamples_neg += len(vlabels_neg)
            if vlabels_pos != []:
                current_vloss_pos = loss_fn(voutputs_pos, vlabels_pos).sum().item()
                running_vloss_pos += current_vloss_pos
            vnsamples_pos += len(vlabels_pos)

            #appending losses / metrics per batch
            ls_avg_vloss.append(vloss.sum().item()/len(vlabels))
            batch_vloss_neg = None
            batch_vloss_pos = None
            if current_vloss_pos != None:
                batch_vloss_pos = current_vloss_pos/len(vlabels_pos)
            if current_vloss_neg != None:
                batch_vloss_neg = current_vloss_neg/len(vlabels_neg)
            ls_avg_vloss_neg.append(batch_vloss_neg)
            ls_avg_vloss_pos.append(batch_vloss_pos)
            ls_metric_batch = []
            for metric_batch in metrics_batch:
                metric_batch.update(vpredictions, vlabels)
                ls_metric_batch.append(metric_batch.compute())
            ls_metric_vals.append(ls_metric_batch)

            #extracting missclassified images
            probabilities_np = vprobabilities.cpu().numpy()
            predictions_np = vpredictions.cpu().numpy()
            mismatch_ind = np.where(predictions_np != labels_np)[0]
            wrong_paths_np = paths_np[mismatch_ind]
            wrong_labels_np = labels_np[mismatch_ind]
            wrong_probabilities_np = probabilities_np[mismatch_ind]
            pathLabelProb = list(zip(wrong_paths_np,wrong_labels_np,wrong_probabilities_np))
            ls_wrong_class+=pathLabelProb
        #finish validation loop
    
    eps=1e-8
    avg_vloss = running_vloss / (vnsamples+eps)
    avg_vloss_neg = running_vloss_neg / (vnsamples_neg+eps)
    avg_vloss_pos = running_vloss_pos / (vnsamples_pos+eps)

    print('----')
    print(f'val loss: {avg_vloss}, class loss: {[avg_vloss_neg, avg_vloss_pos]}')
    print(f'tot n samples: {vnsamples}, class n samples: {[vnsamples_neg, vnsamples_pos]}')
    print('----')

    metric_vals = []
    for metric in metrics:
        metric_vals.append(metric.compute().tolist())

    ls_zipped_high_loss = list(zip(ls_losses,ls_paths,ls_labels))
    ls_zipped_pathLabPredProb_full = list(zip(ls_paths_full,ls_labels_full,ls_predictions_full,ls_probabilities_full))

    return avg_vloss, [avg_vloss_neg, avg_vloss_pos], metric_vals, \
        ls_avg_vloss, [ls_avg_vloss_neg, ls_avg_vloss_pos], ls_metric_vals, \
            ls_zipped_high_loss, ls_wrong_class, \
                ls_zipped_pathLabPredProb_full










def get_data_and_labels(csv_path,ls_cams_filt,parent_dir,n_last_im,day_night):
    """
    Function to extract sample paths and labels from input csv file
    Args:
        csv_path (str): path leading to csv file listing dataset
        ls_cams_filt (list): list of cameras to be filtered
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
    ls_images = [parent_dir+'data/dataset/'+ls_images_SIAM[i][35:] for i in range(len(ls_images_SIAM))] #adapted
    ls_labels = [file_label for _, _, file_label in sorted_image_datesPathsLabels]
    if len(n_last_im)<4:
        print(f"Using images with bkg removed using last {n_last_im}")
        prefix='dataDriveMichele_noBkg/dataDriveMichele_noBkg448_'
        ls_images=[f'{parent_dir}data/dataset/{prefix}{n_last_im}/SBU4/{image[-22:]}' for image in ls_images[(n_last_im+2):]] #adapted
        ls_labels = ls_labels[(n_last_im+2):]
    else:
        print(f"Using images with no bkg removed")
    print(f'len of paths & labels: {len(ls_images)} and {len(ls_labels)}')

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

    n_samples_min = ls_labels.count(min_label)

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






def split_pos_neg_batch(labels,outputs):
    """
    Function to split between positive and negative classes
    Args:
        labels (pytorch class): labels to split
        outputs (pytorch class): model outputs to split
    Returns:
        labels_neg (pytorch class): negative labels
        outputs_neg (pytorch class): negative ouputs
        labels_pos (pytorch class): positive labels
        outputs_pos (pytorch class): positive ouputs
    """

    zipped_data = list(zip(labels,outputs))
    zipped_neg = [(label, output) for label, output in zipped_data if label == 0]
    zipped_pos = [(label, output) for label, output in zipped_data if label == 1]
    labels_neg, outputs_neg, labels_pos, outputs_pos = [], [], [], []
    if zipped_pos != []:
        labels_pos, outputs_pos  = zip(*zipped_pos)
        labels_pos = torch.stack(labels_pos)
        outputs_pos = torch.stack(outputs_pos)
    if zipped_neg != []:
        labels_neg, outputs_neg  = zip(*zipped_neg)
        labels_neg = torch.stack(labels_neg)
        outputs_neg = torch.stack(outputs_neg)

    return labels_neg, outputs_neg, labels_pos, outputs_pos





def get_baseline_metrics(which_baseline,ls_labels,ls_metrics):
    """
    Function to get metrics based on baseline model:
        Args:
            which_baseline (str): which baseline model to use, options are
                'all_neg' (all negative), 'all_pos' (all positive), '50_50' (random 50/50 guessing), 'class_pctg' (random class pctg guessing)
            ls_labels (list): list of image labels
            ls_metrics (list): list of metrics used
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
    for i, metric in enumerate(ls_metrics):
        metric.reset()
        metric.update(predictions,torch.tensor(ls_labels))
        metric_value = metric.compute().tolist()
        ls_metric_values.append(metric_value)
    
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

def load_data_torch(split,
                    ls_cams_filt,
                    parent_dir,
                    n_last_im,
                    day_night,
                    resample,
                    image_size,
                    batch_size,
                    num_workers,
                    add_transforms,
                    mode,
                    n_cams_regroup=0
                    ):

    """
    Function to load data via torch by specifying a series of atributes
    Args:
        split (str): first (chronological) or second (seasonal) split used
        ls_cams_filt (list): list of filtered cameras
        parent_dir (str): parent directory used
        n_last_im (int): number of last images for background removal; if 'none', then original images are taken
        day_night (str): day or night images, or both
        resample (str): resampling method used (can be 'undersample', 'oversample_naive', 'log_oversample', 'log_oversample_2', 'oversample_smote', 'no_resample')
        image_size (int): image resolution
        batch_size (int): batch size
        num_workers (int): number of worked for torch data loading
        add_transforms (list): additional transformations, corresponing to data augmentation
        mode (str): training ('trn'), validation ('val'), or both ('trn_val'), or test ('tst')
        n_cams_regroup (int): number of regrouped cameras for log_oversampling_2
    Returns:
        data_loade (pytorch object): DataLoader object containing data
    """

    if split=='chronological':
        split_name=''
    elif split=='seasonal':
        split_name='_split2'

    if mode=='trn':
        path_csv=f'data/csv_files/dataSDSC_trn{split_name}.csv'
    if mode=='val':
        path_csv=f'data/csv_files/dataSDSC_val{split_name}.csv'
    if mode=='tst':
        path_csv=f'data/csv_files/dataSDSC_tst{split_name}.csv'

    
        
    assert(type(mode)==str)
    assert(type(resample)==str)
    if mode!='trn_val':
        ls_images_imb, ls_labels_imb = get_data_and_labels(path_csv,ls_cams_filt,parent_dir,n_last_im,day_night)
    if mode=='trn_val':
        path_csv_1=f'data/csv_files/dataSDSC_trn{split_name}.csv'
        path_csv_2=f'data/csv_files/dataSDSC_val{split_name}.csv'
        ls_images_imb_1, ls_labels_imb_1 = get_data_and_labels(path_csv_1,ls_cams_filt,parent_dir,n_last_im,day_night)
        ls_images_imb_2, ls_labels_imb_2 = get_data_and_labels(path_csv_2,ls_cams_filt,parent_dir,n_last_im,day_night)
        ls_images_imb = ls_images_imb_1+ls_images_imb_2
        ls_labels_imb = ls_labels_imb_1+ls_labels_imb_2
    if resample=='undersample':
        print('undersampling')
        ls_images, ls_labels = undersample_data(ls_images_imb, ls_labels_imb)
    if resample=='oversample_naive':
        ls_images, ls_labels = oversample_data(ls_images_imb, ls_labels_imb)
    if resample == 'log_oversample':
        ls_images_tmp, ls_labels_tmp = log_oversample_pos(ls_images_imb, ls_labels_imb, ls_cams_filt)
        ls_images, ls_labels = oversample_data(ls_images_tmp, ls_labels_tmp)
    if resample == 'log_oversample_2':
        ls_images_tmp, ls_labels_tmp = log_oversample_pos_2(ls_images_imb, ls_labels_imb, ls_cams_filt, n_cams_regroup)
        ls_images, ls_labels = oversample_data(ls_images_tmp, ls_labels_tmp)
    if resample == 'no_resample':
            ls_images, ls_labels = ls_images_imb, ls_labels_imb

    data_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size))] +
        add_transforms +
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(len(ls_images))
    dataset = CustomDataset(
        ls_images,
        ls_labels,
        transforms=data_transforms,
        imsize=image_size)
    print(len(dataset.img_paths))
    
    if resample=='oversample_smote' and mode=='trn':
        smote = SMOTE(random_state=42)
        X_train = dataset.get_all_pixel_values()
        y_train = ls_labels
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_os_flat, y_os = smote.fit_resample(X_train_flat, y_train)
        X_os = X_os_flat.reshape(X_os_flat.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3])
        y_add = len(X_os[len(X_train):])*['smote_data']
        ls_paths_os = ls_images+y_add

        dataset_full = CustomDataset(
            ls_images=ls_paths_os,
            ls_labels=y_os,
            ls_pixels=X_os,
            transforms=[],
            imsize=image_size)
        
        data_loader = DataLoader(
            dataset_full,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True, 
            persistent_workers=True
        )
        
    if not(resample=='oversample_smote' and mode=='trn'):
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return data_loader


def save_dictionary(data, directory, filename):
    """save data as dictionary on path directory/filename"""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

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











