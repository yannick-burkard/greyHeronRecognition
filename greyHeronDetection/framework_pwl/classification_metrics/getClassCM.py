import sys
from pathlib import Path
import torch
import os
import json

from classification_metrics.classification_metrics_utils import non_max_suppression, xywh2xyxy, process_batch, scale_boxes, get_conf_tsh

#codes here inspired on YOLOv5 framework (yolov5/val.py)

def saveClassCMAndLoss(
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=300,
    single_cls=False,
    augment=False,
    save_hybrid=False,
    half=True,
    model=None,
    dataloader=None,
    save_dir=Path(""),
    compute_loss=None,
    mode='val',
    conf_tsh_eval='none',
    epoch=-1
    ):
    """
    Function to obtain and save the classification confusion matrix during YOLOv5 training (yolov5/train.py) at every epoch, and return the optimal confidence threshold
    Args:
        conf_thres (float): confidence threshold used for non-max suppression
        iou_thres (float): IoU threshold used for non-max suppression
        max_det (int): maximum number of detections for non-max suppression
        single_cls (bool): training on single class or not
        augment (bool): apply aumgentations or not
        save_hybrid (bool): save in hybrid format or not
        half (bool): half model or not
        model (pytorch class): specifies model being used,
        dataloader (pytorch class): data loader used,
        save_dir (Path): specifies saving path,
        compute_loss (YOLOv5 object): compute loss function
        mode (str): evaluate model on training ('trn') or validation ('val') data
        conf_tsh_eval (float or str): specifies conf tsh for model evaluation; if set to 'none', then conf tsh maximizing F1-score is taken
    Returns:
        conf_tsh_stats (float): optimal conf tsh maximizing F1-score
    """

    with torch.inference_mode():

        #initialize variables
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  #get model device, PyTorch model
        half &= device.type != "cpu"  #half precision only supported on CUDA
        model.half() if half else model.float()
        model.eval()
        cuda = device.type != "cpu"
        iouv = torch.linspace(0.5, 0.5, 1, device=device)
        niou = iouv.numel()
        loss = torch.zeros(3, device=device)
        stats = []
        stats_per_im = []
        """
        stats_per_im is a list containing one dictionary per image, with following keys
            'correct': list specifying if predicitons are correct or not (size = number of predicitons)
            'conf': list specifying confidence scores (size = number of predicitons)
            'pred_cls': list specifying predicted classes (size = number of predicitons)
            'true_cls': list specifying true classes (size = number of labels)
        """
        #loop over batches
        for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
            """
            #im contain pixel values for every batch sample
            #target contains bounding box labels and coordinates for every batch sample
            #paths contain original image paths for every batch sample
            #contains origial image dimensions for every batch sample
            """

            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  #uint8 to fp16/32
            im /= 255  #0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  #batch size, channels, height, width

            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls
            
            #NMS: non_max_suppresion removes excessive prediction boxes below confidence threshold or with too much overlap (IoU > iou_threshold), taking the one with maximum confidence
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  #scale BBox coordinate pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  #for autolabelling
            preds = non_max_suppression(preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det)

            #loop over images
            for si, pred in enumerate(preds):

                labels = targets[targets[:, 0] == si, 1:] #filters out #[class, xywh (pixel space)] if image has BBoxes, else if it empty it will return empty image
                nl, npr = labels.shape[0], pred.shape[0]  #number of labels, predictions in current image
                path, shape = Path(paths[si]), shapes[si][0] #path and original shape of current image
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  #list of booleans indicating correct (true) or incorrect (false) predictions at diferent IoU's

                if npr == 0: #no predictions made
                    if nl: #labels present
                        stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    #save in dictionary format for later use
                    stats_per_im.append({'correct': correct,
                                     'conf': [],
                                     'pred_cls': [],
                                     'true_cls': labels[:, 0]})
                    continue

                #Predictions
                if single_cls:
                    pred[:, 5] = 0 #forces one class for all predictions
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  #native-space pred (rescale BBox to original image shape)

                # Evaluate
                if nl: #labels present
                    tbox = xywh2xyxy(labels[:, 1:5])  #target boxes
                    scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  #native-space labels (rescale BBox to original image shape)
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)   

                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  #(correct, conf, pcls, tcls)
                #save again in dictionary format for later use
                stats_per_im.append({'correct': correct,
                                     'conf': pred[:, 4],
                                     'pred_cls': pred[:, 5],
                                     'true_cls': labels[:, 0]})

        #Extract optimal conf tsh (max F1)
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        if len(stats) and stats[0].any():
            conf_tsh_stats = get_conf_tsh(*stats)
        else:
            conf_tsh_stats = 0.2

        if conf_tsh_eval=='none':
            conf_tsh=conf_tsh_stats
        else:
            conf_tsh=conf_tsh_eval
        

        #Determine confusio matrix entries

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for stat in stats_per_im:
            pred_conf_all = stat['conf']
            pred_class = stat['pred_cls']
            if len(pred_conf_all)==0:
                pred_conf_0 = []
            if len(pred_conf_all)!=0:
                pred_conf_0 = pred_conf_all[pred_class==0]

            labs_all = stat['true_cls']
            if len(labs_all)==0:
                labs_0=[]
            if len(labs_all)!=0:
                labs_0 = labs_all[labs_all==0]

            if len(pred_conf_0)==0:
                n_pos=0
            if len(pred_conf_0)!=0:
                n_pos = len(pred_conf_0[pred_conf_0 > conf_tsh])
            n_lab = len(labs_0)

            if n_pos > 0 and n_lab > 0:
                TP+=1
            if n_pos > 0 and n_lab == 0:
                FP+=1
            if n_pos == 0 and n_lab > 0:
                FN+=1
            if n_pos == 0 and n_lab == 0:
                TN+=1
        
        box_loss = loss[0] #box, obj, cls
        obj_loss = loss[1]

        CM_loss = {'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN), 
              'box_loss': float(box_loss.item()), 'obj_loss': float(obj_loss.item()),'epoch':epoch}

        output_dir=f'{save_dir}/saved_additional/'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}classCM_losses_{mode}.json'
        

        with open(output_file, 'a') as f:
            json.dump(CM_loss, f)
            f.write('\n')
        
        model.float()
        
        return conf_tsh_stats












def write_dic(dic,save_dir,name):
    """Function to write dictionary dic with name in directory save_dir"""
    output_dir=f'{save_dir}/saved_additional/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}{name}'
    with open(output_file, 'a') as f:
        json.dump(dic, f)
        f.write('\n')
















def get_stats_per_im_batch(
    conf_thres=0.001,  # confidence threshold (for non_max_suppression, so that we have (almost) all predictions for plots/curves)
    iou_thres=0.6,  # NMS IoU threshold (for non_max_suppression, NOT evaluation!)
    max_det=300,  # maximum detections per image
    single_cls=False,  # treat as single-class dataset ###think whether is makes sense to include this
    augment=False,  # augmented inference
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    half=True,  # use FP16 half-precision inference
    model=None,
    dataloader_batch_i=None,
    compute_loss=None
    ):
     
    """
    Function to evaluate model on batch and obtain corresponding statistics and loss during YOLOv5 training (yolov5/train.py)
    For more detialed comments, see function saveClassCMAndLoss above
    Args:
        conf_thres (float): confidence threshold used for non-max suppression
        iou_thres (float): IoU threshold used for non-max suppression
        max_det (int): maximum number of detections for non-max suppression
        single_cls (bool): training on single class or not
        augment (bool): apply aumgentations or not
        save_hybrid (bool): save in hybrid format or not
        half (bool): half model or not
        model (pytorch class): specifies model being used
        dataloader (pytorch class): data loader used
        compute_loss (YOLOv5 object): compute loss function
    Returns:
        stats_per_im (list): list containing one dictionary per batch sample, with following keys
            'correct': list specifying if predicitons are correct or not (size = number of predicitons)
            'conf': list specifying confidence scores (size = number of predicitons)
            'pred_cls': list specifying predicted classes (size = number of predicitons)
            'true_cls': list specifying true classes (size = number of labels)
        loss (YOLOv5 object): object, box and class losses
    """

    with torch.inference_mode():

        device, pt, jit, engine = next(model.parameters()).device, True, False, False
        half &= device.type != "cpu"
        model.half() if half else model.float()
        model.eval()
        cuda = device.type != "cpu"
        iouv = torch.linspace(0.5, 0.5, 1, device=device)
        niou = iouv.numel()
        loss = torch.zeros(3, device=device)
        stats = []
        stats_per_im = []

        (im, targets, paths, shapes) = dataloader_batch_i

        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()
        im /= 255
        nb, _, height, width = im.shape 

        preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)
        loss += compute_loss(train_out, targets)[1]
        
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        preds = non_max_suppression(preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det)

        for si, pred in enumerate(preds):

            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                stats_per_im.append({'correct': correct,
                                    'conf': [],
                                    'pred_cls': [],
                                    'true_cls': labels[:, 0]})
                continue

            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1) 
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0])) 
            stats_per_im.append({'correct': correct,
                                    'conf': pred[:, 4],
                                    'pred_cls': pred[:, 5],
                                    'true_cls': labels[:, 0]})

        model.float()    
        
    return stats_per_im, loss












def compute_and_save_CM_wStats(stats_per_im,conf_tsh_eval,loss,epoch,save_dir,mode):

        """
        Function to compoute and save confusio matrix given evaluation statistics during YOLOv5 training (yolov5/train.py)
        For more detialed comments, see function saveClassCMAndLoss above
        Args:
            stats_per_im (list): list containing one dictionary per batch sample, with following keys
                'correct': list specifying if predicitons are correct or not (size = number of predicitons)
                'conf': list specifying confidence scores (size = number of predicitons)
                'pred_cls': list specifying predicted classes (size = number of predicitons)
                'true_cls': list specifying true classes (size = number of labels)
            conf_tsh_eval (float or str): specifies conf tsh for model evaluation
            loss (YOLOv5 object): object, box and class losses
            epoch (int): epoch index
            save_dir (Path): specifies saving path,
            mode (str): evaluate model on training ('trn') or validation ('val') data,
        """
        
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for stat in stats_per_im:
            pred_conf_all = stat['conf']
            pred_class = stat['pred_cls']
            if len(pred_conf_all)==0:
                pred_conf_0 = []
            if len(pred_conf_all)!=0:
                pred_conf_0 = pred_conf_all[pred_class==0]

            labs_all = stat['true_cls']
            if len(labs_all)==0:
                labs_0=[]
            if len(labs_all)!=0:
                labs_0 = labs_all[labs_all==0]

            if len(pred_conf_0)==0:
                n_pos=0
            if len(pred_conf_0)!=0:
                n_pos = len(pred_conf_0[pred_conf_0 > conf_tsh_eval])
            n_lab = len(labs_0)

            if n_pos > 0 and n_lab > 0:
                TP+=1
            if n_pos > 0 and n_lab == 0:
                FP+=1
            if n_pos == 0 and n_lab > 0:
                FN+=1
            if n_pos == 0 and n_lab == 0:
                TN+=1
        
        box_loss = loss[0]
        obj_loss = loss[1]

        CM_loss = {'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN), 
              'box_loss': float(box_loss.item()), 'obj_loss': float(obj_loss.item()),'epoch':epoch}

        output_dir=f'{save_dir}/saved_additional/'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}classCM_losses_{mode}.json'
        

        with open(output_file, 'a') as f:
            json.dump(CM_loss, f)
        
        



            





