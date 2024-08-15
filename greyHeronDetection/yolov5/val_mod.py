# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
##########################(modified)##########################
sys.path.append('../..')
from greyHeronClassification.analysis.analysis_utils import plotAndSaveCM
from greyHeronDetection.framework_pwl.classification_metrics.classification_metrics_utils import get_conf_tsh
##########################(modified)##########################


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)








##########################(modified)##########################
#This function has been modified from its original version to allow for computation and saving of classification metrics
@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold (for suppression, NOT evaluation!)
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
    conf_tsh_eval = '0.5', #for evaluation, but if conf_tsh_eval == 'best', set conf_tsh_eval = conf_tsh_stats (maximizes F1 score)
    save_path='none'
):
    
    if conf_tsh_eval != 'best':
        conf_tsh_eval = float(conf_tsh_eval)

    device = select_device(device, batch_size=batch_size)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) #load model
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        print('ENGINE IS TRUE!')
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")
    data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    if pt and not single_cls:  # check --weights are trained on --data
        ncm = model.model.nc
        assert ncm == nc, (
            f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
            f"classes). Pass correct combination of --weights and --data that are trained together."
        )
    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
    task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=rect,
        workers=workers,
        prefix=colorstr(f"{task}: "),
    )[0]

    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    stats = []
    stats_per_im = []
    for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        #NMS: non_max_suppresion removes excessive prediction boxes below confidence threshold or with too much overlap (IoU > iou_threshold), taking the one with maximum confidence
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        preds = non_max_suppression(preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det) 

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            """print(path)
            print(labels)"""
    
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                stats_per_im.append({'correct': correct,
                                     'conf': [],
                                     'pred_cls': [],
                                     'true_cls': labels[:, 0]})
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls) ###pcls is predicted class, tcls is true class
            stats_per_im.append({'correct': correct,
                                     'conf': pred[:, 4],
                                     'pred_cls': pred[:, 5],
                                     'true_cls': labels[:, 0]})

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    mf1, map, map50 = 0, 0, 0 #added in case len(stats) and stats[0].any() == False
    conf_tsh_stats = 0 #added in case len(stats) and stats[0].any() == False
    if len(stats) and stats[0].any():
        #ap_per_class computes metrics, plots precison & recall & precision-recall curves (at IoU 0.5), and returns tp, fp, p, r, f1 at IoU 0.5 (for each conf tsh), and ap (for each conf tsh and IoU)
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names) # ap_class is array of containing all classes
        conf_tsh_stats = get_conf_tsh(*stats)
        #ap at 0.50 and mean over 0.50-0.95
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        #mean accross different classes
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        mf1 = f1.mean()
    if conf_tsh_eval == 'best':
            conf_tsh_eval = conf_tsh_stats


    #compute confusion matrix
            
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    #loop over each image
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

    CM_loss = {'TN': int(TN),  'FP': int(FP), 'FN': int(FN), 'TP': int(TP)}#, 
    CM_ls = [[CM_loss['TN'],CM_loss['FP']],
            [CM_loss['FN'],CM_loss['TP']]]
    tot_neg = CM_loss['TN']+CM_loss['FP']
    tot_pos = CM_loss['FN']+CM_loss['TP']

    eps=1e-8  #prevent division by zero
    CM_norm_ls = [[CM_loss['TN']/(tot_neg+eps),CM_loss['FP']/(tot_neg+eps)],
            [CM_loss['FN']/(tot_pos+eps),CM_loss['TP']/(tot_pos+eps)]]
    
    #compute classification metrics
    acc = (TP+TN)/(TP+FP+TN+FN+eps)
    prec = TP/(TP+FP+eps)
    rec = TP/(TP+FN+eps)
    f1_s = 2*(prec*rec)/(prec+rec+eps)
    spec = TN/(TN+FP+eps)
    bal_acc = (rec+spec)/2

    print('evaluating best model on dataset')
    
    print('\n')
    print('-------------------detection metrics-------------------')
    print(f'mean precision: {mp}')
    print(f'mean recall: {mr}')
    print(f'f1 score: {mf1}')
    print(f'map@0.5: {map50}')
    print(f'map@0.5:0.95: {map}')
    print('\n')

    print('-------------------classification metrics-------------------')
    print(f'accuracy: {acc}')
    print(f'precision: {prec}')
    print(f'recall: {rec}')
    print(f'f1 score: {f1_s}')
    print(f'specificity: {spec}')
    print(f'balanced acc: {bal_acc}')
    print(f'conf_tsh_eval used here is {conf_tsh_eval}')
    print(f'conf_tsh_stats obtained here is {conf_tsh_stats}')
    print('\n')

    #save results

    if save_path != 'none':
        output_dir = f'{save_path}'
        det_metrics = {'mean_precision': str(mp), 'mean_recall': str(mr), 'f1_score': str(mf1), 'map@0.5': str(map50), 'map@0.5:0.95': str(map), 'conf_tsh_stats': conf_tsh_stats} ################
        class_metrics = {'accuracy': str(acc), 'precision': str(prec), 'recall': str(rec), 'f1_score': str(f1_s), 'specificity': str(spec), 'balanced_accuracy': str(bal_acc), 'conf_tsh_eval': conf_tsh_eval}

        print('saving metrics...')

        os.makedirs(output_dir,exist_ok=True)

        with open(f'{output_dir}class_confusion_matrix.json', 'w') as json_file:
            json.dump(CM_loss, json_file)
        with open(f'{output_dir}detection_metrics.json', 'w') as json_file:
            json.dump(det_metrics, json_file)
        with open(f'{output_dir}classification_metrics.json', 'w') as json_file:
            json.dump(class_metrics, json_file)
        plotAndSaveCM(CM_norm_ls,save_path+'confusion_matrix_norm.png','Normalized Confusion Matrix')
        
        print('done saving metrics')
        print('\n')

    model.float() # for training
    
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist())
##########################(modified)##########################


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")

    ##########################(modified)##########################
    parser.add_argument("--save-path", type=str, default='none', help="directory where to save results")
    parser.add_argument("--conf-tsh-eval", type=str, default='0.5', help="confidence threshold for evaluation metrics")
    ##########################(modified)##########################

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
