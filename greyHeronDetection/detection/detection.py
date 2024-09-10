import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import pandas as pd
import time
import glob
import os
import json
import shutil
from datetime import datetime
import yaml
import subprocess
import shutil
import cv2
from detection_utils import xywhn2xyxy, getClassBboxConfTsh, save_dictionary, get_num_gpus
from ultralytics.utils.plotting import Annotator

def detect_config(config_detect):
    """
    Function to apply detection model on specified dataset
    Args:
        config (dictionary): contains list of detection configurations
            'parent_dir' (str): parent directory
            'yolo_path' (str): path of yolov5 repository (relative to parent directory)
            'data_path' (str or list): can be string or list containing directory(ies) and / or file paths (both relative to the parent directory) of the images to be detected
            'model_path' (str): path of model to be applied (relative to parent directory)
            'iou_tsh' (float): IoU threshold for non-max supression, default by yolov5 is 0.45
            'imgsz' (int): image resolution
            'save_dir' (str): path of directory where results should be saved (relative to parent directory)
            'save_im' (bool): save labelled images or not
            'save_crop': save cropped detections or not (yolov5 argument)
            'save_csv': save results to csv file or not (yolov5 argument)
            'conf_tsh' (float or str): specifies confidence threshold; if set to 
                'get_config': extract confidence threshold saved for training configs
                'get_stats': extract confidence thereshold yielding maximum F1 score during training
    """

    print('configs',config_detect)

    parent_dir = config_detect['parent_dir']
    yolo_path = f"{parent_dir}/{config_detect['yolo_path']}"
    data_path = config_detect['data_path']
    conf_tsh = config_detect['conf_tsh']
    iou_tsh =  config_detect['iou_tsh']
    model_path = f"{parent_dir}/{config_detect['model_path']}"
    imgsz = config_detect['imgsz']
    save_dir = f"{parent_dir}/{config_detect['save_dir']}"
    save_im = config_detect['save_im']
    save_crop = config_detect['save_crop']
    save_csv = config_detect['save_csv']
    n_gpus = get_num_gpus()

    #extract conf_tsh_det

    #get model name
    last_slash_idx = model_path.rfind('/')
    last_dot_idx = model_path.rfind('.')
    model_name = model_path[last_slash_idx+1:last_dot_idx]

    #extracting conf tsh in case model is fine-tuned and saved as timestamp
    if conf_tsh=='get_stats' and len(model_name)==15 and ('md' not in model_name): #conf tsh giving max F1 saved in .json file
        dic_path = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{model_name}/dic_max_FitConfEp.json'
        with open(dic_path, 'r') as file:
            dic = json.load(file)
        conf_tsh_det = float(dic['max_conf_tsh_stats'])
        print(f'conf_tsh value obtained from dic_max_FitConfEp.json: {conf_tsh_det}')
    if conf_tsh=='get_config': #specified in training configs
        dic_path = f'{parent_dir}greyHeronDetection/framework_pwl/analysis/output/{model_name}/configurations.txt'
        with open(dic_path, 'r') as file:
            dic = json.load(file)
        conf_tsh_det = float(dic['conf_tsh_fixed'])
        print(f'conf_tsh value obtained from configutations.json: {conf_tsh_det}')
    else:
        conf_tsh_det = conf_tsh


    if type(data_path)==str:
        print('data_path is str')
        if data_path[-4:]=='.JPG':
            ls_images=[f'{parent_dir}/{data_path}']
        else:
            ls_images=glob.glob(f'{parent_dir}/{data_path}/*.JPG')
    elif type(data_path)==list:
        print('data_path is list')
        ls_images=[]
        for path in data_path:
            print('path is',path)
            if path[-4:]=='.JPG':
                ls_images.append(f'{parent_dir}/{path}')
            else:
                ls_images+=glob.glob(f'{parent_dir}/{path}/*.JPG')
            print('ls_images is',ls_images)

    #fine-tuned model
    if 'md' not in model_path:
        n_classes = 1
        class_names = {'0': "grey_heron"}

    #megadetector
    elif 'md' in model_path:
        n_classes = 3
        class_names = {'0': "animal", '1': "person", '2': "vehicle"}

        
    time_stamp_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'current time stamp: {time_stamp_now}')

    #save data to .txt file for yolov5/detect.py input
    output_det = f'{save_dir}/{time_stamp_now}'
    output_yolo = f'{output_det}/output_yolo'
    output_data = f'{output_det}/data'
    if not os.path.exists(output_yolo):
        os.makedirs(output_yolo)
    if not os.path.exists(output_data):
        os.makedirs(output_data)
    save_dictionary(config_detect, output_det, 'configurations_det.json')
    data_txt_path = f'{output_data}/data_det.txt'
    with open(data_txt_path, 'w') as file:
        for item in ls_images:
            file.write("%s\n" % item)
    
    #save job_id
    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ.get('SLURM_JOB_ID')
    else:
        job_id = 'none'
    job_path = f'{output_det}/job_id.txt'
    with open(job_path, "w") as file:
        file.write(job_id)
    
    #detect
    print('starting detection')

    device = ','.join(str(i) for i in range(n_gpus))

    if n_gpus>0:
        print('running with gpus')
        command_detect = [
            'python', 
            '-m', 'torch.distributed.run', 
            '--nproc_per_node', str(n_gpus),
            f"{yolo_path}/detect.py",
            '--source', f"{data_txt_path}", #
            '--weights', model_path, #
            '--imgsz', str(imgsz),
            '--conf-thres', str(conf_tsh_det),
            '--iou-thres', str(iou_tsh),
            '--save-txt',
            '--save-conf',
            '--project', str(output_yolo),
            '--name', '',
            '--exist-ok',
            '--device', device,
        ]
        if save_crop:
            command_detect.append('--save-crop')
        if save_csv:
            command_detect.append('--save-csv')
        subprocess.run(command_detect)

    elif n_gpus==0:
        print('running without gpus')
        command_detect = [
            'python', 
            f"{yolo_path}/detect.py",
            '--source', f"{data_txt_path}",
            '--weights', model_path,
            '--imgsz', str(imgsz),
            '--conf-thres', str(conf_tsh_det),
            '--iou-thres', str(iou_tsh),
            '--save-txt',
            '--save-conf',
            '--project', str(output_yolo),
            '--name', '',
            '--exist-ok'
        ]
        if save_crop:
            command_detect.append('--save-crop')
        if save_csv:
            command_detect.append('--save-csv')
        subprocess.run(command_detect)

    #save labelled images
    if save_im:
        ls_label_paths = glob.glob(f'{output_yolo}/labels/*txt')
        output_labelled = f'{output_det}/detections_labelled'
        if not os.path.exists(output_labelled):
            os.makedirs(output_labelled)
        for im_path in ls_images:
            last_slash_idx = im_path.rfind('/')
            last_dot_idx = im_path.rfind('.')
            im_name = im_path[last_slash_idx+1:last_dot_idx]
            im_label_txt = f'{output_yolo}/labels/{im_name}.txt'
            #save image only if detection has been made
            if im_label_txt in ls_label_paths:
                print('saving labelled',im_path)
                image_cv = cv2.imread(im_path)
                height, width = image_cv.shape[:2]
                classes, bboxes, confs = getClassBboxConfTsh(im_label_txt)
                bboxes_xyxy = xywhn2xyxy(bboxes, w=width, h=height) #xywhn2xyxy
                line_thickness=4
                annotator = Annotator(image_cv, line_width=line_thickness, pil=False)#, example=str(names))
                for i,bbbox_xyxy in enumerate(bboxes_xyxy):
                    annotator.box_label(bbbox_xyxy, f'{class_names[classes[i]]}: {np.round(confs[i],2)}', color=(0,0,255))#, color=colors(c, True))
                im_save = annotator.result()
                output_path = f'{output_labelled}/{im_name}_lab.JPG'
                cv2.imwrite(output_path, im_save)

    print('detection finished!')
    print(f'results saved in {output_det}')


#all paths specified in configs must be relative to parent directory
config_detect = {
    'parent_dir': '/cluster/project/eawag/p05001/repos/greyHeronRecognition',
    'yolo_path': 'greyHeronDetection/yolov5',
    'data_path': 'data/dataset/dataDriveMichele/NEN/2017_NEN1_02090354.JPG',
    'model_path': 'models/detection/md_zenodo_v5b0.0.0.pt',
    'conf_tsh': 0.2,
    'iou_tsh': 0.45,
    'imgsz': 1280,
    'save_dir': 'greyHeronDetection/detection/output',
    'save_im': True,
    'save_crop': True,
    'save_csv': True
    }
detect_config(config_detect)



"""
#example usage:
config_detect = {
    'parent_dir': '/cluster/project/eawag/p05001/civil_service',
    'yolo_path': 'greyHeronDetection/yolov5',
    'data_path': ['greyHeronDetection_tmp/framework_pwl/test_images/SBU3',
                  'greyHeronDetection_tmp/framework_pwl/test_images/SBU2',
                  'dataDriveMichele/NEN/2017_NEN1_02090354.JPG'],
    'model_path': 'greyHeronDetection/saved_models/md_zenodo_v5b0.0.0.pt', ######
    'conf_tsh': 0.2,
    'iou_tsh': 0.45, #NMS IoU tsh (0.45 is default for yolov5)
    'imgsz': 1280,
    'save_dir': 'greyHeronDetection/detections_test',
    'save_im': True,
    'save_crop': True,
    'save_csv': True
    }
detect_config(config_detect)
"""


