import torch
import numpy as np
import os
import json



def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def getClassBboxConfTsh(filename):
    """input is label txt file in YOLOv5 format, returns classes, Bboxes, conf scores"""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        classes = []
        bboxes = []
        confs = []
        for line in lines:
            line_ls = line.strip().split()
            label = line_ls[0]
            bbox = [float(value) for value in line_ls[1:-1]]
            conf = float(line_ls[-1])
            classes.append(label)
            bboxes.append(bbox)
            confs.append(conf)
        return np.array(classes), np.array(bboxes), np.array(confs)
    except FileNotFoundError:
        print("File not found.")
        return None

def save_dictionary(data, directory, filename):
    """save data as dictionary on path directory/filename"""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
