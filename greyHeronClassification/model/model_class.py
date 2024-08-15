import os, sys, time, copy
import numpy as np
from pathlib import Path
from PIL import Image
import datetime
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler 
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torchvision import transforms
from model.model_utils import read_pretrained_model

class modelClass(nn.Module):
    """
    modelClass class inherited from nn.Module
    Variables:
        pretrained_network (str): specifies pretrained model
        num_classes (int): number of classes
        which_weights (str): unfrozen weights from last ('last') or all ('all') layers
    """

    def __init__(
        self,
        pretrained_network="densenet161",
        num_classes=2,
        which_weights='last',
    ):
        super().__init__()
        self.architecture = pretrained_network
        self.num_classes = num_classes
        self.which_weights = which_weights

        #define PyTorch model
        self.model = read_pretrained_model(self.architecture, self.num_classes, self.which_weights)

    #---------------------------------------------------------------------------------

    def forward(self, x):
        "forward pass"
        x = self.model(x)
        return F.log_softmax(x, dim=1)
    













