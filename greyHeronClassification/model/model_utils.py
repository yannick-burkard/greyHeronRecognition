from torchvision import models, transforms
from pathlib import Path
import torch.nn as nn
import numpy as np

def read_pretrained_model(architecture, n_class, which_weights):
    """
    Helper function to load models compactly from pytorch model zoo and prepare them for finetuning
    Args:
        architecture (str): string of model name
        n_class (int): number of classes in final layer
        which_weights (str): unfrozen weights from last ('last') or all ('all') layers
    Returns:
        model (pytorch class): pytorch model
    """
    
    architecture = architecture.lower()
        
    if architecture == "mobilenet":

        model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=n_class, bias=True) #important to access the last layer

        for i, param in enumerate(model.features.parameters()):
            if which_weights == 'last':
                param.requires_grad = False
            elif which_weights == 'all':
                param.requires_grad = True
        for param in model.classifier[1].parameters(): #here one could in principle remove the [1] since it yields the same parameters (there are no parameters for dropout in classifier[0])
            param.requires_grad = True

    return model

