import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import matplotlib.pyplot as plt
from torchvision import transforms

class CustomDataset(Dataset):
    """
    CustomDataset class inherited from Dataset
    Variables:
        ls_images (list): list of image paths
        ls_labels (list): list of image labels
        tranforms (pytorch class): tranformation objects indicating which transormation to be done on the dataset
        imsize (int): image pixel dimensions
        ls_pixels (int): list of pixel values for all images (used for SMOTE)
    """
    def __init__(self, 
                 ls_images,
                 ls_labels,
                 transforms=None,
                 imsize=224,
                 ls_pixels=[]):
        
        self.transforms = transforms
        self.imsize = imsize
        self.img_paths, self.labels = ls_images, ls_labels
        self.index = 0
        self.ls_pixels = ls_pixels
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        if len(self.ls_pixels)==0:
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            to_crop = int(height*100/2448)
            #cropping bottom bar
            image = transforms.functional.crop(img=image,top=0, left=0, height=height - to_crop, width=width)
            if self.transforms:
                image = self.transforms(image)
        if len(self.ls_pixels)!=0:
            image=self.ls_pixels[idx,...]
        return image, label, img_path
            
    def get_all_pixel_values(self):
        all_pixel_values = []
        for idx in range(len(self)):
            """if idx%100==0:
                print(idx)"""
            image, _, _ = self.__getitem__(idx)
            all_pixel_values.append(image)
        return torch.stack(all_pixel_values)
    
    def get_idx_pixel_values(self,idx):
        image, _, _ = self.__getitem__(idx)
        return image


    #--------------------------------------------------------- 

    def imshow(self, inp, label, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.savefig("example_images/example" + str(self.index) + "_" + str(label.item()) + ".jpg")
        self.index = self.index + 1 












































