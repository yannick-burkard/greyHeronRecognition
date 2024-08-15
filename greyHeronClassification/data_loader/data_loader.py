import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from data_loader.custom_dataset import CustomDataset 

def dataLoader(
        ls_images,
        ls_labels,
        image_size, 
        batch_size,
        num_workers,
        add_transforms
        ):
    """
    Function to load data via CustomDataset and DataLoader, while specifying several attributes
    Args:
        ls_images (list): list of image paths
        ls_labels (list): list of image labels
        image_size (int): image resolution
        batch_size (int): batch size
        num_workers (int): number of workers
        add_transforms (list): list woth additional augmentation, corresponding to augmentation
    Returns
        data_loader (DataLoader): data loader containing data
    """

    #nornalization
    data_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size))] +
        add_transforms +
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    
    dataset = CustomDataset(
        ls_images,
        ls_labels,
        transforms=data_transforms,
        imsize=image_size)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return data_loader


























#----------------------------------(some examples)--------------------------------------------


#NO WEIGHTED DATALOADER SO FAR (NOR SINGLE CAMERA ONE) --> CHECK IF NECESSARY


"""
# Example usage (probably some modifications are needed to ...):
if __name__ == "__main__":
    dir_dict = {
        "negatives": "/path/to/negatives",
        "positives": "/path/to/positives"
    }
    ls_inds = []  # Set your desired ls_inds
    learning_set = "all"  # Specify your learning set
    batch_size = 32
    num_workers = 4  # Adjust the number of workers according to your system's capabilities
    transforms = []  # Add your data transforms here

    data_loader = get_heron_data_loader(
        dir_dict, 
        ls_inds, 
        learning_set, 
        batch_size, 
        num_workers, 
        transforms)

    for batch in data_loader:
        images, labels, indices = batch
        # Perform your training/validation/testing operations here
"""




"""

#more complex with custom datasets
def get_data_loaders(data_dir, 
    batch_size, 
    num_workers, 
    pin_memory, 
    image_size, 
    train_transform,
    val_data_transform):
    # Define data transformations
    train_data_transform = transforms.Compose([
        transforms.Resize(image_size),
        train_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_data_transform = transforms.Compose([
        transforms.Resize(image_size),
        val_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom datasets
    train_dataset = CustomDataset(data_dir=os.path.join(data_dir, 'train'), transform=train_data_transform)
    val_dataset = CustomDataset(data_dir=os.path.join(data_dir, 'val'), transform=val_data_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


#and then for later use:
from data_loader import get_data_loaders

data_dir = 'path/to/your/data'
batch_size = 32
num_workers = 4
pin_memory = True
image_size = (224, 224)  # Adjust the image size as needed
train_transform = ...  # Define your training data augmentations
val_transform = ...  # Define your validation data transformations

train_loader, val_loader = get_data_loaders(data_dir, batch_size, num_workers, pin_memory, image_size, train_transform, val_transform)

"""


















"""
#Old codes suggested by GPT

#simple
def get_data_loader(data_dir, batch_size, num_workers, pin_memory, image_size):
    
    # Define data transformations
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create an ImageFolder dataset
    dataset = ImageFolder(root=data_dir, transform=data_transform)

    # Create a data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return data_loader



"""



