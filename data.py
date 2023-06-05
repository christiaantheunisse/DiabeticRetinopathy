import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import copy

"""
    This file contains the dataset class and the data augmentation classes.
"""

class DiabeticRetinopathyDataset(Dataset):
    """The Diabetic Retinopathy dataset from Kaggle."""

    def __init__(self, csv_file: str, root_dir: str, image_dir: str, size: int = None, transform=None):
        """
        Arguments:
            csv_file (string): The csv file with the labels
            root_dir (string): The directory where the data is stored. The csv file should be in this directory.
            image_dir (string): The path to the directory with the images from the root_dir.
            size (int): Only consider the first number of samples from the csv file. 
            transform (Callable, optional): Optional transform to be applied on samples
        """
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.transform = transform
        self.ToTensor = transforms.ToTensor()
        
        self.items = self.df.iloc[:, 0]
        self.labels = self.df.iloc[:, 1]
        if size is not None:
            self.items = self.items[:size]
            self.labels = self.labels[:size]
       
    def __getitem__(self, idx) -> torch.Tensor:
        """
        Can obtain images as tensors from dataset by giving an index or 
        list/np.array/torch.Tensor of indices as input.
        
        Return (torch.Tensor, torch.Tensor): a batch of images (B x C x H x W), labels (B)
        
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        
        assert max(idx) < len(self.items), "Outside the size of the dataset"
        
        selected_items = self.items[idx]
        sel_imgs_paths = [os.path.join(self.image_dir, im) + '.jpeg' for im in self.items[idx].tolist()]
        sel_labels = self.labels[idx].tolist()
        images = [Image.open(p) for p in sel_imgs_paths]
        
        if self.transform:
            images = self.transform(images)
            
        images_tensors = [self.ToTensor(im) for im in images]
        
        return torch.stack(images_tensors, dim=0).squeeze(), torch.Tensor(sel_labels).type(torch.int64).squeeze()
    
    def __len__(self):
        return len(self.items)
    

"""
    Data augmentation methods
"""

class Resize(object):
    """
    A class to resize the samples used for data augmentation.
    
    input is list of PIL objects, output should be list of PIL objects
    
    Arguments:
        output_size (type): ...
    """
    def __init__(self, output_size: int):
        self.output_size = output_size
        
    def __call__(self, samples):
        samples = [im.resize((self.output_size,)*2) for im in samples]
        return samples