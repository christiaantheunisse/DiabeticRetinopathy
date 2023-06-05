import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np

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
        images = [cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in sel_imgs_paths]
        
        if self.transform:
            images = self.transform(images)
        
        # Always convert the images to a torch.Tensor
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
    
    input is list of cv2 objects, output should be list of cv2 objects
    
    Arguments:
        output_size (type): ...
    """
    def __init__(self, output_size: int):
        self.output_size = output_size
        
    def __call__(self, samples):
        samples = [cv2.resize(im, (self.output_size,)*2) for im in samples]
        return samples
    
class CropBlack(object):
    """
    A class to crop the image in such a way that all the unnecessary black background is removed.
    
    input is list of cv2 objects, output should be list of cv2 objects
    
    Arguments:
        output_size (type): ...
    """
    def __init__(self):
        pass
    
    @staticmethod
    def crop_black(im):
        thresholded = im[...,0] > 10  # This results in a True/False mask
        
        height, width = im.shape[:2]
        h = height // 2
        w = width // 2
        
        # Get the center lines of the image, since the eye circle will be the widest at this point,
        # assuming that the eye is centered
        horz_line = thresholded[h, :]
        vert_line = thresholded[:, w]
        
        # Since the array is binary, the index of the first element is returned by the argmax()
        # Search for the first True element in both directions
        h_low, h_high = horz_line.argmax(), len(horz_line) - np.flip(horz_line).argmax()
        v_low, v_high = vert_line.argmax(), len(vert_line) - np.flip(vert_line).argmax()
        
        # Crop the image according to the find values
        return im[v_low:v_high, h_low:h_high]
        
    def __call__(self, samples):
        samples = [self.crop_black(im) for im in samples]
        return samples