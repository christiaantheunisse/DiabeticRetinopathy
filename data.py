import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

"""
    This file contains the dataset class and the data augmentation classes.
"""

class DiabeticRetinopathyDataset(Dataset):
    """The Diabetic Retinopathy dataset from Kaggle."""

    def __init__(self, csv_file: str, 
                 root_dir: str, 
                 image_dir: str, 
                 size: int = None, 
                 transform=None, 
                 sample_rates: Dict[int, float]={0: 0.5, 1: 2., 2: 1., 3: 3., 4: 3.},
                 verify: bool = False,
                ):
        """
        Arguments:
            csv_file (string): The csv file with the labels
            root_dir (string): The directory where the data is stored. The csv file should be in this directory.
            image_dir (string): The path to the directory with the images from the root_dir.
            size (int): Only consider the first number of samples from the csv file. 
            transform (Callable, optional): Optional transform to be applied on samples
            sample_rate (Dict[int, float]): The down-/upsample rate per class. Classes that are not included in the dict
                will not be sampled at all.
            verify (bool): Call the `_reduce_to_available` function when initializing
        """
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.transform = transform
        self.ToTensor = transforms.ToTensor()
        
        self.items = self.df.iloc[:, 0]
        self.labels = self.df.iloc[:, 1]
        
        # Use only found images
        if verify:
            self._reduce_to_available()
        
        if size is not None:
            self.items = self.items[:size]
            self.labels = self.labels[:size]
        
        if sample_rates:
            self.sample_rates = sample_rates
            self._down_up_sample()
          
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
    
    def _reduce_to_available(self):
        """
            Make the dataset only use images and labels from the csv-file that are actually found in the data folder.
        """
        files = [f.replace('.jpeg', '') for f in os.listdir(self.image_dir)]
        new_items, new_labels = [], []
        for item, label in zip(self.items, self.labels):
            if item in files:
                new_items.append(item)
                new_labels.append(label)

        assert len(new_items) == len(new_labels)
        print(f"The new no. of images in the dataset is: {len(new_items)}")
        
        self.items = pd.Series(new_items)
        self.labels = pd.Series(new_labels)
    
    def _down_up_sample(self):
        """
            Change the frequency per class according to self.sample_rates
        """
        # Newly sampled data is saved in this dict
        new_data_dict: Dict[int, Tuple[pd.core.series.Series, pd.core.series.Series]] = dict()

        for key, value in self.sample_rates.items():
            # mask and indices for a certain label class
            mask = self.labels == key
            indices = np.arange(len(self))[mask]

            # if downsampling
            if value < 1:
                # Just use the first ones, since the dataset is already shuffled
                keep_idx = int(len(indices) * value)
                new_labels = self.labels[mask][:keep_idx]
                new_items = self.items[mask][:keep_idx]
            # if oversampling
            else:
                # Concatenate multiple times with itself
                new_labels = pd.concat((self.labels[mask],)*int(value))
                new_items = pd.concat((self.items[mask],)*int(value))

            assert len(new_labels) == len(new_items)
            new_data_dict[key] = (new_labels, new_items)

        # Concatenate the data 
        labels = pd.concat([v[0] for v in new_data_dict.values()]).reset_index(drop=True)
        items = pd.concat([v[1] for v in new_data_dict.values()]).reset_index(drop=True)

        # Shuffle the data (with random seed for reproducibility)
        np.random.seed(33)
        shuffled_indices = np.arange(len(labels))
        np.random.shuffle(shuffled_indices)
        self.labels = labels[shuffled_indices]
        self.items = items[shuffled_indices]
        

"""
    Data augmentation methods
"""

class Resize():
    """
    A class to resize the samples used for data augmentation.
    
    input is list of cv2 objects, output should be list of cv2 objects
    
    Arguments:
        output_size (int): desired height and width of the image
    """
    def __init__(self, output_size: int):
        self.output_size = output_size
        
    def __call__(self, samples):
        samples = [cv2.resize(im, (self.output_size,)*2) for im in samples]
        return samples
    
class CropBlack():
    """
    A class to crop the image in such a way that all the unnecessary black background is removed.
    
    input is list of cv2 objects, output should be list of cv2 objects
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
    

class RandomCrop():
    """
        A class to randomly crop the image to a certain size

        input is list of cv2 objects, output should be list of cv2 objects
    
    Arguments:
        output_size (int): desired height and width of the image
    """
    def __init__(self, output_size: int):
        self.output_size = output_size
        
    def random_crop(self, im):
        h, w = im.shape[:2]
        
        h_crop = np.random.randint(low=0, high=h-self.output_size+1)
        w_crop = np.random.randint(low=0, high=w-self.output_size+1)
        im = im[h_crop:h_crop+self.output_size, w_crop:w_crop+self.output_size]
        
        return im      
        
    def __call__(self, samples):
        samples = [self.random_crop(im) for im in samples]
        return samples
    

class RandomFlip():
    """
        A class to flip the image horizontally and vertically with a certain probability

        input is list of cv2 objects, output should be list of cv2 objects
    
    Arguments:
        horz_prob (float): probability to flip horizontally
        vert_prob (float): probability to flip vertically
    """
    def __init__(self, horz_prob: float = 0.5, vert_prob: float = 0.5):
        self.horz_prob = horz_prob
        self.vert_prob = vert_prob
        
    def flip(self, im):
        # flipcode (int)...
        #   = 0 : flip vertically
        #   > 0 : flip horizontally
        #   < 0 : flip both axes
        
        h_flip = np.random.uniform() < self.horz_prob
        v_flip = np.random.uniform() < self.vert_prob
        
        if h_flip and v_flip:
            im = cv2.flip(im, -1)
        elif h_flip:
            im = cv2.flip(im, 1)
        elif v_flip:
            im = cv2.flip(im, 0)
        
        return im   
    
    def __call__(self, samples):
        samples = [self.flip(im) for im in samples]
        return samples
   
    
class RandomElasticDeformation():
    """
        A class to flip the image horizontally and vertically with a certain probability

        input is list of cv2 objects, output should be list of cv2 objects
    
    Arguments:
        horz_prob (float): probability to flip horizontally
        vert_prob (float): probability to flip vertically
    """
    
    
    # Code based on https://www.kaggle.com/code/bguberfain/elastic-transform-for-data-augmentation/notebook
    def elastic_transformation(self, image, alpha, sigma, alpha_affine, grid_size, random_state=None):
        
        # Generate a matrix of shape (image_shape[0]ximage_shape[1]) filled with random values on [0..1]
        if random_state is None:
            random_state = np.random.RandomState(None)
        
        # Store image parameters
        image_shape = image.shape
        image_shape_size = image_shape[:2]
        
        # Construct the random affine transformation
        center_square = np.float32(image_shape_size) // 2
        center_square_size = min(image_shape_size) // 3
        
        points_one = np.float32([center_square + center_square_size,  [center_square[0]+center_square_size, center_square[1]-center_square_size], center_square - center_square_size])
        
        points_two = points_one + random_state.uniform(-alpha_affine, alpha_affine, size=points_one.shape).astype(np.float32)
        
        # Create Affine transformation matrix and warp image
        affine_matrix = cv2.getAffineTransform(points_one, points_two)
        new_image  = cv2.warpAffine(image, affine_matrix, image_shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        

        # Compute deltas
        delta_x = gaussian_filter((random_state.rand(*image_shape) * 2 - 1), sigma) * alpha
        delta_y = gaussian_filter((random_state.rand(*image_shape) * 2 - 1), sigma) * alpha
        delta_z = np.zeros_like(delta_x) 


        x, y, z = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]), np.arange(image_shape[2]))
        indices = np.reshape(y+delta_y, (-1, 1)), np.reshape(x+delta_x, (-1, 1)), np.reshape(z, (-1, 1))

        # Draw distortion grid on image if wanted
        if grid_size != 0:
            for i in range(0, image.shape[1], grid_size):
                cv2.line(image, (i, 0), (i, image.shape[0]), color=(255,255, 255))
            for j in range(0, image.shape[0], grid_size):
                cv2.line(image, (0, j), (image.shape[1], j), color=(255,255, 255))


        return map_coordinates(image, indices, order=1, mode='reflect').reshape(image_shape)
    
        
        
    def __call__(self, samples):
        samples = [self.elastic_transformation(im, 200,im.shape[0]*0.08 , im.shape[0]*0.08, 0) for im in samples]
        return samples
    
    
    
    
    
    
    
    
    
    
    