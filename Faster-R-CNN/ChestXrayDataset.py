import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import logging

class ChestXrayDataset(Dataset):
    """
    A dataset class for chest X-ray images stored in an HDF5 file.

    Parameters:
        labels_df_path (str): Path to the CSV file containing labels and bounding boxes.
        hdf5_path (str): Path to the HDF5 file containing image data.
        target_size (tuple): Target size to which images are resized.
        phase (str): Indicates the dataset phase ('train' or 'test').
    """
    def __init__(self, labels_df_path, hdf5_path, target_size=(224, 224), phase='train'):
        super().__init__()
    
        self.labels_df = pd.read_csv(labels_df_path)
        self.hdf5_path = hdf5_path
        self.target_size = target_size
        self.phase = phase
        self.transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __getitem__(self, idx):
        image_id = self.labels_df.iloc[idx]['image_id']
        records = self.labels_df[self.labels_df['image_id'] == image_id]
        
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            image_id = self.labels_df.iloc[idx]['image_id']
            image_data = hdf5_file[image_id + ".dicom"][()]
        
        image = Image.fromarray(image_data).convert("RGB")
        image_np = np.array(image)
        
        # Filter out records with 'No finding' or ensure they have default bounding box values
        valid_records = records.dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])
        boxes = valid_records[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = valid_records["class_id"].values + 1  # Adjust labels as needed, '+1' to account for background class

        transformed = self.transform(image=image_np, bboxes=boxes, labels=labels)
        image = transformed['image']
        
        target = {}
        if len(boxes) > 0:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Handle cases with no boxes
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)  # Adjust accordingly
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)


        return image, target

    
    def __len__(self):
        return len(self.labels_df['image_id'].unique())
