import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import cv2


class ChestXrayDataset(Dataset):
    """
    A dataset class for chest X-ray images, designed for use with PyTorch data loaders.

    This class handles loading and preprocessing of chest X-ray images stored in an HDF5 file,
    along with their associated annotations provided in a CSV file. It supports customizable
    image resizing and transformations for data augmentation.

    Parameters:
    - labels_df_path (str): Path to the CSV file containing image annotations.
    - hdf5_path (str): Path to the HDF5 file containing image data.
    - target_size (tuple): Desired output size of the images as (width, height).
    - stage (str): The stage of model training ('fit', 'validate', 'test', 'predict'),
                   which determines the set of transformations to apply.

    Attributes:
    - stage (str): Current stage of model training.
    - labels_df (DataFrame): DataFrame containing image annotations.
    - image_annotations (dict): Aggregated annotations for each image.
    - hdf5_path (str): Path to the HDF5 file.
    - target_size (tuple): Target size for image resizing.
    - transform (A.Compose): Composed Albumentations transformations to apply.
    """

    def __init__(self, labels_df_path, hdf5_path, target_size=(224, 224), stage="fit"):
        super(ChestXrayDataset, self).__init__()

        self.stage = stage

        self.labels_df = pd.read_csv(labels_df_path)

        self.image_annotations = self.aggregate_annotations()
        self.hdf5_path = hdf5_path
        self.target_size = target_size

        self.transform = (
            A.Compose(
                [
                    A.Resize(
                        width=self.target_size[0], height=self.target_size[1], p=1.0
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(p=1.0),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["labels"],
                ),
            )
            if stage == "fit"
            else A.Compose(
                [
                    A.Resize(
                        width=self.target_size[0], height=self.target_size[1], p=1.0
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(p=1.0),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["labels"],
                ),
            )
        )

    def aggregate_annotations(self):
        """
        Aggregates bounding box and label annotations for each image.

        Parses the annotations DataFrame to compile a dictionary that maps each unique
        image ID to its bounding boxes and labels.

        Returns:
        - dict: A dictionary with image IDs as keys, and their "boxes" and "labels"
                aggregated from the annotations DataFrame.
        """
        agg_annotations = {}
        for _, row in self.labels_df.iterrows():
            image_id = row["image_id"]
            if image_id not in agg_annotations:
                agg_annotations[image_id] = {"boxes": [], "labels": []}
            if pd.notna(row["x_min"]):  # If bounding box exists
                agg_annotations[image_id]["boxes"].append(
                    [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
                )
                agg_annotations[image_id]["labels"].append(row["class_id"] + 1)
        return agg_annotations

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Parameters:
        - idx (int): Index of the item to retrieve.

        Returns:
        - tuple or tuple of (str, torch.Tensor, dict): Depending on the stage,
          if 'fit', returns (image, target) where 'image' is the transformed image tensor
          and 'target' is a dictionary with boxes, labels, area, and iscrowd.
          Otherwise, returns (image_id, image, target) including the image ID.
        """

        image_id = list(self.image_annotations.keys())[idx]
        annotations = self.image_annotations[image_id]

        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            image_data = hdf5_file[image_id + ".dicom"][()]
        image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=-1)

        transformed = self.transform(
            image=image_data, bboxes=annotations["boxes"], labels=annotations["labels"]
        )
        image = transformed["image"]
        boxes = transformed["bboxes"]
        labels = transformed["labels"]

        target = {}
        if boxes:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0]
            )
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        if self.stage == "fit":
            return image, target
        return image_id, image, target

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        - int: The total number of images in the dataset.
        """
        return len(self.image_annotations)
