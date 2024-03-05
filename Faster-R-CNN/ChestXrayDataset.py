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

    def __init__(
        self, labels_df_path, hdf5_path, target_size=(224, 224), phase="train"
    ):
        super().__init__()

        self.labels_df = pd.read_csv(labels_df_path)
        self.hdf5_path = hdf5_path
        self.target_size = target_size
        self.phase = phase
        self.transform = A.Compose(
            [A.Resize(target_size[0], target_size[1]), A.Normalize(), ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
        self.hdf5_dataset_cache = {}

    def __getitem__(self, idx):
        image_id = self.labels_df.iloc[idx]["image_id"]
        records = self.labels_df[self.labels_df["image_id"] == image_id]

        if image_id + ".dicom" not in self.hdf5_dataset_cache:
            try:
                with h5py.File(self.hdf5_path, "r") as f:
                    self.hdf5_dataset_cache[image_id + ".dicom"] = f[
                        image_id + ".dicom"
                    ][()]
            except Exception as e:
                logging.error(f"Error loading HDF5 dataset for image {image_id}: {e}")
                return None

        image_array = self.hdf5_dataset_cache[image_id + ".dicom"]
        image_np = np.array(Image.fromarray(image_array).convert("RGB"))

        boxes = records[["x_min", "y_min", "x_max", "y_max"]].values
        labels = records["class_id"].values + 1  # Adjusting class labels if necessary

        if self.phase == "test":
            transformed = self.transform(image=image_np)
            return transformed["image"], image_id

        transformed = self.transform(image=image_np, bboxes=boxes, labels=labels)
        target = {
            "boxes": torch.as_tensor(transformed["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(transformed["labels"], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": (
                torch.tensor(transformed["bboxes"], dtype=torch.float32)[:, 3]
                - torch.tensor(transformed["bboxes"], dtype=torch.float32)[:, 1]
            )
            * (
                torch.tensor(transformed["bboxes"], dtype=torch.float32)[:, 2]
                - torch.tensor(transformed["bboxes"], dtype=torch.float32)[:, 0]
            ),
            "iscrowd": torch.zeros((len(transformed["labels"]),), dtype=torch.int64),
        }
        return transformed["image"], target

    def __len__(self):
        return len(self.labels_df["image_id"].unique())

    def __del__(self):
        # Close and remove cache of HDF5 datasets
        for _, hdf5_dataset in self.hdf5_dataset_cache.items():
            hdf5_dataset.file.close()
        logging.info("Closed all cached HDF5 datasets.")
