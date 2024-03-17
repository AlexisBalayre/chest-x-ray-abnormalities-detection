import lightning as L
from torch.utils.data import DataLoader

from ChestXrayDataset import ChestXrayDataset


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    This function prepares a batch by collating the list of samples into a batch,
    where each sample is a tuple of its attributes. It is used to handle cases where
    the dataset returns a tuple of data points. The function rearranges the batch to
    align each element of the tuple across the data points in the batch.

    Parameters:
    - batch (list): A list of tuples, where each tuple corresponds to a data sample.

    Returns:
    - tuple: A tuple of lists, where each list contains all elements of the batch for
      that position in the original tuple.
    """
    return tuple(zip(*batch))


class ChestXrayDataModule(L.LightningDataModule):
    """
    Data module for chest X-ray images, utilizing PyTorch Lightning for structured data loading.

    This module is designed to handle the loading and preprocessing of chest X-ray image datasets,
    facilitating easy integration into a deep learning pipeline. It is specifically configured
    to work with HDF5 datasets and is customizable in terms of image size, batch size, and the
    number of worker threads for data loading.

    Attributes:
    - hdf5_path (str): Path to the HDF5 file containing the datasets.
    - train_dataset_path (str): Path to the training dataset.
    - val_dataset_path (str): Path to the validation dataset.
    - test_dataset_path (str): Path to the test dataset.
    - target_size (tuple): The dimensions to which the images will be resized.
    - batch_size (int): The size of each data batch.
    - num_workers (int): The number of worker threads for data loading operations.
    """

    def __init__(
        self,
        hdf5_path,
        train_dataset_path,
        val_dataset_path,
        test_dataset_path,
        target_size=(224, 224),
        batch_size=8,
        num_workers=8,
    ):
        super().__init__()

        self.hdf5_path = hdf5_path
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Prepares the datasets for the training, validation, testing, and prediction stages.

        Depending on the stage, this method initializes the corresponding dataset(s)
        using the provided dataset paths and the HDF5 file. It ensures that datasets
        are ready for use when their respective DataLoader is called.

        Parameters:
        - stage (str, optional): The stage for which to setup datasets. Can be 'fit',
          'test', 'predict', or None. If None, datasets for all stages are prepared.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = ChestXrayDataset(
                self.train_dataset_path, self.hdf5_path, self.target_size, stage=stage
            )
            self.val_dataset = ChestXrayDataset(
                self.val_dataset_path, self.hdf5_path, self.target_size, stage=stage
            )
        if stage == "test" or stage is None:
            self.test_dataset = ChestXrayDataset(
                self.test_dataset_path, self.hdf5_path, self.target_size, stage=stage
            )

    def train_dataloader(self):
        """
        Creates a DataLoader for the training dataset.

        Returns:
        - DataLoader: The DataLoader for the training dataset, configured with
          shuffle, batch size, and the custom collate function.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Creates a DataLoader for the validation dataset.

        Returns:
        - DataLoader: The DataLoader for the validation dataset, configured with
          batch size and the custom collate function.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """
        Creates a DataLoader for the test dataset.

        Returns:
        - DataLoader: The DataLoader for the test dataset, configured with
          batch size and the custom collate function.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
