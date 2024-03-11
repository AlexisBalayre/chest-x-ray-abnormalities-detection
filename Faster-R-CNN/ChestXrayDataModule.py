import lightning as L
from torch.utils.data import DataLoader

from ChestXrayDataset import ChestXrayDataset


def collate_fn(batch):
    return tuple(zip(*batch))

class ChestXrayDataModule(L.LightningDataModule):
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

    def prepare_data(self):
        pass

    def setup(self, stage=None):
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

