import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from matplotlib import pyplot as plt

from ChestXrayDataModule import ChestXrayDataModule
from ChestXrayLightningModel import ChestXrayLightningModel


if __name__ == "__main__":
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

    train_df_path = "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/train.csv"
    val_df_path = "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/val.csv"
    test_df_path = "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/test.csv"
    hdf5_path = "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/dicom_images_final.hdf5"

    dataModule = ChestXrayDataModule(
        train_dataset_path=train_df_path,
        val_dataset_path=val_df_path,
        test_dataset_path=test_df_path,
        hdf5_path=hdf5_path,
        batch_size=8,
        num_workers=8
    )

    model = ChestXrayLightningModel(
        learning_rate=0.002,
        num_classes=15
    )

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="cpu",
        devices="auto",
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_map", mode="max")]
    )

    tuner = Tuner(
        trainer=trainer)
    
    lr_finder = tuner.lr_find(model, datamodule=dataModule)
    new_lr = lr_finder.suggestion()
    print(new_lr)

    figure = lr_finder.plot(suggest=True)
    plt.savefig('lr.png', bbox_inches='tight')
    

