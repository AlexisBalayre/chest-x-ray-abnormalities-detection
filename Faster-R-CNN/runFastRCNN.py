import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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
        num_workers=8,
        target_size=(1024, 1024)
    )

    dataModule.setup("fit")
    
    num_epochs = 20     
    num_steps = num_epochs * len(dataModule.train_dataset) // dataModule.batch_size

    model = ChestXrayLightningModel(
        learning_rate=3e-3,
        num_classes=15,
        cosine_t_max=num_steps
    )

    trainer = L.Trainer(
        max_epochs=num_steps,
        accelerator="gpu",
        devices="auto",
        deterministic=True,
        callbacks=[EarlyStopping(monitor="train_loss", mode="min")]
    )

    model.train()
 
    trainer.fit(model, datamodule=dataModule)