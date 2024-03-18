import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from ChestXrayDataModule import ChestXrayDataModule
from ChestXrayLightningModel import ChestXrayLightningModel


if __name__ == "__main__":
    print("Torch CUDA available?", torch.cuda.is_available())

    # Seed for reproducibility across multiple runs
    torch.manual_seed(123)

    # Paths to the dataset and the HDF5 file containing the images
    train_df_path = (
        "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/train.csv"
    )
    val_df_path = (
        "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/val.csv"
    )
    test_df_path = (
        "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/test.csv"
    )
    hdf5_path = "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/dicom_images_final.hdf5"

    # Initialize the data module with dataset paths and processing parameters
    dataModule = ChestXrayDataModule(
        train_dataset_path=train_df_path,
        val_dataset_path=val_df_path,
        test_dataset_path=test_df_path,
        hdf5_path=hdf5_path,
        batch_size=8,  # Batch size for the DataLoader
        num_workers=8,  # Number of worker threads for DataLoader
        target_size=(800, 1000),  # Target resize dimensions for each image
    )

    # Prepare the data module for fitting (training + validation)
    dataModule.setup("fit")

    # Set the total number of training steps based on dataset size, batch size, and epochs
    num_epochs = 16
    num_steps = num_epochs * len(dataModule.train_dataset) // dataModule.batch_size

    # Initialize the model with specific hyperparameters
    model = ChestXrayLightningModel(
        learning_rate=3e-3,
        num_classes=15,
        cosine_t_max=num_steps,
    )

    # Initialize the PyTorch Lightning Trainer with desired configuration and callbacks
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",  # Utilize GPU for training, if available
        devices="auto",  # Automatically use available GPUs
        deterministic=True,  # Ensure reproducibility
        callbacks=[
            EarlyStopping(
                monitor="val_map", mode="max", patience=3
            )  # Early stopping based on validation mAP
        ],
    )

    # Start model training using the configured trainer and data module
    trainer.fit(model, datamodule=dataModule)
