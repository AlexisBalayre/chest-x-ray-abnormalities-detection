import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from ChestXrayDataModule import ChestXrayDataModule
from ChestXrayLightningModelV2 import ChestXrayLightningModel


if __name__ == "__main__":
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)  # Set seed for reproducibility

    train_df_path = "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/train.csv"  # Path to the training dataset
    val_df_path = "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/val.csv"  # Path to the validation dataset
    test_df_path = "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/test.csv"  # Path to the test dataset
    hdf5_path = "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/dicom_images_final.hdf5"  # Path to the HDF5 file containing the datasets

    dataModule = ChestXrayDataModule(
        train_dataset_path=train_df_path,
        val_dataset_path=val_df_path,
        test_dataset_path=test_df_path,
        hdf5_path=hdf5_path,
        batch_size=8,  # Set the batch size for data loading
        num_workers=8,  # Set the number of worker threads for data loading operations
        target_size=(800, 1000),  # Set the target size for image resizing
    )  # Initialize the data module

    dataModule.setup("fit")  # Setup the training dataset

    # Model parameters
    num_epochs = 16  # Set the number of epochs for training
    num_steps = (
        num_epochs * len(dataModule.train_dataset) // dataModule.batch_size
    )  # Calculate the number of steps for the cosine annealing scheduler

    # Initialize the model
    model = ChestXrayLightningModel(
        learning_rate=3e-3,  # Learning Rate
        num_classes=15,  # Number of classes
        cosine_t_max=num_steps,  # Number of steps
    )

    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",  # Use the GPUs for training
        devices="auto", 
        deterministic=True, # For reproducibility
        callbacks=[
            EarlyStopping(monitor="val_map", mode="max")
        ],  # Add an early stopping callback
    )

    model.train()  # Set the model to training mode

    trainer.fit(model, datamodule=dataModule)  # Train the model using the data module
