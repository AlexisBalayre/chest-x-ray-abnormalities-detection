from tqdm import tqdm
import torch
import csv

from ChestXrayDataModule import ChestXrayDataModule
from ChestXrayLightningModel import ChestXrayLightningModel


def format_prediction_string(labels, boxes, scores):
    """
    Creates a formatted prediction string for saving predictions.
    Each prediction comprises a class label, a confidence score, and four bounding box coordinates.

    Parameters:
    - labels (list of int): The list of predicted class labels for each bounding box.
    - boxes (list of list of float): Bounding box coordinates for each prediction, given as [x_min, y_min, x_max, y_max].
    - scores (list of float): The confidence scores for each predicted bounding box.

    Returns:
    - str: A single string containing all formatted predictions separated by spaces.
    """
    # Initialize an empty list for holding each individual prediction string
    pred_strings = []
    for label, score, box in zip(labels, scores, boxes):
        # Format each prediction and append it to the list
        pred_strings.append(f"{label} {score:.4f} {box[0]} {box[1]} {box[2]} {box[3]}")

    # Join all prediction strings into a single string separated by spaces
    return " ".join(pred_strings)


if __name__ == "__main__":
    # Check and print if CUDA is available for PyTorch, facilitating GPU acceleration
    print("Torch CUDA available?", torch.cuda.is_available())

    # Seed the random number generator for reproducibility
    torch.manual_seed(123)

    # Define paths to dataset CSV files and HDF5 file containing the images
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

    # Initialize the data module with paths and parameters
    dataModule = ChestXrayDataModule(
        train_dataset_path=train_df_path,
        val_dataset_path=val_df_path,
        test_dataset_path=test_df_path,
        hdf5_path=hdf5_path,
        batch_size=8,
        num_workers=8,
        target_size=(800, 1000),
    )

    # Load a trained model from a checkpoint
    model = ChestXrayLightningModel.load_from_checkpoint(
        "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/lightning_logs/version_46/checkpoints/epoch=14-step=13890.ckpt",
        num_classes=15,
    )

    # Define the detection threshold for filtering out low-confidence predictions
    detection_threshold = 0.1

    # Prepare the model for evaluation
    model.model.eval()

    # Set up the data module for the test phase
    dataModule.setup("test")

    # Open a CSV file for writing the predictions
    with open("test_predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row in the CSV file
        writer.writerow(["image_id", "PredictionString"])

        # Iterate over the test dataset using a DataLoader
        with torch.inference_mode():
            for image_ids, images, _ in tqdm(
                dataModule.test_dataloader(), desc="Evaluating"
            ):
                # Get model predictions for the batch of images
                outputs = model(images)
                for i, output in enumerate(outputs):
                    image_id = image_ids[i]

                    # Extract boxes, labels, and scores from the model's output
                    boxes = output["boxes"].data.cpu().numpy()
                    labels = (
                        output["labels"].data.cpu().numpy() - 1
                    ) 
                    scores = output["scores"].data.cpu().numpy()

                    # Filter predictions based on the detection threshold
                    valid = scores >= detection_threshold

                    # If there are any valid predictions, format them and save to CSV
                    if valid.any():
                        result = {
                            "image_id": image_id,
                            "PredictionString": format_prediction_string(
                                labels[valid], boxes[valid], scores[valid]
                            ),
                        }

                    # Write the results for each image to the CSV file
                    writer.writerow([result["image_id"], result["PredictionString"]])
