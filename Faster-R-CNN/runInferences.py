from tqdm import tqdm
import torch
import csv

from ChestXrayDataModule import ChestXrayDataModule
from ChestXrayLightningModelV2 import ChestXrayLightningModel


def format_prediction_string(labels, boxes, scores):
    """
    Formats the prediction string with normalized bounding box coordinates.

    Parameters:
    - labels (list of int): Object class labels.
    - boxes (list of list of float): Bounding boxes, each as [x_min, y_min, x_max, y_max].
    - scores (list of float): Confidence scores for each bounding box.
    - width (int): Width of the original image.
    - height (int): Height of the original image.

    Returns:
    - str: A single string containing all predictions, formatted correctly.
    """
    pred_strings = []
    for label, score, box in zip(labels, scores, boxes):
        pred_strings.append(f"{label} {score:.4f} {box[0]} {box[1]} {box[2]} {box[3]}")

    return " ".join(pred_strings)


if __name__ == "__main__":
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

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

    dataModule = ChestXrayDataModule(
        train_dataset_path=train_df_path,
        val_dataset_path=val_df_path,
        test_dataset_path=test_df_path,
        hdf5_path=hdf5_path,
        batch_size=8,
        num_workers=8,
        target_size=(800, 1000),
    )

    model = ChestXrayLightningModel.load_from_checkpoint(
        "/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/lightning_logs/version_46/checkpoints/epoch=14-step=13890.ckpt",
        num_classes=15,
    )

    detection_threshold = 0.1
    results = []
    model.model.eval()
    dataModule.setup("test")

    with open("test_predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "PredictionString"])

        with torch.inference_mode():
            for image_ids, images, _ in tqdm(
                dataModule.test_dataloader(), desc="Evaluating"
            ):
                outputs = model(images)
                for i, output in enumerate(outputs):
                    image_id = image_ids[i]
                    result = {
                        "image_id": image_id,
                        "PredictionString": "14 1.0 0 0 1 1",
                    }

                    boxes = output["boxes"].data.cpu().numpy()
                    labels = output["labels"].data.cpu().numpy() - 1
                    scores = output["scores"].data.cpu().numpy()

                    valid = scores >= detection_threshold

                    if valid.any():
                        result = {
                            "image_id": image_id,
                            "PredictionString": format_prediction_string(
                                labels[valid], boxes[valid], scores[valid]
                            ),
                        }

                    writer.writerow([result["image_id"], result["PredictionString"]])
