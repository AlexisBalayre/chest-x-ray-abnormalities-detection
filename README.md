# Chest X-Ray Abnormalities Detection with Faster R-CNN

This project leverages the Faster R-CNN model with a ResNet-50 backbone, implemented using PyTorch Lightning, for the detection and localization of thoracic abnormalities in chest X-ray images. The model is trained on a curated dataset of chest X-ray scans, each annotated with bounding boxes to identify various abnormalities.

## Project Overview

Chest X-rays are among the most common radiographic examinations performed for diagnosing thoracic diseases. However, the interpretation of these images is highly reliant on the experience of radiologists, leading to the need for automated systems that can assist in accurately identifying and classifying potential abnormalities. This project aims to address this need by employing a Faster R-CNN model, known for its efficiency in object detection tasks.

## Project Structure

```
chest-xray-abnormalities-detection/
│
├── data/                  # Data files including CSVs and image data
│   ├── train/             # Training data and annotations
│   │   ├── train.csv
│   │   └── val.csv
│   ├── test/              # Test data and annotations
│   │   ├── classifier_post_pred.csv # Binary Classification Inference Results
│   │   ├── dicom_metadata.csv # Metadata of DICOM files
│   │   ├── dicom_pixels_features.csv # Pixel Arrays Features extacted from DICOM files
│   │   ├── test_predictions_reformatted.csv # Faster-R-CNN inference results (dataframe form)
│   │   ├── test_predictions.csv Faster-R-CNN inference results (submission form)
│   │   └── test.csv # Test data and annotations
│   └── train_full_dataset.csv # Full dataset compilation for reference
│
├── models-weight/                # Model files and weights
│   ├── binary_classifier.pkl # Binary classification model weights
│   └── Faster-R-CNN-Models/ # Faster-R-CNN models weights
│
├── preprocessing/         # Scripts for data preprocessing
│   ├── dicom_metadata_extraction.py # Extracts DICOM files metadata
│   ├── dicom_pixels_features_extraction.py # Extracts features from DICOM files images
│   └── dicom_pixels_processing.py # Extracts and processes images from DICOM files
│
├── utils/                 # Utility scripts
│   ├── get_pixel_array_from_hdf5.py # Retrieves the pixel array of a DICOM file
│   ├── metrics.py # Evaluation Metrics (mAP, IoU)
│   ├── reformat_prediction_file.py # Convert submission file format to classic dataframe
│   ├── show_image_with_predictions.py # Visualise a chest radiography with predicted or true labels
│   └── split_dataset.py # split the dataset in trainning, validation and test sets
│
├── dataset_analysis.ipynb # Primary Analysis of the Dataset
├── features_extraction.ipynb # Analysis of extracted features
├── model_evaluation.ipynb # Evaluation of the trained model
├── run_binary_classifier_inference.ipynb # Run inference with the binary classifier model
├── train_binary_classifier.ipynb # Train the binary classifier model
├── run_data_preprocessing.py # Run Data Proprocessing (data and features extraction from DICOM files)
├── run_faster-R-CNN_inferences.py # Run inference with the faster-R-CNN model
├── train_faster-R-CNN_model.py # Train the faster-R-CNN model
├── requirements.txt       # Project dependencies
└── README.md              # Project readme with instructions and information

```

### Prerequisites

- Python 3.8 or later
- pip for installing Python packages

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone https://github.com/<your-repository>/chest-xray-abnormalities-detection.git
cd chest-xray-abnormalities-detection
```

2. Create and activate a virtual environment:

```bash
python -m venv chest-x-ray-abnormalities-detection
```

3. Activate the virtual environment using the following command:

```bash
source chest-x-ray-abnormalities-detection/bin/activate
```

4. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### Dataset Preparation

1. Place the full dataset CSV file and DICOM images in the `data/` directory.
2. Use the scripts in `preprocessing/` to extract and process DICOM images and metadata.
3. Split the dataset into training, validation, and test sets with `utils/split_dataset.py`.

## Training the Models

- Train the Faster R-CNN model with `python train_faster-R-CNN_model.py`.
- Train the binary classifier with `jupyter notebook train_binary_classifier.ipynb`.

## Running Inference

- Perform inference with the Faster R-CNN model using `python run_faster-R-CNN_inferences.py`.
- Convert the submission file format with `python utils/reformat_prediction_file.py`.
- Run binary classifier inference using `jupyter notebook run_binary_classifier_inference.ipynb`.

## Model Evaluation

Evaluate the performance of the trained models using the `model_evaluation.ipynb` notebook, focusing on metrics such as mean Average Precision (mAP) and Intersection over Union (IoU).
