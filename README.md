# Chest X-Ray Abnormalities Detection with Faster R-CNN

This project implements a Faster R-CNN model with a ResNet-50 backbone for detecting abnormalities in chest X-ray images. Leveraging the power of PyTorch and PyTorch Lightning, the model is trained on a dataset of chest X-ray scans, annotating each with bounding boxes to localize and classify thoracic abnormalities.

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
│   ├── dicom_metadata_extraction.py
│   ├── dicom_pixels_features_extraction.py
│   └── dicom_pixels_processing.py
│
├── utils/                 # Utility scripts
│   ├── get_pixel_array_from_hdf5.py
│   ├── metrics.py
│   ├── reformat_prediction_file.py
│   ├── show_image_with_predictions.py
│   └── split_dataset.py
│
├── notebooks/             # Jupyter notebooks 
│   ├── dataset_analysis.ipynb
│   ├── features_extraction.ipynb
│   ├── model_evaluation.ipynb
│   ├── run_binary_classifier_inference.ipynb
│   └── train_binary_classifier.ipynb
│
├── run_data_preprocessing.py # Run Data Proprocessing (data and features extraction from DICOM files)
├── run_faster-R-CNN_inferences.py # Run inference with the faster-R-CNN model
├── train_faster-R-CNN_model.py # Train the faster-R-CNN model
├── requirements.txt       # Project dependencies
└── README.md              # Project readme with instructions and information

```

## Dataset

The dataset comprises 18,000 postero-anterior chest X-ray scans, each annotated by experienced radiologists. Annotations include 14 types of thoracic abnormalities and a class for "No finding".

## Model Overview

- **Backbone**: ResNet-50 for feature extraction.
- **Region Proposal Network (RPN)**: Proposes candidate object bounding boxes.
- **RoI Pooling**: Extracts a fixed-size feature vector from each Region of Interest (RoI).
- **Detection Network**: Classifies RoIs and refines their bounding box coordinates.

## Getting Started

```bash
python -m venv chest-x-ray-abnormalities-detection
```

Activate the virtual environment using the following command:

```bash
source chest-x-ray-abnormalities-detection/bin/activate
```

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

