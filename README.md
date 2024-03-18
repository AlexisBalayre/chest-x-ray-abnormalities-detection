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

## Dataset Preparation

1. Place a full dataset CSV file in the `data` folder
2. Modify the paths in the `utils/split_dataset.py` script and run it to prepare the training, validation and test datasets.
3. Extract and process data from the DICOM files by using `run_data_preprocessing.py` script.
4. Have a look on the distribution of the dataset by using `dataset_analysis.ipynb` notebook.

## Training

1. Train the Faster-R-CNN model using transfer learning with `train_faster-R-CNN_model.py` script.
2. Train the Binary classifier model with the `train_binary_classifier.ipynb` notebook.

## Inference

1. Run the Faster-R-CNN model inference with the test dataset using `run_faster-R-CNN_inferences.py`. You'll obtain a submission file.
2. Convert the submission file to more friendly format using `utils/reformat_prediction_file.py`.
3. Run the binary model inference with the test dataset using `run_binary_classifier_inference.ipynb` notebook.

## Model Evaluation

Evaluate the trained model using the `model_evaluation.ipynb` notebook.
