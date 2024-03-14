import os
import pydicom
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import logging
import warnings
import csv

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_dicom_file(filepath):
    """
    Reads and extracts metadata from a DICOM file.

    Parameters:
    - filepath (str): Path to the DICOM file.

    Returns:
    - dict: A dictionary containing the extracted metadata, or None if an error occurs.
    """
    try:
        dicom = pydicom.dcmread(filepath, force=True)
        data = {
            "File Name": os.path.basename(filepath),
            "Transfer Syntax UID": (
                dicom.file_meta.TransferSyntaxUID
                if "TransferSyntaxUID" in dicom.file_meta
                else "NA"
            ),
            "SOP Class UID": (
                dicom.file_meta.MediaStorageSOPClassUID
                if "MediaStorageSOPClassUID" in dicom.file_meta
                else "NA"
            ),
            "SOP Instance UID": (
                dicom.file_meta.MediaStorageSOPInstanceUID
                if "MediaStorageSOPInstanceUID" in dicom.file_meta
                else "NA"
            ),
            "Version": (
                dicom.file_meta.ImplementationVersionName
                if "ImplementationVersionName" in dicom.file_meta
                else "NA"
            ),
            "Photometric Interpretation": (
                dicom.PhotometricInterpretation
                if "PhotometricInterpretation" in dicom
                else "NA"
            ),
            "Pixel Representation": (
                dicom.PixelRepresentation if "PixelRepresentation" in dicom else "NA"
            ),
            "High Bit": dicom.HighBit if "HighBit" in dicom else "NA",
            "Samples per Pixel": (
                dicom.SamplesPerPixel if "SamplesPerPixel" in dicom else "NA"
            ),
            "Bits Allocated": dicom.BitsAllocated if "BitsAllocated" in dicom else "NA",
            "Rescale Intercept": (
                dicom.RescaleIntercept if "RescaleIntercept" in dicom else 0
            ),
            "Rescale Slope": dicom.RescaleSlope if "RescaleSlope" in dicom else 1,
            "Lossy Image Compression": (
                dicom.LossyImageCompression
                if "LossyImageCompression" in dicom
                else "NA"
            ),
            "Lossy Image Compression Ratio": (
                dicom.LossyImageCompressionRatio
                if "LossyImageCompressionRatio" in dicom
                else "NA"
            ),
            "Patient's Sex": dicom.PatientSex if "PatientSex" in dicom else "NA",
            "Patient's Age": dicom.PatientAge if "PatientAge" in dicom else "NA",
            "Rows": dicom.Rows if "Rows" in dicom else "NA",
            "Columns": dicom.Columns if "Columns" in dicom else "NA",
            "Pixel Spacing": (
                str(dicom.PixelSpacing) if "PixelSpacing" in dicom else "NA"
            ),
            "Bits Stored": dicom.BitsStored if "BitsStored" in dicom else "NA",
            "Window Center": (
                str(dicom.WindowCenter) if "WindowCenter" in dicom else "NA"
            ),
            "Window Width": str(dicom.WindowWidth) if "WindowWidth" in dicom else "NA",
        }
        return data
    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None


def check_csv_header(csv_path, fieldnames):
    """
    Ensures the CSV file exists and has the correct header. Creates the file with the header if it doesn't exist.

    Parameters:
    - csv_path (str): Path to the CSV file where metadata will be saved.
    - fieldnames (list): A list of header names for the CSV file.
    """
    if not os.path.isfile(csv_path) or os.stat(csv_path).st_size == 0:
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()


def save_batch_to_csv(csv_path, batch_data):
    """
    Appends a batch of DICOM metadata records to a CSV file.

    Parameters:
    - csv_path (str): Path to the CSV file where metadata will be saved.
    - batch_data (list of dicts): A list of dictionaries containing DICOM metadata.
    """
    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=batch_data[0].keys())
        writer.writerows(batch_data)


def process_files_in_batch(files, csv_path):
    """
    Processes a batch of DICOM files to extract metadata and save to a CSV file.

    Parameters:
    - files (list): A list of paths to DICOM files to be processed.
    - csv_path (str): Path to the CSV file where metadata will be saved.
    """
    batch_data = [
        process_dicom_file(file)
        for file in files
        if process_dicom_file(file) is not None
    ]
    if batch_data:
        save_batch_to_csv(csv_path, batch_data)


def extract_dicom_metadata(input_folder, csv_path, batch_size=10):
    """
    Extracts metadata from all DICOM files in a folder and saves it to a CSV file in batches.

    Parameters:
    - input_folder (str): Path to the folder containing DICOM files.
    - csv_path (str): Path to the CSV file where metadata will be saved.
    - batch_size (int, optional): The number of files to process in each batch.
    """
    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".dicom")
    ]
    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]
    num_processes = min(multiprocessing.cpu_count(), len(batches))

    # Explicitly defined fieldnames based on DICOM metadata structure
    fieldnames = [
        "File Name",
        "Transfer Syntax UID",
        "SOP Class UID",
        "SOP Instance UID",
        "Version",
        "Photometric Interpretation",
        "Pixel Representation",
        "High Bit",
        "Samples per Pixel",
        "Bits Allocated",
        "Rescale Intercept",
        "Rescale Slope",
        "Lossy Image Compression",
        "Lossy Image Compression Ratio",
        "Patient's Sex",
        "Patient's Age",
        "Rows",
        "Columns",
        "Pixel Spacing",
        "Bits Stored",
        "Window Center",
        "Window Width",
    ]

    # Ensure the CSV has headers before processing
    check_csv_header(csv_path, fieldnames)

    with tqdm(total=len(files), desc="Overall Progress", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(process_files_in_batch, batch, csv_path)
                for batch in batches
            ]
            for future in as_completed(futures):
                pbar.update(batch_size)
