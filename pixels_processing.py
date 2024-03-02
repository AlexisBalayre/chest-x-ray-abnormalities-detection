import os
from datetime import datetime
import numpy as np
import pydicom
import h5py
from pydicom.pixel_data_handlers.util import apply_voi_lut
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_dicom_file(filepath):
    try:
        dicom = pydicom.dcmread(filepath, force=True)
        image = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = np.amax(image) - image
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)
        return os.path.basename(filepath), image
    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None, None


def save_batch_to_hdf5(hdf5_path, batch_data):
    with h5py.File(hdf5_path, "a") as hdf5_file:
        for filename, image in batch_data:
            if filename is not None and image is not None:
                unique_filename = get_unique_filename(hdf5_file, filename)
                hdf5_file.create_dataset(
                    unique_filename, data=image, compression="gzip"
                )


def get_unique_filename(hdf5_file, original_filename):
    if original_filename not in hdf5_file:
        return original_filename
    else:
        base, ext = os.path.splitext(original_filename)
        for i in range(1, 1000):
            new_filename = f"{base}_{i}{ext}"
            if new_filename not in hdf5_file:
                return new_filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{base}_{timestamp}{ext}"


def process_files_in_batch(files, hdf5_path):
    batch_data = [process_dicom_file(file) for file in files]
    save_batch_to_hdf5(hdf5_path, batch_data)


def parallel_process_dicom_folder(input_folder, hdf5_path, batch_size=10):
    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".dicom")
    ]
    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]
    num_processes = min(multiprocessing.cpu_count(), len(batches))

    with tqdm(total=len(files), desc="Overall Progress", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(process_files_in_batch, batch, hdf5_path)
                for batch in batches
            ]
            for future in as_completed(futures):
                pbar.update(batch_size)


if __name__ == "__main__":
    input_folder = "/Volumes/ALEXIS/ai_project_cranfield/train"
    hdf5_path = "/Volumes/ALEXIS/ai_project_cranfield/dicom_images.hdf5"
    parallel_process_dicom_folder(input_folder, hdf5_path)
