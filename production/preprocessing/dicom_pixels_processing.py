import os
import numpy as np
import pydicom
import h5py
from pydicom.pixel_data_handlers.util import apply_voi_lut
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import threading
import multiprocessing
from tqdm import tqdm
import logging
import warnings
from datetime import datetime

# Setup logging and warnings
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def process_dicom_file(filepath):
    try:
        dicom = pydicom.dcmread(filepath, force=True)
        image = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = np.amax(image) - image
        image = (
            (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        ).astype(np.uint8)
        return os.path.basename(filepath), image
    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None, None


def get_unique_filename(hdf5_file, original_filename):
    if original_filename not in hdf5_file:
        return original_filename
    else:
        base, ext = os.path.splitext(original_filename)
        for i in range(1, 1000):
            new_filename = f"{base}_{i}{ext}"
            if new_filename not in hdf5_file:
                return new_filename
        return f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"


def process_and_queue_dicom_file(filepath, write_queue):
    processed_data = process_dicom_file(filepath)
    write_queue.put(processed_data)


def writer(write_queue, hdf5_path):
    with h5py.File(hdf5_path, "a") as hdf5_file:
        while True:
            item = write_queue.get()
            if item is None:  # Sentinel to signal done
                break
            filename, image = item
            if filename and image is not None:
                unique_filename = get_unique_filename(hdf5_file, filename)
                hdf5_file.create_dataset(
                    unique_filename, data=image, compression="gzip", chunks=True
                )
            else:
                logging.warning("Skipping a file due to error processing")
            write_queue.task_done()  # Indicate completion of task


def extract_dicom_images(input_folder, hdf5_path):
    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
    ]
    num_processes = multiprocessing.cpu_count()

    with Manager() as manager:
        # Initialize the queue with a maxsize to prevent excessive memory usage
        write_queue = manager.Queue(
            maxsize=60
        )  # Example size, adjust based on your system's capacity

        writer_thread = threading.Thread(target=writer, args=(write_queue, hdf5_path))
        writer_thread.start()

        with tqdm(total=len(files), desc="Processing DICOM Files", unit="file") as pbar:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [
                    executor.submit(process_and_queue_dicom_file, file, write_queue)
                    for file in files
                ]
                for future in as_completed(futures):
                    pbar.update(1)

        write_queue.put(None)  # Signal the writer thread to finish
        writer_thread.join()