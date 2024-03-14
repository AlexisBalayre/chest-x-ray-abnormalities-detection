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
    """
    Reads a DICOM file, applies VOI LUT (if available), normalizes the image, and converts to uint8.

    Parameters:
    - filepath (str): Path to the DICOM file.

    Returns:
    - tuple: Tuple containing the basename of the file and the processed image as a numpy array.
             Returns None, None if an error occurs.
    """
    try:
        dicom = pydicom.dcmread(filepath, force=True)  # Read the DICOM file
        image = apply_voi_lut(dicom.pixel_array, dicom)  # Apply VOI LUT if available
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = np.amax(image) - image  # Invert the image if MONOCHROME1
        image = (
            (image - np.min(image))
            / (np.max(image) - np.min(image))
            * 255  # Normalize the image
        ).astype(
            np.uint8
        )  # Convert to uint8
        return (
            os.path.basename(filepath),
            image,
        )  # Return the basename and the processed image
    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None, None


def get_unique_filename(hdf5_file, original_filename):
    """
    Generates a unique filename for the HDF5 file to avoid name clashes.

    Parameters:
    - hdf5_file (h5py.File): The HDF5 file object.
    - original_filename (str): The original filename to be saved.

    Returns:
    - str: A unique filename within the HDF5 file.
    """
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
    """
    Processes a DICOM file and queues it for writing to the HDF5 file.

    Parameters:
    - filepath (str): Path to the DICOM file.
    - write_queue (multiprocessing.queues.Queue): Queue to hold processed data for writing.
    """
    processed_data = process_dicom_file(filepath)
    write_queue.put(processed_data)


def writer(write_queue, hdf5_path):
    """
    Consumes items from the queue and writes them to an HDF5 file.

    Parameters:
    - write_queue (multiprocessing.queues.Queue): Queue from which to consume data.
    - hdf5_path (str): Path to the HDF5 file.
    """
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
    """
    Main function to extract images from DICOM files and store them in an HDF5 file.

    Parameters:
    - input_folder (str): Folder containing DICOM files.
    - hdf5_path (str): Path to the output HDF5 file.
    """
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
