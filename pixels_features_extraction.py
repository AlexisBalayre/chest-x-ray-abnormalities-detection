import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, canny, hog
from skimage.filters import sobel, gabor
from skimage import exposure
from scipy.stats import skew, kurtosis
import mahotas as mh
import csv
import pywt
from pylab import imshow
import logging
import warnings
import h5py

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

warnings.filterwarnings("ignore")


def extract_intensity_histogram(image, visualize=False):
    """
    Extracts intensity histogram from a given image.

    Parameters:
    - image: A 2D numpy array representing the image.
    - visualize: If True, display the histogram.

    Returns:
    - A tuple containing the histogram and bin edges.
    """
    hist, bin_edges = np.histogram(
        image.ravel(), bins=256, range=(0, 256), density=True
    )

    if visualize:
        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(0, 256), hist, width=0.8, align="center")
        plt.title("Intensity Histogram")
        plt.xlabel("Intensity Value")
        plt.ylabel("Frequency")
        plt.show()

    return hist, bin_edges


def extract_intensity_features(image):
    """
    Extracts intensity features from a given image.

    Parameters:
    - image: A 2D numpy array representing the image.

    Returns:
    - A dictionary containing the extracted features.
    """

    return {
        "intensity_mean": np.mean(image),
        "intensity_std": np.std(image),
        "intensity_median": np.median(image),
    }


def extract_lbp_features(image, P=8, R=1, method="uniform", visualize=False):
    """
    Extract Local Binary Pattern (LBP) features.
    Parameters:
    - image: 2D numpy array, preprocessed image.
    - P: Number of circularly symmetric neighbor set points.
    - R: Radius of circle.
    - method: Method for computing LBP.
    Returns:
    - LBP image.
    """
    lbp_image = local_binary_pattern(image, P, R, method)
    lbp_hist, _ = np.histogram(
        lbp_image.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum() + 1e-6  # Normalize

    if visualize:
        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(0, P + 2), lbp_hist, width=0.8, align="center")
        plt.title("LBP Histogram")
        plt.xlabel("LBP Value")
        plt.ylabel("Frequency")
        plt.show()

    return lbp_hist


def calculate_glcm_features(image, distances, angles):
    """
    Calculate GLCM properties like contrast, correlation, energy, and homogeneity.

    Parameters:
    - image: 2D numpy array, preprocessed image.
    - distances: List of pixel distances for GLCM calculation.
    - angles: List of angles (in radians) for GLCM calculation.

    Returns:
    - Dictionary of GLCM features.
    """
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )
    features = {
        "contrast": graycoprops(glcm, "contrast").mean(),
        "correlation": graycoprops(glcm, "correlation").mean(),
        "energy": graycoprops(glcm, "energy").mean(),
        "homogeneity": graycoprops(glcm, "homogeneity").mean(),
    }
    return features


def extract_haralick_features(image, sigma=5, visualize=False):
    """
    Extract Haralick texture features using mahotas.
    Parameters:
    - image: 2D numpy array, preprocessed image.
    Returns:
    - Haralick features.
    """
    # setting gaussian filter
    gaussian = mh.gaussian_filter(image, sigma)

    # setting threshold value
    gaussian = gaussian > gaussian.mean()

    # making is labelled image
    labeled, n = mh.label(gaussian)

    # computing haralick features
    h_feature = mh.features.haralick(labeled)

    # Visualize the Haralick features
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.title("Labeled Image")
        imshow(labeled)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.title("Haralick Features")
        imshow(h_feature)

    return h_feature


def calculate_gabor_features(image, frequency, theta, visualize=False):
    """
    Apply Gabor filters to extract texture features.

    Parameters:
    - image: 2D numpy array, preprocessed image.
    - frequency: Spatial frequency of the sinusoidal factor.
    - theta: Orientation of the normal to the parallel stripes of a Gabor function.

    Returns:
    - Gabor filtered image.
    """
    gabor_response_real, gabor_response_imag = gabor(
        image, frequency=frequency, theta=theta
    )

    if visualize:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Gabor Filtered Image (Real)")
        plt.imshow(gabor_response_real, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Gabor Filtered Image (Imaginary)")
        plt.imshow(gabor_response_imag, cmap="gray")
        plt.axis("off")

        plt.show()

    return gabor_response_real, gabor_response_imag


def extract_texture_features(image):
    glcm_features = calculate_glcm_features(image, distances=[1], angles=[0])
    lbp_features = extract_lbp_features(image)
    haralick_features = extract_haralick_features(image)
    gabor_real, gabor_imag = calculate_gabor_features(image, frequency=0.6, theta=0)

    # Compute statistics from the Gabor filter response
    mean_real = np.mean(gabor_real)
    std_real = np.std(gabor_real)
    mean_imag = np.mean(gabor_imag)
    std_imag = np.std(gabor_imag)

    features = {
        "glcm_contrast": glcm_features["contrast"],
        "glcm_correlation": glcm_features["correlation"],
        "glcm_energy": glcm_features["energy"],
        "glcm_homogeneity": glcm_features["homogeneity"],
        "mean_gabor_real": mean_real,
        "std_gabor_real": std_real,
        "mean_gabor_imag": mean_imag,
        "std_gabor_imag": std_imag,
    }

    for i, feature in enumerate(lbp_features):
        features[f"lbp_{i}"] = feature

    for i, feature in enumerate(haralick_features):
        features[f"haralick_{i}"] = feature

    return features


def apply_sobel_operator(image, visualize=False):
    """
    Apply the Sobel operator to highlight edges in the image.

    Parameters:
    - image: 2D numpy array, preprocessed image.

    Returns:
    - Sobel filtered image.
    """
    sobel_edges = sobel(image)

    if visualize:
        plt.figure(figsize=(8, 4))
        plt.title("Sobel Filtered Image")
        plt.imshow(sobel_edges, cmap="gray")
        plt.axis("off")
        plt.show()

    return sobel_edges


def apply_canny_edge_detector(image, sigma=1, visualize=False):
    """
    Use the Canny edge detector for identifying edges with more accuracy.

    Parameters:
    - image: 2D numpy array, preprocessed image.
    - sigma: Standard deviation of the Gaussian filter used in Canny edge detector.

    Returns:
    - Canny edge detected image.
    """
    canny_edges = canny(image, sigma=sigma)

    if visualize:
        plt.figure(figsize=(8, 4))
        plt.title("Canny Edge Detected Image")
        plt.imshow(canny_edges, cmap="gray")
        plt.axis("off")
        plt.show()

    return canny_edges


def extract_edges_features(image):
    sobel_edges = apply_sobel_operator(image)
    canny_edges = apply_canny_edge_detector(image)

    # Calculate the number of edges in the image
    sobel_edges_count = np.sum(sobel_edges > 0)
    canny_edges_count = np.sum(canny_edges > 0)

    return {
        "sobel_edges_count": sobel_edges_count,
        "canny_edges_count": canny_edges_count,
    }


def calculate_hog_features(
    image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False
):
    """
    Calculate HOG features to outline the presence of specific shapes or orientations.

    Parameters:
    - image: 2D numpy array, preprocessed image.
    - pixels_per_cell: Size (in pixels) of a cell.
    - cells_per_block: Number of cells in each block.
    - visualize: If True, return an image of the HOG.

    Returns:
    - HOG features, and optionally the HOG image.
    """
    fd, hog_image = hog(
        image,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize,
        feature_vector=True,
    )
    if visualize:
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        plt.figure(figsize=(8, 4))
        plt.title("HOG Image")
        plt.imshow(hog_image_rescaled, cmap="gray")
        plt.axis("off")
        plt.show()

    return fd, hog_image


def extract_shape_features(image):
    hog_features, _ = calculate_hog_features(image)

    features = {"hog_" + str(i): hog_features[i] for i in range(len(hog_features))}

    return features


def calculate_statistical_features(image):
    """
    Calculate statistical features: skewness, kurtosis, and entropy.

    Parameters:
    - image: 2D numpy array, preprocessed image.

    Returns:
    - A dictionary containing the calculated features.
    """
    # Flatten the image to 1D for statistical analysis
    flattened_image = image.flatten()

    # Calculate skewness
    image_skewness = skew(flattened_image)

    # Calculate kurtosis
    image_kurtosis = kurtosis(flattened_image)

    # Calculate entropy
    histogram, _ = np.histogram(flattened_image, bins=256, range=(0, 255))
    histogram_normalized = histogram / histogram.sum()
    entropy = -np.sum(
        histogram_normalized * np.log2(histogram_normalized + np.finfo(float).eps)
    )  # Add epsilon to avoid log(0)

    return {"skewness": image_skewness, "kurtosis": image_kurtosis, "entropy": entropy}


def extract_wavelet_features(image, mode="haar", level=1, visualize=False):
    """
    Extract Wavelet Transform features.
    Parameters:
    - image: 2D numpy array, preprocessed image.
    - mode: Type of wavelet to use.
    - level: Decomposition level.
    Returns:
    - Concatenated wavelet coefficients.
    """
    coeffs = pywt.wavedec2(image, wavelet=mode, level=level)
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]
    features = np.concatenate([cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel()])

    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes[0, 0].imshow(cA, cmap="gray")
        axes[0, 0].set_title("Approximation")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(cH, cmap="gray")
        axes[0, 1].set_title("Horizontal Detail")
        axes[0, 1].axis("off")
        axes[1, 0].imshow(cV, cmap="gray")
        axes[1, 0].set_title("Vertical Detail")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(cD, cmap="gray")
        axes[1, 1].set_title("Diagonal Detail")
        axes[1, 1].axis("off")
        plt.show()

    return features


def extract_fourier_features(image, visualize=False):
    """
    Extract Fourier Transform features.
    Parameters:
    - image: 2D numpy array, preprocessed image.
    Returns:
    - Flattened array of Fourier coefficients.
    """
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)  # Adding 1 to avoid log(0)

    if visualize:
        plt.imshow(magnitude_spectrum, cmap="gray")
        plt.title("Fourier Transform")
        plt.show()

    return magnitude_spectrum.ravel()


def extract_frequency_features(image):
    wavelet_features = extract_wavelet_features(image, visualize=False)
    fourier_features = extract_fourier_features(image, visualize=False)

    features = {
        "wavelet_" + str(i): wavelet_features[i] for i in range(len(wavelet_features))
    }
    features.update(
        {"fourier_" + str(i): fourier_features[i] for i in range(len(fourier_features))}
    )

    return features


def extract_pixels_features(image):
    intensity_features = extract_intensity_features(image)
    texture_features = extract_texture_features(image)
    edges_features = extract_edges_features(image)
    statistical_features = calculate_statistical_features(image)

    features = {
        **intensity_features,
        **texture_features,
        **edges_features,
        **statistical_features,
    }

    return features


def process_file(hdf5_path, filename):
    with h5py.File(hdf5_path, "r") as f:
        if filename in f:
            pixel_array = f[filename][:]
            features = extract_pixels_features(pixel_array)
            return features
    return None


def check_csv_header(csv_path, fieldnames):
    if not os.path.isfile(csv_path) or os.stat(csv_path).st_size == 0:
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()


def save_batch_to_csv(csv_path, batch_data):
    # Append mode without writing headers every time
    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=batch_data[0].keys())
        writer.writerows(batch_data)


def process_batch(hdf5_path, file_batch):
    """
    Processes a batch of files, extracting features for each.
    """
    features_list = []
    with h5py.File(hdf5_path, "r") as f:
        for filename in file_batch:
            if filename in f:
                pixel_array = f[filename][:]
                features = extract_pixels_features(pixel_array)
                features_list.append(features)
    return features_list


def parallel_pixels_features_extraction(hdf5_path, csv_path):
    with h5py.File(hdf5_path, "r") as f:
        hdf5_keys = list(f.keys())

    num_processes = min(multiprocessing.cpu_count(), len(hdf5_keys) // 10 + 1)
    batches = [hdf5_keys[i : i + 10] for i in range(0, len(hdf5_keys), 10)]

    # Assuming `process_file` function has been modified accordingly
    first_image_features = process_file(hdf5_path, hdf5_keys[0])
    fieldnames = list(first_image_features.keys())

    check_csv_header(csv_path, fieldnames)

    with tqdm(total=len(hdf5_keys), desc="Overall Progress", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(process_batch, hdf5_path, batch) for batch in batches
            ]
            for future in as_completed(futures):
                batch_data = future.result()
                save_batch_to_csv(csv_path, batch_data)
                pbar.update(len(batch_data))


if __name__ == "__main__":
    hdf5_path = "/Volumes/ALEXIS/ai_project_cranfield/dicom_images_final.hdf5"
    csv_path = "/Volumes/ALEXIS/ai_project_cranfield/dicom_pixels_features.csv"
    parallel_pixels_features_extraction(hdf5_path, csv_path)
