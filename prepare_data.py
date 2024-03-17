from preprocessing.dicom_metadata_extraction import extract_dicom_metadata
from preprocessing.dicom_pixels_processing import extract_dicom_images
from preprocessing.dicom_pixels_features_extraction import extract_images_features

dicom_files_test_folder = "/Volumes/ALEXIS/ai_project_cranfield/test"
dicom_metadata_test_csv_path = "./dicom_metadata_test.csv"
dicom_images_features_test_csv_path = "./dicom_pixels_features_test.csv"
hdf5_test_path = "/Volumes/ALEXIS/ai_project_cranfield/dicom_images_final_test.hdf5"

if __name__ == "__main__":
    print("Extracting DICOM metadata...")
    extract_dicom_metadata(
        dicom_files_test_folder, dicom_metadata_test_csv_path, batch_size=10
    )
    print("Done!")

    print("Processing and writing DICOM images to HDF5...")
    extract_dicom_images(input_folder, hdf5_test_path)
    print("Done!")

    print("Extracting DICOM images features...")
    extract_images_features(hdf5_path, dicom_images_features_test_csv_path)
    print("Done!")
