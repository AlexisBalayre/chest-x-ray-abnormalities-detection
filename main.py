import pydicom
import os

# Path to your DICOM file
train_data_folder = 'train_data'

for dirName, subdirList, fileList in os.walk(train_data_folder):
    for filename in fileList:
        if ".dicom" in filename.lower():
            dicom_file_path = os.path.join(dirName, filename)
            ds = pydicom.dcmread(dicom_file_path)
            print(ds)
