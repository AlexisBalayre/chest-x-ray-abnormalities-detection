import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    data_path, output_path, test_size=0.15, val_size=0.1, random_state=42
):
    """
    Splits a dataset into training, validation, and test sets, and saves them to specified paths.

    Parameters:
    - data_path (str): The path to the CSV file containing the dataset to be split.
    - output_path (str): The base path where the split datasets will be saved.
    - test_size (float): The proportion of the dataset to include in the test split.
    - val_size (float): The proportion of the training dataset to include in the validation split.
    - random_state (int): The seed used by the random number generator.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test DataFrames.
    """

    # Load the dataset from a CSV file
    data = pd.read_csv(data_path)

    # Separate features and the target variable
    X = data.drop("class_id", axis=1)
    y = data["class_id"]

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Further split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    # Combine the features and target variable for each set
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    val = pd.concat([X_val, y_val], axis=1)

    # Save the datasets to CSV files
    train.to_csv(output_path + "train/train.csv", index=False)
    val.to_csv(output_path + "train/val.csv", index=False)
    test.to_csv(output_path + "test/test.csv", index=False)

    # Return the training and test DataFrames
    return train, test


if __name__ == "__main__":
    input_file_path = "train.csv"
    output_folder_path = "data/"
    # Call the function with the specified input file and output folder paths
    split_dataset(input_file_path, output_folder_path)
