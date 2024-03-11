import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    data_path, output_path, test_size=0.15, val_size=0.1, random_state=42
):
    data = pd.read_csv(data_path)
    X = data.drop("class_id", axis=1)
    y = data["class_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    val = pd.concat([X_val, y_val], axis=1)

    train.to_csv(output_path + "train.csv", index=False)
    test.to_csv(output_path + "test.csv", index=False)
    val.to_csv(output_path + "val.csv", index=False)
    return train, test


if __name__ == "__main__":
    split_dataset("train.csv", "data/")
