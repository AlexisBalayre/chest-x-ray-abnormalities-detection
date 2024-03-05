import pandas as pd
from sklearn.model_selection import train_test_split


def spit_data(data_path, output_path, test_size=0.2, random_state=42):
    data = pd.read_csv(data_path)
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state, shuffle=True
    )
    train.to_csv(output_path + "train.csv", index=False)
    test.to_csv(output_path + "test.csv", index=False)
    return train, test


if __name__ == "__main__":
    spit_data("train.csv", "data/")
