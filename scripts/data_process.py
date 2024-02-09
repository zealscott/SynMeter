# Description:
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from lib.commons import read_csv
from lib.info import ROOT_DIR


REAL_DATA_PATH = ROOT_DIR + "/datasets"


def split_data_stratify(data, test_size=0.2, val_size=0.2):
    """
    data: pandas dataframe
    split data into train, val, and test
    use stratify to make sure the distribution of the data/label is the same
    do not use any preprocessing, return dataframe
    """
    # split data into data and label
    label = data["label"]
    data = data.drop(columns=["label"])
    # split data into train, val, and test
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, test_size=test_size, random_state=0, stratify=label
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=0, stratify=y_train
    )
    # combine data and label
    train = pd.concat([x_train, y_train], axis=1)
    val = pd.concat([x_val, y_val], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    return train, val, test


def get_label_name(dataset):
    """
    get the label name of the dataset
    only used for LLM models
    """
    orginal_data_path = os.path.join(REAL_DATA_PATH, dataset, "original",dataset + ".csv")
    data = pd.read_csv(orginal_data_path)
    return data.columns[-1] # the last column is the label by default

def split_data(data, test_size=0.2, val_size=0.2):
    """
    data: pandas dataframe
    split data into train, val, and test
    do not use any preprocessing, return dataframe
    """
    train, test = train_test_split(data, test_size=test_size, random_state=0)
    train, val = train_test_split(train, test_size=val_size, random_state=0)
    return train, val, test


def main():
    """
    process real datasets, split into train, val, and test
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="adult")

    args = parser.parse_args()

    item_path = os.path.join(REAL_DATA_PATH, args.dataset)
    print("processing " + item_path)

    csv_filename = os.path.join(item_path, args.dataset + ".csv")
    meta_filename = os.path.join(item_path, args.dataset + ".json")
    data, meta_data, discrete_cols = read_csv(csv_filename, meta_filename)
    if meta_data["task"] != "regression":
        # split data with stratify
        train, val, test = split_data_stratify(data, test_size=0.2, val_size=0.2)
    else:
        train, val, test = split_data(data, test_size=0.2, val_size=0.2)

    # add the data size of train, val, test to meta_data
    meta_data["train_size"] = len(train)
    meta_data["val_size"] = len(val)
    meta_data["test_size"] = len(test)
    # the last column is the label by default, we only need the name for label for LLM models
    meta_data["label"] = get_label_name(args.dataset)
    # rewrite the meta_data
    json.dump(meta_data, open(meta_filename, "w"))
    # save the train, val, test data
    train.to_csv(os.path.join(item_path, "train.csv"), index=False)
    val.to_csv(os.path.join(item_path, "val.csv"), index=False)
    test.to_csv(os.path.join(item_path, "test.csv"), index=False)

    # calculate the number of categorical and numerical columns from meta_data
    categorical_cols = []
    numerical_cols = []
    for col in meta_data["columns"]:
        if col["type"] == "categorical":
            categorical_cols.append(col["name"])
        elif col["type"] == "continuous":
            numerical_cols.append(col["name"])
        else:
            raise ValueError("column type should be categorical or continuous")
    print("-" * 20)
    print("number of numerical columns: ", len(numerical_cols))
    print("number of categorical columns: ", len(categorical_cols))

    print("-" * 20)
    print("train size: ", len(train))
    print("val size: ", len(val))
    print("test size: ", len(test))


if __name__ == "__main__":
    main()
