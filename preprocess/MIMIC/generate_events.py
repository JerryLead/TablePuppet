import os
import sys

sys.path.insert(0, os.path.abspath("."))

from mimic3benchmark.readers import DecompensationReader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from mimic3models import common_utils

import numpy as np
import pandas as pd
import argparse


def read_and_extract_features(reader, count, period, features):
    read_chunk_size = 1000
    Xs = []
    ys = []
    names = []
    ts = []
    for i in range(0, count, read_chunk_size):
        j = min(count, i + read_chunk_size)
        ret = common_utils.read_chunk(reader, j - i)
        X = common_utils.extract_features_from_rawdata(
            ret["X"], ret["header"], period, features
        )
        Xs.append(X)
        ys += ret["y"]
        names += ret["name"]
        ts += ret["t"]
    Xs = np.concatenate(Xs, axis=0)
    return (Xs, ys, names, ts)


def transform_names_to_id(names, type="train"):
    ids = []
    for name in names:
        [subject_id, file_name, _] = name.split("_")
        path = os.path.join("data", "root", type, subject_id, file_name + ".csv")
        episode = pd.read_csv(path)
        if len(episode["Icustay"]) != 1:
            raise Exception("Run exception")
        id = episode["Icustay"][0]
        ids.append(id)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data of decompensation task",
        default=os.path.join(
            os.path.dirname(__file__),
            "/home/xxx/machine_learning/mimic3-benchmarks/data/decompensation",
        ),
    )
    parser.add_argument(
        "--period",
        type=str,
        default="all",
        help="specifies which period extract features from",
        choices=[
            "first4days",
            "first8days",
            "last12hours",
            "first25percent",
            "first50percent",
            "all",
        ],
    )
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        help="specifies what features to extract",
        choices=["all", "len", "all_but_len"],
    )
    args = parser.parse_args()

    train_reader = DecompensationReader(
        dataset_dir=os.path.join(args.data, "train"),
        listfile=os.path.join(args.data, "train_listfile.csv"),
    )

    val_reader = DecompensationReader(
        dataset_dir=os.path.join(args.data, "train"),
        listfile=os.path.join(args.data, "val_listfile.csv"),
    )

    test_reader = DecompensationReader(
        dataset_dir=os.path.join(args.data, "test"),
        listfile=os.path.join(args.data, "test_listfile.csv"),
    )

    print("Reading data and extracting features ...")

    (test_X, test_y, test_names, test_ts) = read_and_extract_features(
        test_reader, test_reader.get_number_of_examples(), args.period, args.features
    )

    test_ids = transform_names_to_id(test_names, type="test")

    (val_X, val_y, val_names, val_ts) = read_and_extract_features(
        val_reader, val_reader.get_number_of_examples(), args.period, args.features
    )

    valid_ids = transform_names_to_id(val_names)

    (train_X, train_y, train_names, train_ts) = read_and_extract_features(
        train_reader, train_reader.get_number_of_examples(), args.period, args.features
    )

    print("Imputing missing values ...")
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", copy=True)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    print("Normalizing the data to have zero mean and unit variance ...")
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    train_ids = transform_names_to_id(train_names)

    X = np.vstack((train_X, val_X, test_X))
    y = train_y + val_y + test_y
    ids = train_ids + valid_ids + test_ids
    hours = train_ts + val_ts + test_ts

    df = pd.DataFrame(X, columns=[f"{i}" for i in range(1, X.shape[1] + 1)])
    df.insert(0, "ICUSTAY_ID", ids)
    df.insert(1, "HOUR", hours)
    df.insert(2, "LABEL", y)

    df.to_csv("events.csv", index=False)


if __name__ == "__main__":
    main()
