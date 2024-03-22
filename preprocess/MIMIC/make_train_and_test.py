import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

import os

import utils


def split_train_test():
    df = pd.read_csv("train/stays.csv")
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df.reset_index(drop=True)
    test_df.reset_index(drop=True)
    train_df.to_csv("train/stays.csv", index=False)
    test_df.to_csv("test/stays.csv", index=False)


def transform_data(data_path, target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    for csv_file in csv_files:
        table_name = os.path.splitext(csv_file)[0]
        data = pd.read_csv(os.path.join(data_path, csv_file))
        if csv_file == "admissions.csv":
            data = utils.one_hot_code(data, "ETHNICITY")
        elif csv_file == "patients.csv":
            gender_map = {"F": 0, "M": 1}
            data["GENDER"] = data["GENDER"].map(gender_map)
        elif csv_file == "stays.csv":
            data = utils.one_hot_code(data, "FIRST_CAREUNIT")
            data = utils.one_hot_code(data, "LAST_CAREUNIT")
            data = utils.one_hot_code(data, "FIRST_WARDID")
            data = utils.one_hot_code(data, "LAST_WARDID")
        data = data.sample(frac=1, random_state=42)
        data.reset_index(drop=True)
        data.to_csv(os.path.join(target_path, csv_file), index=False)


def copy_files():
    source_folder = "train"
    target_folder = "test"

    for file_name in os.listdir(source_folder):
        if file_name.endswith(".csv"):
            source_file = os.path.join(source_folder, file_name)
            target_file = os.path.join(target_folder, file_name)
            shutil.copy(source_file, target_file)


if __name__ == "__main__":
    transform_data("data", "train")
    copy_files()
    split_train_test()
