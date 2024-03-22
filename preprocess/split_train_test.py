from sklearn.model_selection import train_test_split
import pandas as pd
import os
import shutil
import argparse


parser = argparse.ArgumentParser(description="Split train dataset and test dataset.")
parser.add_argument("data_path", help="Data path")
parser.add_argument(
    "--split_file", required=True, type=str, help="The file to be splited"
)

args = parser.parse_args()

copy_files = []
for file in os.listdir(args.data_path):
    if file.endswith(".csv") and file != args.split_file:
        copy_files.append(file)

train_path = os.path.join(args.data_path, "train")
test_path = os.path.join(args.data_path, "test")
if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)

split_df = pd.read_csv(os.path.join(args.data_path, args.split_file))
train_df, test_df = train_test_split(split_df, test_size=0.15, random_state=42)
train_df.reset_index(drop=True)
test_df.reset_index(drop=True)
train_df.to_csv(os.path.join(train_path, args.split_file), index=False)
test_df.to_csv(os.path.join(test_path, args.split_file), index=False)

for file in copy_files:
    shutil.copy(os.path.join(args.data_path, file), os.path.join(train_path, file))
    shutil.copy(os.path.join(args.data_path, file), os.path.join(test_path, file))
