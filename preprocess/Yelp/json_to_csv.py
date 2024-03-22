"""
Transform the dataset from json format to csv format.
"""
import json
import csv
from tqdm import tqdm
import os

data_base_path = "xxx/dataset/Yelp"

json_files = {
    "business": os.path.join(
        data_base_path, "raw", "json", "yelp_academic_dataset_business.json"
    ),
    "checkin": os.path.join(
        data_base_path, "raw", "json", "yelp_academic_dataset_checkin.json"
    ),
    "review": os.path.join(
        data_base_path, "raw", "json", "yelp_academic_dataset_review.json"
    ),
    "tip": os.path.join(
        data_base_path, "raw", "json", "yelp_academic_dataset_tip.json"
    ),
    "user": os.path.join(
        data_base_path, "raw", "json", "yelp_academic_dataset_user.json"
    ),
}

csv_files = {
    "business": os.path.join(data_base_path, "raw", "csv", "yelp_dataset_business.csv"),
    "checkin": os.path.join(data_base_path, "raw", "csv", "yelp_dataset_checkin.csv"),
    "review": os.path.join(data_base_path, "raw", "csv", "yelp_dataset_review.csv"),
    "tip": os.path.join(data_base_path, "raw", "csv", "yelp_dataset_tip.csv"),
    "user": os.path.join(data_base_path, "raw", "csv", "yelp_dataset_user.csv"),
}

total_lines = {}

for name in json_files.keys():
    print(f"Begin getting {name} line number")
    with open(json_files[name], encoding="utf-8") as f:
        total_lines[name] = sum(1 for _ in f)
    print(f"Finish getting {name} line number")

for name in json_files.keys():
    print(f"Begin converting {name}")
    with open(json_files[name], "r", encoding="utf-8") as input_file, open(
        csv_files[name], "w", newline="", encoding="utf-8"
    ) as output_file:
        csv_writer = csv.writer(output_file)
        csv_header = None
        for line in tqdm(input_file, total=total_lines[name]):
            data = json.loads(line)
            if csv_header is None:
                csv_header = data.keys()
                csv_writer.writerow(csv_header)
            csv_writer.writerow(data.values())
