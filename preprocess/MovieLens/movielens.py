"""
MovieLens-1M dataset preprocessing code
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

from tqdm.auto import tqdm
from loguru import logger

import argparse
import sys
import os

tqdm.pandas()

config = {
    "handlers": [
        {"sink": sys.stdout, "format": "{time: YYYY-MM-DD HH:mm:ss} - {message}"},
    ],
}
logger.configure(**config)

parser = argparse.ArgumentParser(description="preprocess MovieLens-1M dataset.")
parser.add_argument(
    "data_path", type=str, help="Original data path of MovieLens-1M dataset"
)
parser.add_argument(
    "target_path", type=str, help="Target data path of MovieLens-1M dataset"
)

args = parser.parse_args()

data_path = args.data_path
target_path = args.target_path

if not os.path.exists(target_path):
    os.mkdir(target_path)

# Deal with movies
logger.info("Begin dealing with movies")
movies = pd.read_csv(
    os.path.join(data_path, "movies.dat"),
    delimiter="::",
    encoding="ISO-8859-1",
    header=None,
    index_col=0,
)

# Add index name and column name
movies.rename_axis("movieid", inplace=True)
movies.columns = ["title", "genres"]
logger.info("Added index name and column name")

# Extract time from the movie title into a separate column
movies[["title", "year"]] = movies["title"].str.extract("(.*)(\(\d{4}\))")
movies["year"] = movies["year"].str.strip("()")
logger.info("Extracted time from the movie title")

# One-hot encode genres
genres = movies["genres"].str.get_dummies()
movies = pd.concat([movies, genres], axis=1)
movies.drop("genres", axis=1, inplace=True)
logger.info("One-hot encoded genres")

# get title embedding
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


def encode_title(title, embedding_size):
    input_ids = tokenizer.encode(title, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state[:, 0, :embedding_size].numpy()[0, :]
    return embeddings


movies["title_embeddings"] = movies["title"].progress_apply(
    encode_title, embedding_size=32
)
title_embeddings = movies["title_embeddings"].apply(lambda x: pd.Series(x))
title_embeddings.columns = [
    f"title_embedding_{i}" for i in range(len(title_embeddings.columns))
]
movies = pd.concat([movies, title_embeddings], axis=1)
movies.drop(columns=["title", "title_embeddings"], inplace=True)
logger.info("Generated title embeddings")

movies.to_csv(os.path.join(target_path, "movies.csv"))
logger.info(f"Saved movies to {os.path.join(target_path, 'movies.csv')}")

del movies

# Deal with users
logger.info("Begin dealing with users")
users = pd.read_csv(
    os.path.join(data_path, "users.dat"), delimiter="::", header=None, index_col=0
)

# Add index name and column name
users.rename_axis("userid", inplace=True)
users.columns = ["gender", "age", "occupation", "zip-code"]
users.drop(columns=["zip-code"], inplace=True)
logger.info("Added index name and column name")

# Map gender to 0 or 1
gender_map = {"F": 0, "M": 1}
users["gender"] = users["gender"].map(gender_map)
logger.info("Finished mapping gender to 0 or 1")

# One-hot encode occupation
occupations = pd.get_dummies(users["occupation"])
occupations.columns = [f"occupation_{i}" for i in range(len(occupations.columns))]
users = pd.concat([users, occupations], axis=1)
users.drop(columns=["occupation"], inplace=True)
logger.info("One-hot encoded occupations")

users.to_csv(os.path.join(target_path, "users.csv"))
logger.info(f"Saved users to {os.path.join(target_path, 'users.csv')}")

del users

# Deal with ratings
logger.info("Begin dealing with ratings")
ratings = pd.read_csv(
    os.path.join(data_path, "ratings.dat"), delimiter="::", header=None
)
ratings.columns = ["userid", "movieid", "rating", "timestamp"]
logger.info("Added index name and column name")

ratings.to_csv(os.path.join(target_path, "ratings.csv"), index=False)
logger.info(f"Saved ratings to {os.path.join(target_path, 'ratings.csv')}")
