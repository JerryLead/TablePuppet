import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
import os


tqdm.pandas()

data_path = "origin"
target_path = "processed"

movies = pd.read_csv(
    os.path.join(data_path, "movies.dat"),
    delimiter="::",
    encoding="ISO-8859-1",
    header=None,
    index_col=0,
)

movies.rename_axis("movieid", inplace=True)
movies.columns = ["title", "genres"]

movies[["title", "year"]] = movies["title"].str.extract("(.*)(\(\d{4}\))")
movies["year"] = movies["year"].str.strip("()")

movies = movies.assign(genres=movies["genres"].str.split("|")).explode("genres")

genres = movies["genres"].str.get_dummies()
movies = pd.concat([movies, genres], axis=1)
movies.drop("genres", axis=1, inplace=True)

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

movies.to_csv(os.path.join(target_path, "movies_genres_split.csv"))
