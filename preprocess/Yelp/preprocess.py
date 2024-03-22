"""
Make the table easy to handle, with only numerical values.
"""
import pandas as pd
import os

source_path = "xxx/dataset/Yelp/filtered"
target_path = "xxx/dataset/Yelp/processed"

if not os.path.exists(target_path):
    os.mkdir(target_path)

restaurant = pd.read_csv(os.path.join(source_path, "restaurant.csv"))
restaurant.drop(
    columns=[
        "name",
        "address",
        "city",
        "postal_code",
        "latitude",
        "longitude",
        "categories",
        "hours",
        "attributes",
        "state",
    ],
    inplace=True,
)
restaurant.dropna(inplace=True)
restaurant = restaurant[restaurant["is_open"] == 1]
restaurant.drop(columns=["is_open"], inplace=True)

user = pd.read_csv(os.path.join(source_path, "user.csv"))
user.drop(columns=["name", "elite"], inplace=True)
user.dropna(inplace=True)
user["friends"] = user["friends"].str.split(",").apply(len)
user["yelping_since"] = pd.to_datetime(user["yelping_since"])
user["year_used"] = pd.Timestamp.now().year - user["yelping_since"].dt.year
user.drop(columns=["yelping_since"], inplace=True)

review = pd.read_csv(os.path.join(source_path, "review.csv"))
review.drop(columns=["date"], inplace=True)

embedding = pd.read_csv(os.path.join(source_path, "embedding_768.csv"))
embedding = embedding.add_prefix("embedding_")

review_with_embedding = pd.concat([review, embedding], axis=1)
del review, embedding
review_with_embedding.drop(columns=["stars", "text"], inplace=True)
review_with_embedding.rename(columns={"embedding_label": "label"}, inplace=True)
review_with_embedding.dropna(inplace=True)
review_with_embedding["label"] = review_with_embedding["label"].astype(int)

merged_dataframe = pd.merge(review_with_embedding, restaurant, on="business_id").merge(
    user, on="user_id", suffixes=("_business", "_user")
)

review_with_embedding = review_with_embedding[
    review_with_embedding["review_id"].isin(merged_dataframe["review_id"])
]
user = user[user["user_id"].isin(merged_dataframe["user_id"])]
restaurant = restaurant[restaurant["business_id"].isin(merged_dataframe["business_id"])]

restaurant.to_csv(os.path.join(target_path, "restaurant.csv"), index=False)
user.to_csv(os.path.join(target_path, "user.csv"), index=False)
review_with_embedding.to_csv(os.path.join(target_path, "review.csv"), index=False)

merged_dataframe = merged_dataframe.assign(label=merged_dataframe.pop("label"))
merged_dataframe.drop(columns=["review_id", "user_id", "business_id"], inplace=True)
merged_dataframe.to_csv(os.path.join(target_path, "joined.csv"), index=False)
