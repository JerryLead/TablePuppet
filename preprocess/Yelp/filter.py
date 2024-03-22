"""
Keep only restaurants.
"""
import pandas as pd

review_table = pd.read_csv("csv/yelp_dataset_review.csv")
business_table = pd.read_csv("csv/yelp_dataset_business.csv")
user_table = pd.read_csv("csv/yelp_dataset_user.csv")

business_table = business_table[
    business_table["business_id"].isin(business_table["business_id"])
    & business_table["categories"].str.lower().str.contains("restaurants")
]
review_table = review_table[
    review_table["business_id"].isin(business_table["business_id"])
]
user_table = user_table[user_table["user_id"].isin(review_table["user_id"])]

business_table.to_csv("filtered/restaurant.csv", index=False)
review_table.to_csv("filtered/review.csv", index=False)
user_table.to_csv("filtered/user.csv", index=False)
