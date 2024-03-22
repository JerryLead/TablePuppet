from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import os

data_base_path = "xxx/dataset/Yelp"

review_table = pd.read_csv(os.path.join(data_base_path, "filtered", "review.csv"))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

device = torch.device("cuda:1")
model.to(device)

embeddings = []
with tqdm(total=review_table.shape[0]) as pbar:
    for review in review_table["text"]:
        if not isinstance(review, str):
            continue
        encoded_review = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded_review["input_ids"].to(device)
        attention_mask = encoded_review["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0, :]
            embeddings.append(embedding)
        pbar.update(1)

# Save embeddings
embedding_table = pd.DataFrame(embeddings)
embedding_table["label"] = review_table["stars"]
embedding_table.to_csv(
    os.path.join(data_base_path, "filtered", "embedding.csv"), index=False
)
