import json
import pickle
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

dataset = pd.DataFrame(columns=["headline", "is_sarcastic"])
fullpath = "data/raw/Sarcasm_Headlines_Dataset.json"
headlines = []
targets = []

with open(fullpath) as file:
    for line in file:
        _json = json.loads(line)
        headlines.append(_json["headline"])
        targets.append(_json["is_sarcastic"])

dataset["headline"] = headlines
dataset["is_sarcastic"] = targets

dataset = dataset.dropna()
dataset = dataset.sample(frac=1).reset_index(drop=True)
print(dataset.shape)

X_train, X_test, y_train, y_test = train_test_split(
    dataset["headline"].values,
    dataset["is_sarcastic"].values,
    test_size=0.2,
    random_state=42,
)
X_test, X_valid, y_test, y_valid = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

encoded_data_train = tokenizer(
    X_train.tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
    add_special_tokens=True,
    return_token_type_ids=False,
    return_attention_mask=True,
)

encoded_data_val = tokenizer(
    X_valid.tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
    add_special_tokens=True,
    return_token_type_ids=False,
    return_attention_mask=True,
)

encoded_data_test = tokenizer(
    X_test.tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
    add_special_tokens=True,
    return_token_type_ids=False,
    return_attention_mask=True,
)

with open("data/processed/encoded_data.train", "wb") as handle:
    pickle.dump(encoded_data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/processed/encoded_data.val", "wb") as handle:
    pickle.dump(encoded_data_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/processed/encoded_data.test", "wb") as handle:
    pickle.dump(encoded_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/processed/labels.train", "wb") as handle:
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/processed/labels.val", "wb") as handle:
    pickle.dump(y_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/processed/labels.test", "wb") as handle:
    pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
