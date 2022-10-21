import time
import torch
import pickle
import torch.onnx
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from transformers import logging, AutoTokenizer

from models.architecture import DistilBERT
from models.train_model import train, evaluate

from onnxruntime.quantization import quantize_dynamic, QuantType


def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------- LOGGING LEVEL ---------- #
logging.set_verbosity_error()

# ---------- HYPERPARAMETERS ---------- #
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 4

# ---------- LOAD DATA ---------- #
with open("data/processed/encoded_data.train", "rb") as handle:
    encoded_data_train = pickle.load(handle)

with open("data/processed/encoded_data.val", "rb") as handle:
    encoded_data_val = pickle.load(handle)

with open("data/processed/encoded_data.test", "rb") as handle:
    encoded_data_test = pickle.load(handle)

with open("data/processed/labels.train", "rb") as handle:
    train_labels = pickle.load(handle)

with open("data/processed/labels.val", "rb") as handle:
    valid_labels = pickle.load(handle)

with open("data/processed/labels.test", "rb") as handle:
    test_labels = pickle.load(handle)

# ---------- DATALOADER PREPARATION --------- #

# Train-set
input_ids_train = encoded_data_train["input_ids"]
attention_masks_train = encoded_data_train["attention_mask"]
# token_type_ids_train = encoded_data_train["token_type_ids"]
labels_train = torch.tensor(train_labels)
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataloader_train = DataLoader(dataset_train, sampler=None, batch_size=BATCH_SIZE)

# Test-set
input_ids_test = encoded_data_test["input_ids"]
attention_masks_test = encoded_data_test["attention_mask"]
# token_type_ids_test = encoded_data_test["token_type_ids"]
labels_test = torch.tensor(test_labels)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
dataloader_test = DataLoader(dataset_test, sampler=None, batch_size=BATCH_SIZE)

# Validation-set
input_ids_val = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val["attention_mask"]
# token_type_ids_val = encoded_data_val["token_type_ids"]
labels_val = torch.tensor(valid_labels)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataloader_val = DataLoader(dataset_val, sampler=None, batch_size=BATCH_SIZE)

# ---------- START TRAIN LOOP ---------- #
model = DistilBERT()

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay=0.1
)
loss_func = torch.nn.CrossEntropyLoss()

model.to(device)
loss_func.to(device)

validation_threshold = np.inf
for epoch in range(EPOCHS):

    start_time = time.time()
    train_loss = train(model, dataloader_train, optimizer, loss_func, device)
    valid_loss, valid_acc = evaluate(
        model, dataloader_val, loss_func, device, print_report=False
    )
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(
        f"\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Val. F1: {valid_acc*100:.2f}%"
    )

    if valid_loss < validation_threshold:
        validation_threshold = valid_loss
    else:
        print(f"Early stopping ...")
        break

# Calculate test loss and accuracy
test_loss, test_acc = evaluate(
    model, dataloader_test, loss_func, device, print_report=True
)
print(f"Test Loss: {test_loss:.3f} | Test f1: {test_acc*100:.2f}%")

# ---------- SAVE MODEL IN ONNX FORMAT ---------- #
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

dummy_model_input = tokenizer(
    "This is a sample.",
    return_tensors="pt",
    add_special_tokens=True,
    return_token_type_ids=False,
    return_attention_mask=True,
)

torch.onnx.export(
    model.to("cpu"),
    tuple(dummy_model_input.values()),
    f="models/model_fp32.onnx",
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "token_type_ids": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
    },
    do_constant_folding=True,
    opset_version=11,
    export_params=True,
    verbose=True,
)
print("Full precision ONNX model exported at ", "models/model_fp32.onnx")

# ---------- QUANTIZE ONNX MODEL ---------- #
model_fp32 = "models/model_fp32.onnx"
model_quant = "models/model_8bit.onnx"
quantized_model = quantize_dynamic(model_fp32, model_quant)
print("Quantized ONNX model exported at ", "models/model_8bit.onnx")
