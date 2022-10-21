import time
import numpy as np
import onnxruntime
import streamlit as st
from transformers import AutoTokenizer, logging

logging.set_verbosity_error()


class DistilBERTInference:
    def __init__(self):

        self.full_ort_session = onnxruntime.InferenceSession(
            "models/model_fp32.onnx", providers=["CUDAExecutionProvider"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.classes = {0: "normal", 1: "sarcastic"}

    def tokenize(self, raw_input):

        self.encoded_input = self.tokenizer(
            raw_input,
            return_tensors="np",
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
        )

        self.ids = self.encoded_input["input_ids"]
        self.mask = self.encoded_input["attention_mask"]

    def predict(self):

        full_ort_outputs = self.full_ort_session.run(
            None,
            {
                "input_ids": self.ids,
                "attention_mask": self.mask,
            },
        )

        pred = int(np.argmax(full_ort_outputs[0], axis=1))
        return self.classes[pred]


@st.experimental_singleton
def load_model():
    return DistilBERTInference()


inference = load_model()
raw_input = st.text_input(label="Sarcasm detector", placeholder="Enter text ...")
if raw_input:

    time_now = time.time()
    inference.tokenize(raw_input)
    full_out = inference.predict()
    time_passed_full = str(time.time() - time_now)

    col1, col2 = st.columns(2)
    col1.metric("Prediction", full_out)
    col2.metric("Inference time (s)", time_passed_full)
