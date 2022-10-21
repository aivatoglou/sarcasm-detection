import torch
from transformers import AutoModel


class DistilBERT(torch.nn.Module):
    def __init__(self):

        super(DistilBERT, self).__init__()
        self.distilbert = AutoModel.from_pretrained(
            "distilbert-base-uncased",
        )
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, text, mask):

        distil_out = self.distilbert(input_ids=text, attention_mask=mask)
        hidden_state = distil_out[0]
        pooled_output = hidden_state[:, 0]
        out = self.dropout(pooled_output)
        out = self.fc(out)
        return out
