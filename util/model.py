import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer

from config import config

class BERTChunking(nn.Module):
    def __init__(self):
        super(BERTChunking, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(config.bert_name, cache_dir=config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path)

    def forward(self, x):
        with torch.no_grad():
            x = self.tokenizer.tokenize(x)
            x = torch.tensor([x])

            _, x = self.bert(x, output_all_encoded_layers=False)

        return x