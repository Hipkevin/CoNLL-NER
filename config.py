import torch
from transformers import BertTokenizer


class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_name = 'bert-base-cased'
        self.bert_path = 'emb'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name, cache_dir=self.bert_path)

        self.pad_size = 128
        self.dropout = 0.5
        self.num_layers = 2

        self.batch_size = 32
        self.epoch_size = 1
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4

        self.label2idx = dict()

        with open('data/label.txt', 'r', encoding='utf8') as file:
            tags = file.read().split('\n')
        for idx, t in enumerate(tags):
            if t:
                self.label2idx[t] = idx

        self.label_size = len(self.label2idx)
        self.tagger_input = 768


config = Config()
