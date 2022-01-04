import torch
import torch.nn as nn

from transformers import BertModel
from torchcrf import CRF

"""
Abstract Class (without abc)
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

class Tagger(nn.Module):
    def __init__(self):
        super(Tagger, self).__init__()

class NERModel(nn.Module):
    def __init__(self, encoder: Encoder, tagger: Tagger):
        super(NERModel, self).__init__()

        self.encoder = encoder
        self.tagger = tagger

    def forward(self, x):
        out = self.encoder(x)

        return out, self.tagger(out)


"""
Sub-Class
"""
class BertModelEncoder(Encoder):
    def __init__(self, config):
        super(BertModelEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_name, cache_dir=config.bert_path)
        self.fc = nn.Linear(config.tagger_input, config.label_size)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.fc(x)

        return x

# TODO: adjust the lstm and softmax structure (refactor the trainer)
class BiLSTMEncoder(Encoder):
    def __init__(self, config):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=config.tagger_input,
                            hidden_size=config.tagger_input // 2,
                            dropout=config.dropout,
                            num_layers=config.num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(config.tagger_input, config.label_size)

    def forward(self, x):
        return self.fc(self.lstm(x))

class SoftmaxTagger(Tagger):
    def __init__(self):
        super(SoftmaxTagger, self).__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return torch.argmax(self.softmax(x), dim=0)

class CRFTagger(Tagger):
    def __init__(self, config):
        super(CRFTagger, self).__init__()
        self.device = config.device
        self.tag_num = config.label_size

        self.crf = CRF(config.label_size, batch_first=True)

    def forward(self, x):
        return self.crf.decode(x)

    def loss_func(self, out, y):
        weight = torch.tensor([1] * self.tag_num, dtype=torch.float32).to(self.device)
        # weight[0] = 0.08
        # weight[1:] = 0.9

        criterion = nn.CrossEntropyLoss(weight=weight)

        crf_loss = - self.crf(out, y, reduction='token_mean')  # crf-loss

        predict_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        for Y_hat, Y in zip(out, y):
            predict_loss += criterion(Y_hat, Y)

        return crf_loss, predict_loss