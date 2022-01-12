import abc

import torch
import torch.nn as nn

from transformers import BertModel
from torchcrf import CRF

from typing import List, Tuple

"""
Interface ==> Encoder

Abstract Class ==> Tagger
loss_func must be implemented

NERModel combines the two classes
out --> features extracted by encoder
tag --> decoded by tagger (CRF -- > viterbi | Softmax --> argmax) 
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

class Tagger(abc.ABC, nn.Module):
    def __init__(self):
        super(Tagger, self).__init__()

    @abc.abstractmethod
    def loss_func(self, out, y):
        pass

class NERModel(nn.Module):
    def __init__(self, encoder: Encoder, tagger: Tagger):
        super(NERModel, self).__init__()

        self.encoder = encoder
        self.tagger = tagger

    def forward(self, x) -> Tuple[torch.Tensor, List[List[int]]]:
        out = self.encoder(x)

        return out, self.tagger(out)


"""
Sub-Class implement the abstract interface
"""
class BertModelEncoder(Encoder):
    def __init__(self, config):
        super(BertModelEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_name, cache_dir=config.bert_path)
        self.fc = nn.Linear(config.tagger_input, config.label_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 取bert最后一个隐藏层的输出
        x = self.bert(x)[0]
        x = self.fc(x)
        x = self.dropout(x)

        return x

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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.dropout(self.fc(out))

class BertBiLSTMEncoder(Encoder):
    def __init__(self, config):
        super(BertBiLSTMEncoder, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(config.bert_name, cache_dir=config.bert_path)
        self.lstm_encoder = BiLSTMEncoder(config)

    def forward(self, x):
        return self.lstm_encoder(self.bert_encoder(x)[0])

class SoftmaxTagger(Tagger):
    def __init__(self, config):
        super(SoftmaxTagger, self).__init__()

        self.tag_num = config.label_size
        self.device = config.device

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x) -> List[List[int]]:
        tag = torch.argmax(self.softmax(x), dim=-1)
        return tag.tolist()

    def loss_func(self, out, y):
        weight = torch.tensor([1] * self.tag_num, dtype=torch.float32).to(self.device)
        # 代价敏感，可以调整少数类标签带来的损失
        # weight[:] = 0.9
        # weight[-1] = 0.08

        criterion = nn.CrossEntropyLoss(weight=weight)

        predict_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        for Y_hat, Y in zip(out, y):
            predict_loss += criterion(Y_hat, Y)

        return predict_loss

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

        # cost sensitive learning
        # change the weights of few samples' cost

        # weight[:] = 0.9
        # weight[-1] = 0.08

        criterion = nn.CrossEntropyLoss(weight=weight)

        crf_loss = - self.crf(out, y, reduction='token_mean')  # crf-loss

        predict_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        for Y_hat, Y in zip(out, y):
            predict_loss += criterion(Y_hat, Y)

        return crf_loss, predict_loss