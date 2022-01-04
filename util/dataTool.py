import torch

from torch.utils.data import Dataset
from tqdm import tqdm

from config import config


def getData(path):
    with open(path, 'r', encoding='utf8') as file:
        data_text = file.readlines()

    X, Y, x, y = list(), list(), list(), list()
    for sample in data_text:
        if sample != '\n':
            item = sample.split(' ')
            x.append(item[0])
            y.append(item[-1].replace('\n', ''))
        else:
            X.append(x)
            Y.append(y)
            x, y = list(), list()

    return X, Y


class NERDataset(Dataset):
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

    def __init__(self, path, config):
        text, labels = getData(path)

        X, Y = list(), list()
        for i in tqdm(range(len(text))):
            sentence, label = text[i], labels[i]

            sentence_length = len(sentence)
            tag = [config.label2idx[l] for l in label]

            if sentence_length < config.pad_size:
                sentence.extend(['[PAD]'] * (config.pad_size - sentence_length))
                tag.extend([config.label2idx['O']] * (config.pad_size - sentence_length))
            else:
                sentence = sentence[0: config.pad_size]
                tag = tag[0: config.pad_size]

            token = config.tokenizer.encode(sentence, add_special_tokens=False)

            X.append(token)
            Y.append(tag)

        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)