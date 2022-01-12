import torch

from torch.utils.data import Dataset
from tqdm import tqdm

# 数据接口层一般包含数据读取函数和torch训练使用的Dataset类
# 数据读取函数负责从不同数据源，将数据初步转换为需要的格式
# Dataset类负责类型转换，以及进一步的预处理


def getData(path):
    """
    get source data
    :param path: data file path
    :return: text content and label (List[str])
    """
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
    # Dataset类必须实现的接口
    # DataLoader的sampler会调用这个方法进行mini-batch的获取
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    # 实现该接口后可以用len()函数获取数据集大小
    def __len__(self):
        return len(self.X)

    def __init__(self, path, config):
        text, labels = getData(path)

        X, Y = list(), list()
        for i in tqdm(range(len(text))):
            sentence, label = text[i], labels[i]

            sentence_length = len(sentence)
            tag = [config.label2idx[l] for l in label]

            # padding
            # 为了进行并行化的训练，需要把数据维度统一
            # 超过padding_size的样本则截断
            # 小于则填补特殊的token，该token不具有实际含义
            if sentence_length < config.pad_size:
                sentence.extend(['[PAD]'] * (config.pad_size - sentence_length))
                tag.extend([config.label2idx['O']] * (config.pad_size - sentence_length))
            else:
                sentence = sentence[0: config.pad_size]
                tag = tag[0: config.pad_size]

            # 通过bert的分词器对样本进行编码，获取每个token的id
            # add_special_tokens设置为True时，会在样本的首尾加上[CLS]和[SEP]
            token = config.tokenizer.encode(sentence, add_special_tokens=False)

            X.append(token)
            Y.append(tag)

        # 转换为tensor，类型为整型
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)