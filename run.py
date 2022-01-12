import torch
import random
import numpy as np

from config import config

from torch.utils.data import DataLoader
from util.dataTool import NERDataset

from util.model import BertModelEncoder, BiLSTMEncoder, BertBiLSTMEncoder, CRFTagger, SoftmaxTagger, NERModel
from util.trainer import train, test

from multiprocessing import cpu_count

# seed everything
# 设置随机数，保证相同环境下实验结果可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # 通过数据接口加载数据集
    print("Data Loading...")
    train_set = NERDataset('data/eng.train', config)
    test_set = NERDataset('data/eng.testa', config)
    val_set = NERDataset('data/eng.testb', config)

    # 计算CPU核心数量，设置为num_workers
    cpu_num = cpu_count()

    # 训练时使用DataLoader，方便获取mini-batch
    # batch_size：每次梯度更新使用的样本数量
    # drop_last：按照指定batch分割数据时，存在余数。训练时drop，测试时保留
    # shuffle：打乱数据集
    # pin_memory：将数据从CPU加载到GPU时，为该loader分配固定的显存，提高IO效率
    # num_workers：将数据加载到GPU的线程数量，合适的线程数量可以提高IO效率，从而提高GPU利用率
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              drop_last=True, shuffle=True, pin_memory=True, num_workers=cpu_num)
    test_loader = DataLoader(test_set, batch_size=config.batch_size,
                             drop_last=False, shuffle=True, pin_memory=True, num_workers=cpu_num)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            drop_last=False, shuffle=True, pin_memory=True, num_workers=cpu_num)

    # encoder和tagger的初始化
    print("Model Loading...")
    encoder = BertModelEncoder(config)
    # encoder = BiLSTMEncoder(config)
    # encoder = BertBiLSTMEncoder(config)

    tagger = CRFTagger(config)
    # tagger = SoftmaxTagger(config)

    # 组装NER模型
    model = NERModel(encoder, tagger).to(config.device)

    # 设置优化器，常见的有SGD、Adam、RMSprop等
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)

    # 通过训练接口训练模型
    print("Training")
    model = train(model, train_loader, val_loader, optimizer, config)

    # 测试
    test(model, test_loader, config)