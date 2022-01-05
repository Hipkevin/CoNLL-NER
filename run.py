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
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    print("Data Loading...")
    train_set = NERDataset('data/eng.train', config)
    test_set = NERDataset('data/eng.testa', config)
    val_set = NERDataset('data/eng.testb', config)

    cpu_num = cpu_count()
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              drop_last=True, shuffle=True, pin_memory=True, num_workers=cpu_num)
    test_loader = DataLoader(test_set, batch_size=config.batch_size,
                             drop_last=False, shuffle=True, pin_memory=True, num_workers=cpu_num)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            drop_last=False, shuffle=True, pin_memory=True, num_workers=cpu_num)

    print("Model Loading...")
    encoder = BertModelEncoder(config)
    # encoder = BiLSTMEncoder(config)
    # encoder = BertBiLSTMEncoder(config)

    tagger = CRFTagger(config)
    # tagger = SoftmaxTagger(config)

    model = NERModel(encoder, tagger).to(config.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training")
    model = train(model, train_loader, val_loader, optimizer, config)

    test(model, test_loader, config)