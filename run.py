import torch

from config import config

from torch.utils.data import DataLoader
from util.dataTool import NERDataset

from util.model import BertModelEncoder, CRFTagger, NERModel
from util.trainer import train, test


if __name__ == '__main__':
    print("Data Loading...")
    train_set = NERDataset('data/eng.train', config)
    test_set = NERDataset('data/eng.testa', config)
    val_set = NERDataset('data/eng.testb', config)

    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              drop_last=True, shuffle=True, pin_memory=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=config.batch_size,
                             drop_last=False, shuffle=True, pin_memory=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            drop_last=False, shuffle=True, pin_memory=True, num_workers=1)

    print("Model Loading...")
    encoder = BertModelEncoder(config)
    tagger = CRFTagger(config)

    model = NERModel(encoder, tagger).to(config.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training")
    # model = train(model, train_loader, val_loader, optimizer, config)

    test(model, test_loader, config)