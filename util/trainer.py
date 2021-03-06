from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


def train(model, train_loader, val_loader, optimizer, config):
    model.train()
    for epoch in range(config.epoch_size):

        # 每个epoch遍历所有数据
        for idx, data in enumerate(train_loader):

            # 加载数据集至指定设备
            x, y = data[0].to(config.device), data[1].to(config.device)

            out, _ = model(x)

            # 使用CRF时，计算CRF层的loss
            # tagger_loss与predict_loss相加后得到loss，
            # 求偏导之后分别对tagger和encoder进行反向传播优化
            if "CRF" in str(type(model.tagger)):
                tagger_loss, predict_loss = model.tagger.loss_func(out, y)
                loss = tagger_loss + predict_loss
            else:
                loss = model.tagger.loss_func(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            if idx % 10 == 0:
                val_f1 = 0
                for i, data in enumerate(val_loader):
                    val_text, val_y = data[0].to(config.device), data[1]
                    _, tag = model(val_text)
                    val_f1 += evaluate(val_y, tag)

                val_f1 /= len(val_loader)
                if "CRF" in str(type(model.tagger)):
                    print(f'epoch: {epoch + 1} batch: {idx} '
                          f'| tagger_loss: {tagger_loss} predict_loss: {predict_loss} | val_macro_f1: {val_f1}')
                else:
                    print(f'epoch: {epoch + 1} batch: {idx} | loss: {loss} | val_macro_f1: {val_f1}')

    return model


def evaluate(Y, Y_hat):
    f1 = 0
    num = len(Y)
    for y, y_hat in zip(Y, Y_hat):
        f1 += f1_score(y, y_hat, average='macro')

    return f1 / num


def test(model, test_loader, config):
    model.eval()
    Y = []
    Tag = []
    for i, data in tqdm(enumerate(test_loader)):
        text, y = data[0].to(config.device), data[1]
        _, tag = model(text)

        Y += y
        Tag += tag

    y = []
    y_hat = []
    for i, j in zip(Y, Tag):
        y += list(i.numpy())
        y_hat += j

    print(classification_report(y_true=y, y_pred=y_hat))

    macro_f1 = f1_score(y_true=y, y_pred=y_hat, average="macro")
    micro_f1 = f1_score(y_true=y, y_pred=y_hat, average="micro")
    print("macro-F1: ", macro_f1)
    print("micro-F1: ", micro_f1)