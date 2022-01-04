import torch
from sklearn.metrics import f1_score, classification_report

def train(model, train_loader, val_loader, optimizer, config):
    model.train()
    for epoch in range(config.epoch_size):

        for idx, data in enumerate(train_loader):

            x, y = data[0].to(config.device), data[1].to(config.device)

            out,_ = model(x)

            tagger_loss, predict_loss = model.tagger.loss_func(out, y)
            loss = tagger_loss + predict_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                for i, data in enumerate(val_loader):
                    val_text, val_y = data[0].to(config.device), data[1]
                    _, tag = model(val_text)
                    val_f1 = evaluate(val_y, tag)
                print(f'epoch: {epoch + 1} batch: {idx} '
                      f'| tagger_loss: {tagger_loss} predict_loss: {predict_loss} | val_f1: {val_f1}')

    return model

def evaluate(Y, Y_hat):
    f1 = 0
    num = len(Y)
    for y, y_hat in zip(Y, Y_hat):
        f1 += f1_score(y, y_hat, average='macro')

    return f1/num

def test(model, test_loader, config):
    model.eval()
    p_collection = list()
    y_collection = list()
    for idx, data in enumerate(test_loader):
        x, y = data[0].to(config.device), data[1].to('cpu')

        predict = torch.argmax(model.predict(x), dim=1).to('cpu')
        p_collection += predict.tolist()
        y_collection += y.tolist()

    print(classification_report(y_true=y_collection, y_pred=p_collection))

    macro_f1 = f1_score(y_true=y_collection, y_pred=p_collection, average="macro")
    micro_f1 = f1_score(y_true=y_collection, y_pred=p_collection, average="micro")
    print("macro-F1: ", macro_f1)
    print("micro-F1: ", micro_f1)