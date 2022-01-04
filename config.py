
class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.bert_name = 'bert-base-cased'
        self.bert_path = 'emb'

        self.pad_size = 128

        self.label2idx = dict()

        with open('data/label.txt', 'r', encoding='utf8') as file:
            tags = file.read().split('\n')
        for idx, t in enumerate(tags):
            if t:
                self.label2idx[t] = idx


config = Config()
