
from util.dataTool import NERDataset
from config import config

if __name__ == '__main__':
    train = NERDataset('data/eng.train', config)

    # import torch
    # from transformers import BertModel, BertTokenizer
    #
    # # 这里我们调用bert-base模型，同时模型的词典经过小写处理
    # model_name = 'bert-base-cased'
    # # 读取模型对应的tokenizer
    # tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir='emb')
    # # 载入模型
    # model = BertModel.from_pretrained(model_name)
    # # 输入文本
    # input_text = "Here is some text to encode".split(' ')
    # # 通过tokenizer把文本变成 token_id
    # input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    # # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
    # input_ids = torch.tensor([input_ids])
    # # 获得BERT模型最后一个隐层结果
    # with torch.no_grad():
    #     last_hidden_states = model(input_ids)[0]
    #
    # print(last_hidden_states.size())