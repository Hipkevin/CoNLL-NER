# CoNLL 英文语料命名实体识别
[en](README.md) | ch

## 数据集格式
    数据集包含已经划分好的训练集、测试集和验证集

    格式如下，使用Entity标签作为Y
    -DOCSTART- -X- O O
    -sentnce- -pos- -Chuck- -Entity-

## 项目结构
    -data  # 原始数据
    -emb # BERT模型存放路径（如果没有指定的模型文件，transformers会自行下载，速度很快）

    -util
        -dataTool.py  # 数据接口
        -model.py  # 模型定义
        -trainer.py  # 训练和测试接口

    config.py  # 实验参数配置
    run.py
    requirement.txt  # 项目依赖

    EDA.ipynb # 探索性数据分析，确定一些超参数。如通过观察文本长度分布确定padding_size

### 模型设计
    将NER模型解耦，分解为encoder和tagger

    encoder负责文本特征变换，可以使用bert或者lstm
    tagger负责序列标注，可以使用softmax或者crf

### 使用方法
    # 部署相关依赖
    chmod 755 deploy
    ./deploy

    ./gpu n  # 监控GPU使用情况，n为每秒刷新频率
             # 如果GPU利用率低，IO角度可以考虑pin_memory或者增加num_workers，
             # 训练角度可以考虑增加batch_size
    ./run  # 开始训练

## Baseline性能 (1 ep | macro)
| Model | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: |
| Bert-CRF | 0.71 | 0.68 | 0.69 |
| Bert-softmax | - | - | - |
| Bert-BiLSTM-CRF | - | - | - |
| Bert-BiLSTM-softmax | - | - | - |

## 优化建议
从Bert-CRF实验结果来看，存在support极少的标签影响整体分值，可以从以下角度着手优化
- 使用代价敏感学习，提高模型对少数类的敏感性
- 删除或增加含有少数类标签的样本
- baseline训练时发现加入dropout后模型难收敛，考虑增加训练轮次
- 使用不同的结构
- 增大batch_size，使用大显存或者DDP分布式训练
- 增加epoch --> 动态调整学习率（如余弦退火或者加入带热重启的衰减）