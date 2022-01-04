# CoNLL-English NER Task

## Motivation
    Course Project
    review the pytorch framework and sequence-labeling task
    practice using the transformers of Huggingface

## Dataset Introduction
    A train set, a test set and a validation set in the data file

    -DOCSTART- -X- O O
    -sentnce- -pos- -Chuck- -Entity-

## Project Structure
    -data  # source data
    -emb # BERT model files

    -util
        -dataTool.py  # data interface
        -model.py
        -trainer.py  # train and evaluate

    config.py  # parameters in the project
    run.py
    requirement.txt

    EDA.ipynb # exploratory data analasis, 
              # which aims to confirm the hyper-params in the trials

### Coding Pattern
    For keeping the convenience and simplicity of experiments,
    decouple the model into two units: encoder and tagger
    
    model ==> encoder + tagger
    
    In such a way, encoder extracts the context and linguistit features,
    which will be received by tagger to output BIO tags.

## Baseline Performance
| Model | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: |
| Bert-softmax | - | - | - | - |
| Bert-CRF | - | - | - | - |
| Bert-BiLSTM-softmax | - | - | - | - |
| Bert-BiLSTM-CRF | - | - | - | - |

## Conclusion