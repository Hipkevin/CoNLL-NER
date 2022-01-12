# CoNLL-English NER Task
en | [ch](README_ch.md)

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

### Usage
    chmod 755 deploy
    ./deploy

    ./gpu n  # monitor the GPU (refresh every n seconds)
    ./run  # start

## Baseline Performance (1 ep | macro)
| Model | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: |
| Bert-CRF | 0.71 | 0.68 | 0.69 |
| Bert-softmax | - | - | - |
| Bert-BiLSTM-CRF | - | - | - |
| Bert-BiLSTM-softmax | - | - | - |

## Optimization
- cost sensitive learning or drop the few classes
- dropout to improve the generalization performance
- different backbone structures
- DDP training --> large GPU caches for a large batch_size
- more epochs --> schedule the learning rate dynamically while training