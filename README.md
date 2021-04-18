# Sentence Representations from NLI
# Structure

```
Sentence_Representations_from_NLI
├── demo.py
│   └── Demo using pretrained models to encode sentences
├── eval.py
│   └── Evaluating pretrained model performance with SentEval tools
├── models.py
│   └── models for encoders, NLINet and NLITrainer
├── train.py
│   └── file for training a NLI model.
└── utils.py
    └── utils for loading data and pretrained model.
```


# Requirements

## Dependancies
pytorch==1.8.1

torchtext==0.9.1

pytorch-ligntning==1.2.7

nltk==3.6.1

sklearn==0.24.1

install sparcy: https://spacy.io/usage

install SentEval: https://github.com/facebookresearch/SentEval

# How to start
# Training
### Running AWE

```python
python -u train.py --debug False --glove_name 840B --epochs 20 --batch_size 64 --encoder_type AWE
```

### Running LSTM

```python
python -u train.py --debug False --glove_name 840B --epochs 30 --batch_size 64 --encoder_type LSTM_Encoder --lstm_num_hidden 2048
```

### Running BLSTM without max-pooling

```python
python -u train.py --debug False --glove_name 840B --epochs 30 --batch_size 128 --encoder_type BLSTM_Encoder --lstm_num_hidden 2048 --max_pooling False
```

### Running BLSTM with max-pooling

```python
python -u train.py --debug False --glove_name 840B --epochs 30 --batch_size 128 --encoder_type BLSTM_Encoder --lstm_num_hidden 2048 --max_pooling True
```

## Evaluation
## Download pretrained models
pretrained models can be downloaded from:
https://drive.google.com/drive/folders/1iEfPM3F9hY0l_k39CJl98s48tF6Rjx6V?usp=sharing
Load pretrained models from checkpoints/

### Evaluating with SentEval tools
```python
python -u eval.py --encoder_type [Encoder_name] --glove_name 840B
```

Encoder_name choices: AWE, LSTM_Encoder, BLSTM_Encoder, BLSTM_Encoder_Max

# Results
See report.pdf and demo.ipynb