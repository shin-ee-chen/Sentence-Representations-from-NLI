# Sentence Representations from NLI

## Report Conclusion



# Structure

```
Sentence_Representations_from_NLI
├── eval.py
│   └── TODO
├── models.py
│   └── TODO
├── test.py
│   └── TODO
├── train.py
│   └── file for training a NLI model.
└── utils.py
    └── utils for loading data.
```

# Students



# Requirements

## Environment

We provide a conda environment which contains all packages you might need for running the repo. For your own computer, the environment.yml suggests the local packages required. As we do not have local computer with GPU to train all the models, rather we use Lisa environment with GPU provided by the deep learning course at UvA with environment_lisa.yml which installs the environment atcs2021 with CUDA 10.1 support.

Running on gpu (note all the models are trained on gpu, so you will get error if you try to load the pretrained models on cpu):
- add the following lines in your ".bashrc":
```shell
module load 2019
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12
```
- install the required packages:
```shell
pip install pytorch==1.8.1
pip install torchtext==0.9.1
pip install pytorch-ligntning==1.2.7
pip install sklearn
```
- add the following line at the beginning of your experiment script (.sh), before running your Python script:
```shell
source activate atcs2021
```
Follow the detailed description [here](https://github.com/uvadlc/uvadlc_practicals_2020/blob/master/assignment_1/1_mlp_cnn/README.md).

# How to start

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
python -u train.py --debug False --glove_name 840B --epochs 30 --batch_size 64 --encoder_type BLSTM_Encoder --lstm_num_hidden 2048 --max_pooling False
```

### Running BLSTM with max-pooling

```python
python -u train.py --debug False --glove_name 840B --epochs 30 --batch_size 64 --encoder_type BLSTM_Encoder --lstm_num_hidden 2048 --max_pooling True
```



# Results