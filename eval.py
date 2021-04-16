from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
import io

import utils
from models import NLITrainer

from  torchtext.legacy import data
from torchtext.vocab import GloVe

import torch

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = '../SentEval'
# path to the NLP datasets 
PATH_TO_DATA = '../SentEval/data/'
# path to glove embeddings
PATH_TO_VEC = '../SentEval/pretrained/glove.6B.300d.txt'

CHECKPOINT_PATH = "./checkpoints_lisa"

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
    

def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    params.TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, 
                tokenize="spacy",
                tokenizer_language = 'en_core_web_sm')
    params.TEXT.build_vocab(samples, vectors=GloVe(name= "6B", dim= 300))

    params.embedding_dim = 300
    return


def batcher(params, batch):
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    text_tup = params.TEXT.process(batch)
    trainer = utils.load_latest(NLITrainer, CHECKPOINT_PATH, params.encoder_type, 
                                inference=True, 
                                map_location=params.device, silent = False)
    
    trainer.embedding = utils.load_pretrained_embed(params.TEXT,params.embedding_dim)
    
    with torch.no_grad(): 
        emb = trainer.encode(text_tup)
    return emb.cpu().numpy()


# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 
                    'kfold': 10, 'encoder_type':"AWE", "device":"cpu"}
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False 
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    # transfer_tasks = ['MR' ,]
                    #   'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                    #   'MRPC', 'SICKEntailment']
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)
