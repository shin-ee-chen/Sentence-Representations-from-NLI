from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
import io
import os
import time
import argparse

import utils
from models import NLITrainer

import torch
from  torchtext.legacy import data
from torchtext.vocab import GloVe



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
    params.TEXT.build_vocab(samples, vectors=GloVe(name= params.glove_name, dim= 300))
    params.trainer = utils.load_latest(NLITrainer, CHECKPOINT_PATH, params.encoder_type, 
                                       inference=True, 
                                       map_location=params.device, 
                                       silent = False)
    
    params.trainer.embedding = utils.load_pretrained_embed(params.TEXT,params.embedding_dim)
    
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
    
    with torch.no_grad(): 
        emb = params.trainer.encode(text_tup).to(params.device)
    return emb.detach().numpy()


# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
# params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 
#                     'kfold': 10, 'encoder_type':"LSTM_Encoder", "device":"cpu", "embedding_dim":300}
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False 
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', default="BLSTM_Encoder_Max", type=str, 
                        choices=["AWE", "LSTM_Encoder", "BLSTM_Encoder", "BLSTM_Encoder_Max"],
                        help='Type of encoder, choose from [AWE, LSTM_Encoder, BLSTM_Encoder]')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")
    parser.add_argument('--glove_name', type=str, default= "6B",
                        help="glove name: 6B/840B")
    # parser.add_argument('--usepytorch', default= default=(False if not torch.cuda.is_available() else True),
    #                     help="glove name: 6B/840B")

    args = parser.parse_args()
    args.usepytorch = False if not torch.cuda.is_available() else True

    # params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 
    #                 'kfold': 10, 'encoder_type':args.encoder_type, "device":args.device, "embedding_dim":300}
    
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': args.usepytorch, 'kfold': 10,
                       'encoder_type':args.encoder_type, "device":args.device, 'embedding_dim':300, 
                       'glove_name': args.glove_name}

    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                     'tenacity': 5, 'epoch_size': 4}
    

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    
    # transfer_tasks = ['MR', 'CR', 'SUBJ','MPQA',  'SST2', 'TREC',
    #                   'MRPC', 'SICKRelatedness','SICKEntailment', STS14']
    transfer_tasks = ['MR']
    # senteval prints the results and returns a dictionary with the scores
    
    
    
    time_b = time.time()
    results = se.eval(transfer_tasks)

    time_e = time.time()
    # print(results)
    if not os.path.exists('SentEval_results'):
        os.makedirs('SentEval_results')
    output_path = os.path.join('SentEval_results', f'{args.encoder_type}_results.txt')
    file = open(output_path, 'w') 

    for k, v in results.items():
        file.write(str(k)+' '+str(v["acc"])+'\n')
        
    file.close()
    print(f"File save to {output_path}")
    print(f"Time elapsed:{(time_e-time_b)/60.0} mins")

    