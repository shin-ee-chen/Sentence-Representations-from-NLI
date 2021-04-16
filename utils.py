# import datasets
from  torchtext.legacy import data
from torchtext.legacy.datasets import SNLI
from torchtext.vocab import GloVe

import torch
import torch.nn as nn

import argparse
import os
# import spacy
# from spacy.tokenizer import Tokenizer

def load_data(batch_size, embedding_dim, glove_name = "6B", device = "cpu"):
      # set up fields
      TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, 
                        tokenize="spacy",
                        tokenizer_language = 'en_core_web_sm'
                        )
      LABEL = data.Field(sequential=False)

      # make splits for data
      train, val, test = SNLI.splits(TEXT, LABEL)

      # build the vocabulary
      TEXT.build_vocab(train, vectors=GloVe(name= glove_name, dim= embedding_dim))
      # TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=300))
      LABEL.build_vocab(train, specials_first=False)

      vocab_size = len(TEXT.vocab)

      train_iter, val_iter, test_iter = data.Iterator.splits(
                                        (train, val, test), 
                                        batch_size=batch_size, 
                                        device = device)

      return train_iter, val_iter, test_iter, TEXT


def load_pretrained_embed(TEXT,embedding_dim):
      PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] #<pad>, 0
      UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token] #<unk>, 1

      with torch.no_grad():
          embedding = nn.Embedding(len(TEXT.vocab), embedding_dim, padding_idx=PAD_IDX)
          embedding.weight.data.copy_(TEXT.vocab.vectors)
          embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
          embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)
          embedding.weight.requires_grad = False
      return embedding



def load_latest(trainer, path, save_name, inference=False, map_location=None, silent = False):
    """Loads the last found model from the checkpoints directory
    Args:
        trainer: PyTorch Lightning module that has the .load_from_checkpoint attribute
        save_name: model name
        inference (bool, optional): whether or not to freeze weights. Defaults to False.
        map_location (device, optional): which device to map the loaded model to. Defaults to None.
        silent (bool, optional): suppresses printing unless no model is found
        
        Code copied from https://github.com/shin-ee-chen/UvA_FACT_2021/blob/main/utils/reproducibility.py
    """

    def version_to_number(filename):
        return int(filename.rsplit('_', 1)[-1])

    def checkpoint_to_numbers(filename):
        parts = filename.split('=')
        if len(parts) == 3: # epoch=[a]-step=[b].ckpt
            a = int(parts[1][:-5]) # strip '-step'
            b = int(parts[2][:-5]) # strip '.ckpt'
        elif len(parts) == 2: # epoch=[a].ckpt
            a = int(parts[1][:-5]) # strip '.ckpt'
            b = 0
        else:
            return filename
        return (a, b)

    def find_latest_version(save_name):
        save_loc = os.path.join(
            path, save_name, 'lightning_logs')
        folders = os.listdir(save_loc)
        if len(folders) == 0: return "None"
        folders.sort(key=version_to_number)
        latest_version = folders[-1]
        checkpoints = os.listdir(os.path.join(save_loc, latest_version, 'checkpoints'))
        if len(checkpoints) == 0: return "None"
        checkpoints.sort(key=checkpoint_to_numbers)
        cpt = checkpoints[-1]
        return os.path.join(save_loc, latest_version, 'checkpoints', cpt)

    pretrained_filename = find_latest_version(save_name)
    print("pretrained_filename: ", pretrained_filename)
    if os.path.isfile(pretrained_filename):
        if not silent:
            print("Found pretrained model at %s" % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = trainer.load_from_checkpoint(
            pretrained_filename, map_location=map_location)
    else:
        sys.exit(f"{save_name} model not found.")

    if inference:
        model.eval()
        model.freeze()

    return model


if __name__ == '__main__':
      parser = argparse.ArgumentParser()

      # # Model hyperparameters
      # parser.add_argument('--embedding_dim', default=300, type=int,
      #                     help='Dimensionality of latent space')
      # parser.add_argument("--lstm_num_hidden", type=int, default=3, 
      #                     help="encoder nhid dimension")
      # parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
      #                   help="Device to run the model on.")
      # parser.add_argument('--max_pooling', type=str, choices=["False","True"] ,default= "True",
      #                   help="Whether to use small dataset to debug")

      # parser.add_argument('--dpout_lstm', default=0., type=float,
      #                   help='dropout rate of lstm') 

      # config = parser.parse_args()

      train_iter, val_iter, _, TEXT = load_data(4, 300)
      i = 0
    #   encoder = LSTM_Encoder(config)
      for batch in val_iter:
            # word vector
            text, labels = [batch.premise, batch.hypothesis], batch.label
            
            # print(text[0])
            # print("break for text 0")
            # out = encoder(text[0])
            
            i += 1
            # print(out.shape)
            # [4 , 4096]
            if i > 1:
                break