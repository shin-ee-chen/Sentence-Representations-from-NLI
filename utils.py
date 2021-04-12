# import datasets
from  torchtext.legacy import data
from torchtext.legacy.datasets import SNLI
from torchtext.vocab import GloVe

# use for testing
from models import LSTM_Encoder
from models import BLSTM_Encoder

import torch
import argparse
# import spacy
# from spacy.tokenizer import Tokenizer

def load_data(batch_size, embedding_dim, glove_name = "6B", device = "cpu"):
      # set up fields
      TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, tokenize="spacy")
      LABEL = data.Field(sequential=False)

      # make splits for data
      train, val, test = SNLI.splits(TEXT, LABEL)

      # build the vocabulary
      TEXT.build_vocab(train, vectors=GloVe(name= glove_name, dim= embedding_dim))
      # TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=300))

      vocab_size = len(TEXT.vocab)
      LABEL.build_vocab(train, specials_first=False)

      train_iter, val_iter, test_iter = data.Iterator.splits(
                                        (train, val, test), 
                                        batch_size=batch_size, 
                                        device = device)
      return train_iter, val_iter, test_iter, vocab_size



# def spacy_tokenize(text):
#       nlp = spacy.load("en_core_web_sm")
#       tokenizer = Tokenizer(nlp.vocab)
#       return [tok.text for tok in tokenizer(text)]


if __name__ == '__main__':
      parser = argparse.ArgumentParser()

      # Model hyperparameters
      parser.add_argument('--embedding_dim', default=300, type=int,
                          help='Dimensionality of latent space')
      parser.add_argument("--lstm_num_hidden", type=int, default=3, 
                          help="encoder nhid dimension")
      parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")
      parser.add_argument('--max_pooling', type=str, choices=["False","True"] ,default= "True",
                        help="Whether to use small dataset to debug")
      config = parser.parse_args()

      train_iter, _, _, config.vocab_size = load_data(4, config.embedding_dim)
      i = 0
      encoder = BLSTM_Encoder(config)
      for batch in train_iter:
            # word vector
            text, labels = [batch.premise, batch.hypothesis], batch.label
            
            # print(text[0])
            # print("break for text 0")
            out = encoder(text[0])
            
            i += 1
            print(out.shape)
            # [4 , 4096]
            if i > 1:
                break