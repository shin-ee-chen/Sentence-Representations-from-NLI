# import datasets
from torchtext.legacy import data
from torchtext.legacy.datasets import SNLI
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader

import torch
import spacy
from spacy.tokenizer import Tokenizer




def load_data(batch_size, max_len = 10, device = "cpu"):
      # set up fields
      TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, 
                          fix_length= max_len)
      LABEL = data.Field(sequential=False)

      # make splits for data
      train, val, test = SNLI.splits(TEXT, LABEL)

      # build the vocabulary
      TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
      # TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=300))

      vocab_size = len(TEXT.vocab)
      LABEL.build_vocab(train, specials_first=False)

      train_iter, val_iter, test_iter = data.Iterator.splits(
                                        (train, val, test), 
                                        batch_size=batch_size, 
                                        device = device)
      return train_iter, val_iter, test_iter, vocab_size



def spacy_tokenize(text):
      nlp = spacy.load("en_core_web_sm")
      tokenizer = Tokenizer(nlp.vocab)
      return [tok.text for tok in tokenizer(text)]


if __name__ == '__main__':
      train_iter, _, _, vocab_size = load_data(4)
      i = 0
      train_iter = train_iter[0:10]
      
      embed = torch.nn.Embedding(vocab_size, 300, padding_idx=1)
      for item in train_iter:
            # word vector
            print(embed(item.premise[0]).shape)

            print(item.label)
            i += 1
            print()
            #   print(text.shape)
            if i > 1:
                break