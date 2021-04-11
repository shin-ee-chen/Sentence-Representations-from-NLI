import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()
         # 这个input dim是encoder的output size(average的话是batch_size, max_len, emb_dim)
        self.encoder_type = config.encoder_type

        if self.encoder_type == "AWE":
            self.embed = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=1)
            input_size = 4 * config.embedding_dim
        # elif self.encoder_type == :

        else:
            self.encoder = eval(self.encoder_type)(config)
            input_size = config.lstm_num_hidden * 4

        self.linears = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Linear(512, 3)
        )
       
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, text_tup):
        # text_tup = prems, hypos
        # u = self.encoder(prems)
        # v = self.encoder(hypos)
        if self.encoder_type == "AWE":
            u = self._awe_encoding(text_tup[0])
            v = self._awe_encoding(text_tup[1])

        else:
            u = self.encoder(text_tup[0])
            v = self.encoder(text_tup[1])
            
        features = torch.cat((u, v, torch.abs(u-v), u * v), dim = 1)

        out = self.softmax(self.linears(features))

        return out


    def _awe_encoding(self, input):
        x = self.embed(input)
        return torch.mean(x, dim = 1)


class LSTM_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm_num_hidden = config.lstm_num_hidden
        self.device = config.device

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=1)
        self.lstm = nn.LSTM(config.embedding_dim, config.lstm_num_hidden, 1,
                            # dropout= 1 - config.dropout_keep_prob, 
                            batch_first = True
                            )
        self.prev_state = None
        


    def forward(self, input):
        if self.prev_state == None:
            self.prev_state = (torch.zeros(1, input.shape[0], 
                                           self.lstm_num_hidden).to(self.device),
                               torch.zeros(1, input.shape[0],
                                           self.lstm_num_hidden).to(self.device))
        _, (h_n, _) = self.lstm(self.embedding(input), self.prev_state)
        return h_n.squeeze(0)