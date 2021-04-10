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

        if config.encoder_type == "AWE":
            self.embed = nn.Embedding(config.vocab_size, 300, padding_idx=1)
            input_size = 4 * 300

        else:
            self.encoder = eval(config.encoder_type)(config)
            input_size = 300

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

        features = torch.cat((u, v, torch.abs(u-v), u * v), dim = 1)

        out = self.softmax(self.linears(features))

        return out


    def _awe_encoding(self, input):
        x = self.embed(input)
        return torch.mean(x, dim = 1)

