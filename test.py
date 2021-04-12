import argparse
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


if __name__ == '__main__':
    # x = torch.rand(10,4,2048)
    # print(x[1][0].shape)
    
    # batch_size, seq_len, n_hidden
    seq = torch.tensor([[5,2,1], [3,1,1], [4,-5,6], [9,4,1]])
    lens = [2, 1, 3, 2]
    # packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
    # print(packed)
    sent_output = [x[:l] for x, l in zip(seq, lens)]
    print(sent_output)
    emb = [torch.max(x, 0)[0] for x in sent_output]
    print(emb)        
    emb = torch.stack(emb, 0)
    print(emb)