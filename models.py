import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import utils

class NLITrainer(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = NLINet(config)
        # # Create loss module
        self.config = config
        self.loss_module = nn.CrossEntropyLoss()
        
        # # Example input for visualizing the graph in Tensorboard
        # self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

       
    def forward(self, inout):
        return self.model(input)


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        text, labels = [batch.premise, batch.hypothesis], batch.label
        preds = self.model(text)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log('train_acc', acc, on_step=False, on_epoch=True) # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        return loss # Return tensor to call ".backward" on


    def validation_step(self, batch, batch_idx):
        text, labels = [batch.premise, batch.hypothesis], batch.label
        preds = self.model(text).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # self.config.lr /= 2 

        self.log('val_acc', acc) # By default logs it per epoch (weighted average over batches)


    def test_step(self, batch, batch_idx):
        text, labels = [batch.premise, batch.hypothesis], batch.label
        preds = self.model(text).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards



class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()
         # 这个input dim是encoder的output size(average的话是batch_size, max_len, emb_dim)
        self.encoder_type = config.encoder_type

        if self.encoder_type == "AWE":
            # self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=1)
            # self.embedding.weight.data.copy_(config.pretrained_embed)
            with torch.no_grad():
                self.embedding = utils.load_pretrained_embed(config.TEXT,config.embedding_dim)
            encode_size = 4 * config.embedding_dim

        elif self.encoder_type == "LSTM_Encoder":
            self.encoder = eval(self.encoder_type)(config)
            encode_size = 4 * config.lstm_num_hidden

        elif self.encoder_type == "BLSTM_Encoder":
            self.encoder = eval(self.encoder_type)(config)
            encode_size = config.lstm_num_hidden * 4 * 2
        
        else:
            print("Error Encoder Type")
            exit(-1)

        
        # self.classifer = nn.Sequential(
        #     nn.Linear(self.encode_size, 512),
        #     nn.Linear(512, 512),
        #     nn.Linear(512, 3)
        # )
        self.classifer =  nn.Sequential(
            nn.Dropout(p=config.dpout_fc),
            nn.Linear(encode_size, config.fc_dim),
            nn.Tanh(),
            nn.Dropout(p=config.dpout_fc),
            nn.Linear(config.fc_dim, config.fc_dim),
            nn.Tanh(),
            nn.Dropout(p=config.dpout_fc),
            nn.Linear(config.fc_dim, 3),
        )
       
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, text_tup):
        if self.encoder_type == "AWE":
            u = self._awe_encoding(text_tup[0])
            v = self._awe_encoding(text_tup[1])

        else:
            u = self.encoder(text_tup[0])
            v = self.encoder(text_tup[1])
            
        features = torch.cat((u, v, torch.abs(u-v), u * v), dim = 1)

        out = self.classifer(features)
        out = self.softmax(out)

        return out

    def encode(self, text_tup):
        """
        Inputs:
        text_tup - [text_data, text_lengths]
        """
        if self.encoder_type == "AWE":
             emb = _awe_encoding(text_tup[0])
        else:
            emb = self.encoder(text_tup)
        
        return emb

    def _awe_encoding(self, input):
        x = self.embedding(input[0])
        return torch.mean(x, dim = 1)

    

class LSTM_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm_num_hidden = config.lstm_num_hidden
        self.device = config.device

        with torch.no_grad():
            self.embedding = utils.load_pretrained_embed(config.TEXT,config.embedding_dim)

        # TEXT = config.TEXT
        # PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        # UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        # self.embedding = nn.Embedding(len(TEXT.vocab), config.embedding_dim, padding_idx=PAD_IDX)
        # self.embedding.weight.data.copy_(TEXT.vocab.vectors)

        # self.embedding.weight.data[UNK_IDX] = torch.zeros(300)
        # self.embedding.weight.data[PAD_IDX] = torch.zeros(300)

        self.lstm = nn.LSTM(config.embedding_dim, config.lstm_num_hidden, 1,
                            dropout=config.dpout_lstm
                            )
                            
        self.prev_state = None
        

    def forward(self, input):
        if self.prev_state == None:
            self.prev_state = (torch.zeros(1, input[0].shape[0], 
                                           self.lstm_num_hidden),
                               torch.zeros(1, input[0].shape[0],
                                           self.lstm_num_hidden))
        emb = self.embedding(input[0])
        
        
        packed_input = nn.utils.rnn.pack_padded_sequence(emb, input[1].cpu(), batch_first=True, 
                                                         enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_input, self.prev_state)

        # output size is (batch_size, lstm_num_hidden)
        return h_n.squeeze(0)


class BLSTM_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_pooling = config.max_pooling
        self.lstm_num_hidden = config.lstm_num_hidden
        self.device = config.device

        # self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=1)
        # self.embedding.weight.data.copy_(config.pretrained_embed)
        with torch.no_grad():
            self.embedding = utils.load_pretrained_embed(config.TEXT,config.embedding_dim)

        self.lstm = nn.LSTM(config.embedding_dim, config.lstm_num_hidden, 1,
                            dropout= config.dpout_lstm, 
                            bidirectional = True
                            )
        self.prev_state = None
        


    def forward(self, input):
        if self.prev_state == None:
            self.prev_state = (torch.zeros(2, input[0].shape[0], 
                                           self.lstm_num_hidden),
                               torch.zeros(2, input[0].shape[0],
                                           self.lstm_num_hidden))
        emb = self.embedding(input[0])
        packed_input = nn.utils.rnn.pack_padded_sequence(emb, input[1].cpu(), batch_first=True, 
                                                         enforce_sorted=False)
        output, (h_n, _) = self.lstm(packed_input, self.prev_state)
        
        
        if self.max_pooling == "True":
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
            # output size is (batch_size, seq_len, lstm_num_hidden * 2)
            sent_output = [x[:l] for x, l in zip(output, input[1])]
            emb = [torch.max(x, 0)[0] for x in sent_output]       
            out = torch.stack(emb, 0)

        else:
            # output size is (batch_size, lstm_num_hidden * 2)
            out = torch.cat((h_n[0], h_n[1]), 1)
            
        return out