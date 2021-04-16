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
        
        self.config = config
        self.embedding = config.embedding
        # Create model        
        self.model = NLINet(config)
        # # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # # Example input for visualizing the graph in Tensorboard
        # self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

       
    def forward(self, texts):
        texts = [self.process_batch(texts[0]),self.process_batch(texts[1])]
        return self.model(texts)


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        
        return [optimizer], [scheduler]


    def process_batch(self, text_tup):
        "text_tup:[text_data, length]"
        text_emb = self.embedding(text_tup[0])

        return [text_emb, text_tup[1]]
    
    def encode(self, text_tup):
        "text_tup:[text_data_emb, length]"
        text_tup = self.process_batch(text_tup)
        return self.model.encode(text_tup)
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the train data loader.
        text = [self.process_batch(batch.premise), self.process_batch(batch.hypothesis)]
        labels = batch.label
        preds = self.model(text)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log('train_acc', acc, on_step=False, on_epoch=True) # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        return loss # Return tensor to call ".backward" on


    def validation_step(self, batch, batch_idx):
        text = [self.process_batch(batch.premise), self.process_batch(batch.hypothesis)]
        labels = batch.label
        preds = self.model(text).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # self.config.lr /= 2 

        self.log('val_acc', acc) # By default logs it per epoch (weighted average over batches)


    def test_step(self, batch, batch_idx):
        text = [self.process_batch(batch.premise), self.process_batch(batch.hypothesis)]
        labels = batch.label
        preds = self.model(text).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards



class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()
         # 这个input dim是encoder的output size(average的话是batch_size, max_len, emb_dim)
        self.encoder_type = config.encoder_type

        if self.encoder_type == "AWE":
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


    def forward(self, texts):
        u = self.encode(texts[0])
        v = self.encode(texts[1])

        # else:
        #     u = self.encoder(text_tup[0])
        #     v = self.encoder(text_tup[1])
            
        features = torch.cat((u, v, torch.abs(u-v), u * v), dim = 1)

        out = self.classifer(features)
        out = self.softmax(out)

        return out

    def encode(self, text_tup):
        """
        Inputs:
        text_tup - [text_data_embedding, text_lengths]
        """
        if self.encoder_type == "AWE":
             emb = torch.mean(text_tup[0], dim = 1)
        else:
            emb = self.encoder(text_tup)
        
        return emb

    

class LSTM_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm_num_hidden = config.lstm_num_hidden
        self.device = config.device

        self.lstm = nn.LSTM(config.embedding_dim, config.lstm_num_hidden, 1,
                            dropout=config.dpout_lstm,
                            bidirectional = False
                            )
                            
        self.prev_state = None
        

    def forward(self, input):
        if self.prev_state == None:
            self.prev_state = (torch.zeros(1, input[0].shape[0], 
                                           self.lstm_num_hidden).to(input[0].device),
                               torch.zeros(1, input[0].shape[0],
                                           self.lstm_num_hidden).to(input[0].device))
        # emb = self.embedding(input[0])
        
        
        packed_input = nn.utils.rnn.pack_padded_sequence(input[0], input[1].cpu(), 
                                                         batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_input, self.prev_state)

        # output size is (batch_size, lstm_num_hidden)
        return h_n.squeeze(0)


class BLSTM_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_pooling = config.max_pooling
        self.lstm_num_hidden = config.lstm_num_hidden
        self.device = config.device

        self.lstm = nn.LSTM(config.embedding_dim, config.lstm_num_hidden, 1,
                            dropout= config.dpout_lstm, 
                            bidirectional = True
                            )
        self.prev_state = None
        


    def forward(self, input):
        if self.prev_state == None:
            self.prev_state = (torch.zeros(2, input[0].shape[0], 
                                           self.lstm_num_hidden).to(input[0].device),
                               torch.zeros(2, input[0].shape[0],
                                           self.lstm_num_hidden).to(input[0].device))
    
        packed_input = nn.utils.rnn.pack_padded_sequence(input[0], input[1].cpu(), batch_first=True, 
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