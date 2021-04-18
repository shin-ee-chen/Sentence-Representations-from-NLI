import time
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import utils
from models import NLITrainer


class MyLRCallback(pl.Callback):
    def on_init_start(self, trainer):
        self.best_acc = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        trainer.optimizer.lr = trainer.optimizer.lr * 0.1
        # print(outputs)
        trainer.train = False


def train_model(args):
    """
    Function for training and testing a NLI model.
    Inputs:
        args - Namespace object from the argument parser
    """
    
    save_name = args.encoder_type
    CHECKPOINT_PATH = "./checkpoints"
    if args.encoder_type == "BLSTM_Encoder" and args.max_pooling == "True":
        save_name = save_name + "_Max"
   
    # Load dataset
    train_loader, val_loader, test_loader, TEXT = utils.load_data(args.batch_size, 
                                                                       args.embedding_dim,
                                                                       glove_name = args.glove_name,
                                                                       device = args.device)

    print("Data loaded")
    args.embedding = utils.load_pretrained_embed(TEXT,args.embedding_dim)
    print("Pretrained vectors loaded")
    # Create a PyTorch Lightning trainer with the generation callback
    if args.debug == "True":
        myLRcallback = MyLRCallback()
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                                  # Where to save models
                             checkpoint_callback=ModelCheckpoint(save_weights_only=True, 
                                                                 mode="max", monitor="val_acc"), # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                             gpus=1 if str(args.device)=="cuda" else 0,                                                     # We run on a single GPU (if possible)
                             max_epochs=args.epochs,                                                                             # How many epochs to train for if no patience is set
                             callbacks=[LearningRateMonitor("epoch")], 
                            #  callbacks= [myLRcallback],                                               # Log learning rate every epoch
                             progress_bar_refresh_rate=10,
                             limit_train_batches=10,
                             limit_val_batches=10,
                             limit_test_batches=100
                             ) 
    else:
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                                  # Where to save models
                             checkpoint_callback=ModelCheckpoint(save_weights_only=True, 
                                                                 mode="max", monitor="val_acc"), # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                             gpus=1 if str(args.device)=="cuda" else 0,                                                     # We run on a single GPU (if possible)
                             max_epochs=args.epochs,                                                                             # How many epochs to train for if no patience is set
                             callbacks=[LearningRateMonitor("epoch")],                                                   # Log learning rate every epoch
                             progress_bar_refresh_rate=100
                             )                                                                   # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need



    pl.seed_everything(args.seed) # To be reproducable
    
    
    model = NLITrainer(args)
    trainer.fit(model, train_loader, val_loader)
    model = NLITrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best ch
    
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    
    return model




if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument("--lstm_num_hidden", type=int, default=10, 
                        help="encoder nhid dimension")
    parser.add_argument('--fc_dim', default=512, type=int,
                        help='nhid dimension of fully connect layers')
    parser.add_argument('--dpout_fc', default=0., type=float,
                        help='dropout rate of fc') 
    parser.add_argument('--dpout_lstm', default=0., type=float,
                        help='dropout rate of lstm')                 
    parser.add_argument('--encoder_type', default="AWE", type=str, 
                        choices=["AWE", "LSTM_Encoder", "BLSTM_Encoder"],
                        help='Type of encoder, choose from [AWE, LSTM_Encoder, BLSTM_Encoder]')
    parser.add_argument('--max_pooling', type=str, choices=["False","True"] ,default= "True",
                        help="Whether to use small dataset to debug")


    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size to use for training')

    # Other hyperparameters
    parser.add_argument('--epochs', default=3, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    
    parser.add_argument('--log_dir', default='logs/', type=str,
                        help='Directory where the PyTorch Lightning logs ' + \
                             'should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation. '+ \
                             'Not to be used in conjuction with SLURM jobs.')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")
    parser.add_argument('--debug', type=str, choices=["False","True"] ,default= "True",
                        help="Whether to use small dataset to debug")
    parser.add_argument('--glove_name', type=str, default= "840B",
                        help="glove name: 6B/840B")

                        
    args = parser.parse_args()

    time_b = time.time()
    train_model(args)
    time_e = time.time()
    print("Time elapsed:{:.2} mins".format((time_e - time_b)/60))