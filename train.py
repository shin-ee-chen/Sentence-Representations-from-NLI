import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import argparse
import os
import utils
from models import NLINet

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
        lambda1 = lambda epoch: 0.65 ** epoch
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        # need to change scheduler 
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        
        # return [optimizer], [scheduler]
        return [optimizer]


    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        text, labels = [batch.premise[0], batch.hypothesis[0]], batch.label
        preds = self.model(text)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log('train_acc', acc, on_step=False, on_epoch=True) # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        return loss # Return tensor to call ".backward" on


    def validation_step(self, batch, batch_idx):
        text, labels = [batch.premise[0], batch.hypothesis[0]], batch.label
        preds = self.model(text).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # self.config.lr /= 2 
        # print(self.hparams.lr)
        # self.log('lr', self.config.lr)
        self.log('val_acc', acc) # By default logs it per epoch (weighted average over batches)


    def test_step(self, batch, batch_idx):
        text, labels = [batch.premise[0], batch.hypothesis[0]], batch.label
        preds = self.model(text).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards


class LRCallback(pl.Callback):
    
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        

    def on_epoch_end(self, trainer, pl_module):
        pass

def train_model(args):
    """
    Function for training and testing a NLI model.
    Inputs:
        args - Namespace object from the argument parser
    """
    
    save_name = args.encoder_type
    # device = config.device
   
    # Load dataset
    train_loader, val_loader, test_loader, args.vocab_size = utils.load_data(batch_size=args.batch_size, 
                                                                             device = args.device)
    CHECKPOINT_PATH = "./checkpoints"
    
    if args.debug:
        train_batches=10
        val_batches=10
        test_batches=10
    else:
        train_batches=1.0
        val_batches=1.0
        test_batches=1.0
    
    # Create a PyTorch Lightning trainer with the generation callback
    if args.debug:
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                                  # Where to save models
                             checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"), # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                             gpus=1 if str(args.device)=="cuda" else 0,                                                     # We run on a single GPU (if possible)
                             max_epochs=args.epochs,                                                                             # How many epochs to train for if no patience is set
                             callbacks=[LearningRateMonitor("epoch")],                                                   # Log learning rate every epoch
                             progress_bar_refresh_rate=1,
                             limit_train_batches=train_batches,
                             limit_val_batches=val_batches,
                             limit_test_batches=test_batches
                             ) 
    else:
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                                  # Where to save models
                             checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"), # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                             gpus=1 if str(args.device)=="cuda" else 0,                                                     # We run on a single GPU (if possible)
                             max_epochs=args.epochs,                                                                             # How many epochs to train for if no patience is set
                             callbacks=[LearningRateMonitor("epoch")],                                                   # Log learning rate every epoch
                             progress_bar_refresh_rate=1
                             )                                                                   # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    # trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    # trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")

    # if os.path.isfile(pretrained_filename):
    #     print("Found pretrained model at %s, loading..." % pretrained_filename)
    #     # model = CIFARTrainer.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    # else:
        # pl.seed_everything(42) # To be reproducable
        # model = NLITrainer(model_name=model_name, **kwargs)
        # trainer.fit(model, train_loader, val_loader)
        # model = CIFARTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    # val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    # result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

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
    # parser.add_argument('--embedding_dim', default=300, type=int,
    #                     help='Dimensionality of latent space')
    # parser.add_argument('--z_dim', default=32, type=int,
    #                     help='Dimensionality of latent space')
    # parser.add_argument('--hidden_dims_gen', default=[128, 256, 512], 
    #                     type=int, nargs='+',
    #                     help='Hidden dimensionalities to use inside the ' + \
    #                          'generator. To specify multiple, use " " to ' + \
    #                          'separate them. Example: \"128 256 512\"')
    # parser.add_argument('--hidden_dims_disc', default=[512, 256], 
    #                     type=int, nargs='+',
    #                     help='Hidden dimensionalities to use inside the ' + \
    #                          'discriminator. To specify multiple, use " " to ' + \
    #                          'separate them. Example: \"512 256\"')
    # parser.add_argument('--dp_rate_gen', default=0.1, type=float,
    #                     help='Dropout rate in the discriminator')
    parser.add_argument('--encoder_type', default="AWE", type=str,
                        help='Type of encoder, choose from [AWE, ]')

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
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' + \
                             'To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='logs/', type=str,
                        help='Directory where the PyTorch Lightning logs ' + \
                             'should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation. '+ \
                             'Not to be used in conjuction with SLURM jobs.')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")
    parser.add_argument('--debug', type=bool, default=True,
                        help="Whether to use small dataset to debug")

    args = parser.parse_args()

    train_model(args)