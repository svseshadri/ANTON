import pandas as pd
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import h5py
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import Callback


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        """
        Multi-layer Perceptron designed to match CircuitNet implementation style.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output features
        """
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
            
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward Pass"""
        return self.model(x)

#Modifying EmbeddingDataset class to handle h5 files    
#class EmbeddingDataset(Dataset):
#    def __init__(self, csv_file):
#        self.data = pd.read_csv(csv_file)
#        self.embeddings = self.data.iloc[:, :-1].values
#        self.scores = self.data.iloc[:, -1].values

#    def __len__(self):
#        return len(self.data)

#    def __getitem__(self, idx):
#        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
#        score = torch.tensor(self.scores[idx], dtype=torch.float32)
#        return embedding, score

class EmbeddingDataset(Dataset):
    def __init__(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            self.embeddings = torch.tensor(f['embeddings'][:], dtype=torch.float32)
            self.scores = torch.tensor(f['targets'][:], dtype=torch.float32)
            self.embedding_dim = f.attrs['embedding_dim']

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.scores[idx]


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = None
        self.val_loss = None

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss = trainer.callback_metrics.get('train_loss_epoch', None)
        if self.train_loss is not None:
            self.train_loss = self.train_loss.item()

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss = trainer.callback_metrics.get('val_loss', None)
        if self.val_loss is not None:
            self.val_loss = self.val_loss.item()

class MLPTrainer(pl.LightningModule):
    def __init__(self, input_dim: int, batch_size: int, learning_rate: float, 
                 max_epochs: int, gradient_clip_val: float, T_max: int, eta_min: float):
        super().__init__()
        self.save_hyperparameters()
        
        # Hidden dimensions configured for ~8.3M parameters
        hidden_dims = [2048, 2048, 1024]
        
        self.model = MLP(input_dim, hidden_dims, output_dim=1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat.squeeze(), y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
                    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat.squeeze(), y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.hparams['T_max'],
            eta_min=self.hparams['eta_min']
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    pl.seed_everything(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Add these after your existing seed settings
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Hyperparameters
    HYPERPARAMETERS = {
        'input_dim': 1280,
        'batch_size': 16,
        'learning_rate': 1.1596396728105288e-05,
        'max_epochs': 97,
        'gradient_clip_val': 0.8610360884285442,
        'T_max': 88,
        'eta_min': 6.2631475968196166e-09
    }

    
    run_dir = os.path.join("MLP_avGFP")
    os.makedirs(run_dir, exist_ok=True)

    # Save hyperparameters as a text file
    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        for key, value in HYPERPARAMETERS.items():
            f.write(f"{key}: {value}\n")

    # Data loaders
    train_dataset = EmbeddingDataset('data/avGFP/avgfp_train_ESM2_embeddings.h5')
    val_dataset = EmbeddingDataset('data/avgfp_val_ESM2_embeddings.h5')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=HYPERPARAMETERS['batch_size'], 
        shuffle=True, 
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context='spawn',
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=HYPERPARAMETERS['batch_size'],
        pin_memory=True,
        num_workers=32,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context='spawn',
    )
    
    # Model
    model = MLPTrainer(**HYPERPARAMETERS)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=run_dir,
        filename='mlp-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    metrics_callback = MetricsCallback()
    
    # Logger
    logger = TensorBoardLogger(save_dir=run_dir, name="tb_logs", version="")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=HYPERPARAMETERS['max_epochs'],
        callbacks=[checkpoint_callback, lr_monitor, metrics_callback],
        logger=logger,
        gradient_clip_val=HYPERPARAMETERS['gradient_clip_val'],
        gradient_clip_algorithm="norm",
        enable_progress_bar=True,
        log_every_n_steps=1,
        deterministic=True,
        accelerator='gpu',
        devices=1,
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final model
    trainer.save_checkpoint(os.path.join(run_dir, "final_model.ckpt"))