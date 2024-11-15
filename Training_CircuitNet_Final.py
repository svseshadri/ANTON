import pandas as pd
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import Callback

def clip_sin(x):
    A = 3.388235
    eps = 0.01
    mask = (x >= -3) & (x <= 3)
    return torch.where(mask, A * torch.sin(torch.pi / 6 * x),
                       torch.where(x < -3, -A + eps * (x + 3), A + eps * (x - 3)))

class CMU(nn.Module):
    def __init__(self, num_neurons, num_ports, activation):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_ports = num_ports
        
        # Linear transformation and product between neurons
        self.V = nn.Parameter(torch.randn(num_neurons, num_neurons, 2))
        self.W = nn.Parameter(torch.randn(num_neurons, num_neurons, num_neurons))

        #I/O port masks
        self.register_buffer('P_in', torch.zeros(num_neurons))
        self.register_buffer('P_out', torch.zeros(num_neurons))
        self.P_in[:num_ports] = 1
        self.P_out[-num_ports:] = 1

        #Inter-CMU transformation
        self.W_ji = nn.Parameter(torch.randn(num_neurons, num_neurons))
        #Bias
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        #Normalization
        self.norm = nn.LayerNorm(num_neurons)
        #Activation
        if activation == "clip_sin":
            self.activation = clip_sin
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = torch.tanh

    def initialize(self, data):
        with torch.no_grad():
            self.state = data
    
    def forward(self, x_other=None):
        x = self.state
        linear_j = torch.einsum('bi,ij->bj', x, self.V[:,:,0])
        linear_k = torch.einsum('bi,ij->bj', x, self.V[:,:,1])
        
        # Product between neurons
        product = 0.5 * torch.einsum('bi,ijk,bk->bj', x, self.W, x)
        
        intra_update = self.activation(self.norm(linear_j + linear_k + product + self.bias))

        #Inter-CMU update
        if x_other is not None:
            x_out = self.P_out * x_other
            inter_update = self.activation(self.norm(self.P_in * torch.matmul(x_out, self.W_ji.t())))
        else:
            inter_update = 0

        self.state = intra_update + inter_update
        return self.state

class CircuitNetLP(pl.LightningModule):
    def __init__(self, input_dim, num_cmus, num_neurons, num_ports, num_iterations, activation):
        super().__init__()
        self.num_cmus = num_cmus
        self.num_neurons = num_neurons
        self.num_ports = num_ports
        self.num_iterations = num_iterations
        
        # Input layer
        self.input_projection = nn.Linear(input_dim, num_cmus * num_neurons)
        
        # Create CMUs
        self.cmus = nn.ModuleList([CMU(num_neurons, num_ports, activation) for _ in range(num_cmus)])
        
        # Inter-CMU connections
        self.register_buffer('connection_mask', self._create_connection_mask())
        
        # Output layer
        self.output_layer = nn.Linear(num_cmus * num_neurons, 1)

    def _create_connection_mask(self):
        mask = torch.zeros(self.num_cmus, self.num_cmus)
        for i in range(self.num_cmus):
            for j in range(self.num_cmus):
                if abs(i - j) <= 2 or abs(i - j) >= self.num_cmus - 2:
                    mask[i, j] = 1
        return mask

    def forward(self, x):
        # Input layer
        x = self.input_projection(x)
        x = x.view(-1, self.num_cmus, self.num_neurons)

        # Initialize CMUs with projected input
        for i in range(self.num_cmus):
            self.cmus[i].initialize(x[:, i, :])
        
        # Perform signal transmission iterations
        for _ in range(self.num_iterations):
            new_x = []
            for i in range(self.num_cmus):
                # Gather inputs from connected CMUs
                connected_cmus = self.connection_mask[i].nonzero().squeeze()
                x_other = torch.stack([self.cmus[j].state for j in connected_cmus]).sum(dim=0) if connected_cmus.numel() > 0 else None
                
                # Update CMU state
                new_x.append(self.cmus[i](x_other))
            
            x = torch.stack(new_x, dim=1)
        
        # Output layer
        x = x.view(-1, self.num_cmus * self.num_neurons)
        return self.output_layer(x)
    
# Old EmbeddingDataset to handle CSV files
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

# New EmbeddingDataset class to handle HDF5 files for avGFP. Loads
#everything into memory at initialization for better memory management

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

class CircuitNetTrainer(pl.LightningModule):
    def __init__(self, input_dim, num_cmus, num_neurons, num_ports, num_iterations, batch_size,
                 learning_rate, max_epochs, gradient_clip_val, activation, T_max, eta_min):
        super().__init__()
        self.save_hyperparameters()
        self.model = CircuitNetLP(input_dim, num_cmus, num_neurons, num_ports, num_iterations, activation)
        self.test_outputs = []

#Removed random Xavier weight initialization to more closely reflect the paper.  Different CMUs are
#directly initialized by different portions of the input data
    
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
        'num_cmus': 5,
        'num_neurons': 114,
        'num_ports': 40,
        'num_iterations': 8,
        'batch_size': 16,
        'learning_rate': 1.4982696171983996e-06,
        'max_epochs': 110,
        'gradient_clip_val': 8.55759990993581,
        'activation' : "clip_sin",
        'T_max' : 136,
        'eta_min' : 9.042713531820194e-09,
    }

    run_dir = os.path.join("CircuitNet_avGFP2")
    os.makedirs(run_dir, exist_ok=True)

    # Save hyperparameters as a text file
    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        for key, value in HYPERPARAMETERS.items():
            f.write(f"{key}: {value}\n")

    # Data loaders
    train_dataset = EmbeddingDataset('data/avGFP/avgfp_train_ESM2_embeddings.h5')
    val_dataset = EmbeddingDataset('data/avGFP/avgfp_val_ESM2_embeddings.h5')
    
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
    model = CircuitNetTrainer(**HYPERPARAMETERS)
    
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
        callbacks=[checkpoint_callback, early_stop_callback, 
                   lr_monitor, metrics_callback],
        logger=logger,
        gradient_clip_val=HYPERPARAMETERS['gradient_clip_val'],
        gradient_clip_algorithm="norm",
        enable_progress_bar=True,
        log_every_n_steps=1,
        deterministic=True,
        accelerator='gpu',
        devices=1,
        strategy='single_device',
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final model
    trainer.save_checkpoint(os.path.join(run_dir, "final_model.ckpt"))