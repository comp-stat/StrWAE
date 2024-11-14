from typing import Optional

import torch
from torch import nn, optim
import torch.nn.functional as F
from networks.base import MLPBlock
from penalties.mmd import mmd_penalty
from utils.initialize import init_params

from utils.sgdr import CosineAnnealingWarmUpRestarts

class Autoencoder(nn.Module):
    def __init__(self, 
                 label_dim: int, 
                 latent_dim: int,
                 learning_rate: float,
                 encoder: nn.Module = MLPBlock,
                 decoder: nn.Module = MLPBlock,
                 activation: nn.Module = nn.LeakyReLU):
        super().__init__()

        self.encoder = encoder()
        self.decoder = decoder()
        
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.activation = activation(inplace=True)


    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def get_losses(self, x: torch.Tensor):
        x_hat = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='none')
        recon_loss = recon_loss.sum(dim=[1,2,3]).mean(dim=[0])
        return {'recon_loss': recon_loss}
    
    def get_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr)

        scheduler = None
        return {"optimizer": optimizer, "scheduler": scheduler}


class Embedder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 base_channels: int,
                 label_dim: int,
                 latent_dim: int,
                 attr_dim: int = 0,
                 input_size: int = 128,
                 kernel_size: int = 5,
                 conv_layers: int = 3,
                 fc_layers: int = 0,
                 learning_rate: float = 0.001,
                 optim_beta: float = 0.5,
                 encoder: nn.Module = MLPBlock,
                 activation: nn.Module = nn.LeakyReLU):
        
        super().__init__()

        # encoder: entity embedding networks
        # decoder: the last layer of classifier
        self.lr = learning_rate
        self.attr_dim = attr_dim
        self.latent_dim = latent_dim
        self.optim_beta = optim_beta

        if "convnet2" not in str(encoder):
            # downscaling with the kernel of size "kernel_size"
            self.encoder = encoder(
                in_channels=in_channels,
                base_channels=base_channels,
                latent_dim=latent_dim,
                kernel_size=kernel_size,
                input_size=input_size,
                conv_layers=conv_layers,
                fc_layers=fc_layers,
                activation=activation
            )
        else: # when encoder==networks.convnet2.Encoder
            # Downscaling with half a kernel size of "kernel_size" and the other half of 3.
            half_layers = (conv_layers+1)//2
            self.encoder = encoder(
                in_channels=in_channels,
                base_channels=base_channels,
                latent_dim=latent_dim,
                kernel_sizes=[kernel_size]*half_layers+[3]*(conv_layers-half_layers),
                input_size=input_size,
                scaling_steps=1,
                conv_steps=0,
                block_layers=conv_layers,
                fc_layers=fc_layers,
                activation=activation,
            )

        enc_dim = (
            (2**(conv_layers-2)*base_channels * (input_size//(2**(conv_layers-1)))**2)//2
        )
        self.encoder.encoder_y = nn.Sequential(
            nn.Linear(
                enc_dim, latent_dim
            ),
            nn.BatchNorm1d(latent_dim),
        )

        self.encoder.encoder_s = nn.Linear(enc_dim, attr_dim) if attr_dim > 0 else nn.Identity()

        self.decoder = nn.Sequential(
            activation(inplace=True),
            nn.Linear(latent_dim, label_dim)
        )

    def forward(self, x: torch.Tensor):
        y_embedding, s_logit = self.encode(x)
        y_logit = self.decode(y_embedding)

        return y_logit, s_logit
    
    def encode(self, x: torch.Tensor):
        latent = self.encoder(x)
        y_embedding = self.encoder.encoder_y(latent)
        s_logit = self.encoder.encoder_s(latent)

        return y_embedding, s_logit
    
    def decode(self, z: torch.Tensor):
        return self.decoder(z)
    
    def get_losses(self,
                   x: torch.Tensor,
                   y: Optional[torch.Tensor] = None,
                   s: Optional[torch.Tensor] = None,
                   mode: str = "",
                   valid: bool = False):

        y_embedding, s_logit = self.encode(x)
        y_logit = nn.Identity()

        if mode == "labeled":
            y_logit = self.decode(y_embedding)
            if len(y.size()) > 1: # one-hot encoded
                y = y.max(dim=1).indices.squeeze()
            loss = F.cross_entropy(y_logit, y)

        elif mode == "attribute":
            loss = F.binary_cross_entropy_with_logits(
                s_logit, s.float(), reduction="none"
            ).sum(dim=1).mean()

        if not valid:
            return loss
        else:
            return loss, s_logit, y_logit
    
    def get_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            betas=(self.optim_beta, 0.999)
        )

        scheduler = None

        return {
            "optimizer": optimizer,
            "scheduler": scheduler
        }
