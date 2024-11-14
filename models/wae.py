import torch
from torch import nn, optim
import torch.nn.functional as F

from networks.base import MLPBlock
from models.base import Autoencoder

from penalties.mmd import mmd_penalty


"""
=================================
Various WAE models (unsupervised)
=================================
"""

class WAE_MMD(Autoencoder): # for image dataset
    def __init__(self, 
                 in_channels: int, 
                 base_channels: int, 
                 latent_dim: int,
                 learning_rate: float,
                 encoder: nn.Module = MLPBlock,
                 decoder: nn.Module = MLPBlock,
                 activation: nn.Module = nn.LeakyReLU):

        super().__init__(in_channels, base_channels, latent_dim, learning_rate, activation, encoder, decoder)
        self.latent_dim = latent_dim
    
    def get_losses(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z) # range: [-1.0, 1.0]
        recon_loss = F.mse_loss(x_hat, x, reduction='none')
        recon_loss = recon_loss.sum(dim=[1,2,3]).mean(dim=[0])
        
        z_prior = torch.randn_like(z)
        mmd = mmd_penalty(z, z_prior)
        return {'recon_loss': recon_loss, 'mmd_penalty': mmd}

    def get_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr
        )

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 0.5**(epoch>=30) * 0.2**(epoch>=50)
        )
        
        return {"optimizer": optimizer, "scheduler": scheduler}

class WAE_GAN(Autoencoder):
    def __init__(self,
                 disc_size: int,
                 label_dim: int,
                 latent_dim: int,
                 disc_layers: int,
                 learning_rate: float,
                 learning_rate_gan: float,
                 optim_beta: float = 0.5,
                 optim_gan_beta: float = 0.5,
                 encoder: nn.Module = MLPBlock,
                 decoder: nn.Module = MLPBlock,
                 discriminator: nn.Module = MLPBlock,
                 activation: nn.Module = nn.LeakyReLU):
        
        super().__init__(
            label_dim,
            latent_dim,
            learning_rate,
            encoder,
            decoder,
            activation
        )

        self.lr_gan = learning_rate_gan
        self.optim_beta = optim_beta
        self.optim_gan_beta = optim_gan_beta
        self.discriminator = discriminator(
            input_dim=latent_dim,
            hidden_dim=disc_size,
            layers=disc_layers,
            activation=activation,
        )

    def get_losses(self,
                   x: torch.Tensor,
                   mode: str = "",
                   valid: bool = False):
        
        z = self.encoder(x)

        if mode == "reconstruction":
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x, reduction='none')
            loss = loss.sum(dim=[1,2,3]).mean(dim=[0])

        elif mode == "adversarial":
            q_z = self.discriminator(z)
            loss = F.binary_cross_entropy_with_logits(
                q_z, torch.ones_like(q_z)
            )

        elif mode == "discriminator":
            z_prior = torch.randn_like(z)
            p_z = self.discriminator(z_prior)
            q_z = self.discriminator(z.detach())
            loss = (
                F.binary_cross_entropy_with_logits(
                    p_z, torch.ones_like(p_z)
                )
                + F.binary_cross_entropy_with_logits(
                    q_z, torch.zeros_like(q_z)
                )
            )

        if not valid:
            return loss
        else:
            return loss, nn.Identity(), nn.Identity()
    
    def get_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            betas=(self.optim_beta, .999)
        )
        optimizer_adv = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_gan,
            betas=(self.optim_gan_beta, .999)
        )

        scheduler = None
        scheduler_adv = None

        return {
            "optimizers": {
                "reconstruction": optimizer,
                "generator": optimizer,
                "discriminator": optimizer_adv
            },
            "schedulers": {
                "reconstruction": scheduler,
                "generator": None,
                "discriminator": scheduler_adv
            }
        }
