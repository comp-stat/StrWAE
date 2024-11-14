from typing import Optional

import torch
from torch import nn, optim
import torch.nn.functional as F

from networks.base import MLPBlock
from models.wae import WAE_GAN

from penalties.hsic import hsic_penalty
from utils.sgdr import CosineAnnealingWarmUpRestarts
from penalties import mmd_penalty


class AAE(WAE_GAN):
    """
        Task: Semi-supervised Classification
        Data: MNIST, SVHN (X = image, Y = digit label)
        Decoder structure:
            - (Z, Y) -> X
            - Independence of Z and Y is implicitly assumed
    """
    def __init__(self,
                 in_channels: int,
                 base_channels: int,
                 hidden_size: int,
                 disc_size: int,
                 label_dim: int,
                 latent_dim: int,
                 linear_bn: bool,
                 kernel_size: int,
                 input_size: int,
                 conv_layers: int,
                 fc_layers: int,
                 disc_layers: int,
                 learning_rate: float,
                 learning_rate_gan: float,
                 learning_rate_sup: float,
                 encoder: nn.Module = MLPBlock,
                 decoder: nn.Module = MLPBlock,
                 discriminator: nn.Module = MLPBlock,
                 activation: nn.Module = nn.LeakyReLU):
        
        super().__init__(
            disc_size=disc_size,
            label_dim=label_dim,
            latent_dim=latent_dim,
            disc_layers=disc_layers,
            learning_rate=learning_rate,
            learning_rate_gan=learning_rate_gan,
            encoder=encoder,
            decoder=decoder,
            discriminator=discriminator,
            activation=activation
        )
        self.lr_sup = learning_rate_sup
        self.latent_dim = latent_dim

        '''
        Encoding (= f(X, Y_hat)):
            1. encoder: x -> h
            2. encoder.classifier: h -> y_hat
            3. encoder.styler: h -> z
        '''
        self.encoder = encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            hidden_size=hidden_size,
            latent_dim=latent_dim + label_dim,
            linear_bn=linear_bn,
            kernel_size=kernel_size,
            input_size=input_size,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            activation=activation
        ) # cnn
        self.encoder.classifier = nn.Linear(latent_dim + label_dim, label_dim)
        self.encoder.styler = nn.Linear(latent_dim + label_dim, latent_dim)
        
        '''
        Decoding (= g(Y, Z))
            1. decoder: (Y, Z) -> x_hat
        '''
        self.decoder = decoder(
            base_channels=base_channels,
            out_channels=in_channels,
            hidden_size=hidden_size,
            latent_dim=latent_dim + label_dim,
            linear_bn=linear_bn,
            kernel_size=(
                kernel_size if kernel_size % 2 == 0 else ((kernel_size // 2) + 1) * 2
            ),
            output_size=input_size,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            activation=activation
        ) # (z,y) -> x; cnn

        self.discriminator_y = discriminator(
            input_dim=label_dim,
            hidden_dim=disc_size,
            layers=disc_layers,
            activation=activation
        )

        self.discriminator_z = discriminator(
            input_dim=latent_dim,
            hidden_dim=disc_size,
            layers=disc_layers,
            activation=activation
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        z, y_logit, y_hat = self.encode(x)
        x_hat = self.decode(z, y_hat if y is None else y)
        return x_hat
    
    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        label = self.encoder.classifier(z)
        y_hat = F.softmax(label, dim=1) # probability vector
        style = self.encoder.styler(z)

        return style, label, y_hat

    def decode(self, z: torch.Tensor, y: torch.Tensor):
        return self.decoder(torch.cat([z, y], dim=1))
    
    def first_operation(self,
                        labeled_x: torch.Tensor,
                        unlabeled_x: Optional[torch.Tensor] = None):
        
        labeled_z, labeled_y_logit, labeled_y_hat = (
            self.encode(labeled_x)
        )
        unlabeled_z, unlabeled_y_hat = nn.Identity(), nn.Identity()

        if not unlabeled_x is None:
            unlabeled_z, _, unlabeled_y_hat = self.encode(unlabeled_x)

        return (
            labeled_z, labeled_y_logit, labeled_y_hat,
            unlabeled_z, unlabeled_y_hat
        )

    def get_losses(self,
                   ingredient,
                   labeled_x: torch.Tensor,
                   labeled_y: torch.Tensor,
                   unlabeled_x: Optional[torch.Tensor] = None,
                   mode: str = "",
                   valid: bool = False):
        
        (
            labeled_z, labeled_y_logit, labeled_y_hat,
            unlabeled_z, unlabeled_y_hat
        ) = ingredient

        # train
        if not valid:
            if mode == "reconstruction": # using unlabeled data
                unlabeled_x_hat = self.decode(unlabeled_z, unlabeled_y_hat)
                loss = torch.mean(
                    torch.sum((unlabeled_x - unlabeled_x_hat).pow(2), dim=[1,2,3])
                )

            elif mode == "generator": # using unlabeled data 
                unlabeled_q_z = self.discriminator_z(unlabeled_z)
                unlabeled_q_y = self.discriminator_y(unlabeled_y_hat)
                loss_z = (
                    F.binary_cross_entropy_with_logits(
                        unlabeled_q_z, torch.ones_like(unlabeled_q_z)
                    )
                )
                loss_y = (
                    F.binary_cross_entropy_with_logits(
                        unlabeled_q_y, torch.ones_like(unlabeled_q_y)
                    )
                )
                loss = loss_z + loss_y

            elif mode == "discriminator": # using unlabeled data
                real_z = (
                    torch.randn(len(unlabeled_z), self.latent_dim)
                ).type_as(unlabeled_z)

                p_z = self.discriminator_z(real_z)
                q_z = self.discriminator_z(unlabeled_z.detach())

                adv_z_loss = (
                    F.binary_cross_entropy_with_logits(
                        p_z, torch.ones_like(p_z)
                    )
                    + F.binary_cross_entropy_with_logits(
                        q_z, torch.zeros_like(q_z)
                    )
                )

                # sample Y from the discrete uniform
                indices_y = torch.randint(
                    low=0, high=self.label_dim, size=(len(unlabeled_y_hat), )
                )
                real_y = (
                    torch.eye(self.label_dim)[indices_y].type_as(
                        unlabeled_y_hat
                    )
                )

                p_y = self.discriminator_y(real_y)
                q_y = self.discriminator_y(unlabeled_y_hat.detach())

                adv_y_loss = (
                    F.binary_cross_entropy_with_logits(
                        p_y, torch.ones_like(p_y)
                    )
                    + F.binary_cross_entropy_with_logits(
                        q_y, torch.zeros_like(q_y)
                    )
                )
                
                loss = adv_z_loss + adv_y_loss

            elif mode == "supervised": # using labeled data
                loss = F.cross_entropy(labeled_y_logit, labeled_y)
            
            elif mode == "hsic":
                labeled_loss = hsic_penalty(labeled_z, labeled_y_hat)
                unlabeled_loss = hsic_penalty(unlabeled_z, unlabeled_y_hat)

                loss = labeled_loss + unlabeled_loss
                
            elif mode == "mmd":
                loss = mmd_penalty(
                    unlabeled_y_hat,
                    F.one_hot(labeled_y, num_classes=10).float()
                )

            else:
                NotImplementedError("Invalid mode: {}".format(mode))

            return loss
        
        else: # validation
            if mode == "reconstruction":
                labeled_x_hat = self.decode(labeled_z, labeled_y_hat)
                loss = 1/2 * torch.mean(
                    torch.sum((labeled_x - labeled_x_hat).pow(2), dim=[1,2,3])
                )
                
            elif mode == "generator":
                labeled_q_z = self.discriminator_z(labeled_z)
                labeled_q_y = self.discriminator_y(labeled_y_hat)
                loss_z = F.binary_cross_entropy_with_logits(
                    labeled_q_z, torch.ones_like(labeled_q_z)
                )
                loss_y = F.binary_cross_entropy_with_logits(
                    labeled_q_y, torch.ones_like(labeled_q_y)
                )
                loss = loss_z + loss_y

            elif mode == "supervised":
                loss = F.cross_entropy(labeled_y_logit, labeled_y)
            
            elif mode == "hsic":
                loss = hsic_penalty(labeled_z, labeled_y_hat)
                
            else:
                NotImplementedError("Invalid mode: {}".format(mode))
            
            return loss, labeled_y_logit, labeled_y_hat

    def get_optimizers(self):
        
        optimizer_autoencoder = optim.SGD(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            momentum=0.9
        )

        scheduler_autoencoder = CosineAnnealingWarmUpRestarts(
            optimizer_autoencoder,
            T_0=50,
            T_mult=1,
            eta_max=self.lr,
            T_up=25,
            gamma=0.5
        )

        optimizer_generator = optim.SGD(
            list(self.encoder.parameters()),
            lr=self.lr_gan,
            momentum=0.1
        )

        scheduler_generator = CosineAnnealingWarmUpRestarts(
            optimizer_generator,
            T_0=50,
            T_mult=1,
            eta_max=self.lr_gan,
            T_up=25,
            gamma=0.5
        )

        optimizer_discriminator = optim.SGD(
            list(self.discriminator_y.parameters())
            + list(self.discriminator_z.parameters()),
            lr=self.lr_gan,
            momentum=0.1
        )

        scheduler_discriminator = CosineAnnealingWarmUpRestarts(
            optimizer_discriminator,
            T_0=50,
            T_mult=1,
            eta_max=self.lr_gan,
            T_up=25,
            gamma=0.5
        )
        
        optimizer_sup = optim.SGD(
            list(self.encoder.parameters()),
            lr=self.lr_sup,
            momentum=0.9
        )

        scheduler_sup = CosineAnnealingWarmUpRestarts(
            optimizer_sup,
            T_0=50,
            T_mult=1,
            eta_max=self.lr_sup,
            T_up=25,
            gamma=0.5
        )

        return {
            "optimizers": {
                "reconstruction": optimizer_autoencoder,
                "generator": optimizer_generator,
                "discriminator": optimizer_discriminator,
                "supervised": optimizer_sup,
                "mmd": optimizer_generator,
                "hsic": optimizer_generator
            },
            "schedulers": {
                "reconstruction": scheduler_autoencoder,
                "generator": scheduler_generator,
                "discriminator": scheduler_discriminator,
                "supervised": scheduler_sup,
                "mmd": scheduler_generator,
                "hsic": scheduler_generator
            }
        }
