from typing import Optional, List

import torch
from torch import nn, optim
import torch.nn.functional as F

from networks.base import MLPBlock
from models.wae import WAE_GAN

from penalties.hsic import hsic_penalty, dhsic_penalty
from penalties.mmd import mmd_penalty

from utils.sgdr import CosineAnnealingWarmUpRestarts

"""
=====================================================
StrWAE to Invariant Representations (Semi-supervised)
=====================================================
"""

class StrWAE_semi_cls(WAE_GAN):
    """
        Task: Semi-supervised Classification
        Data: MNIST, SVHN (X = image, Y = digit label)
        Decoder structure: 
            - (Z, Y) -> X
            - Z independent of Y
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
                 optim_beta: float,
                 optim_gan_beta: float,
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
            optim_beta=optim_beta,
            optim_gan_beta=optim_gan_beta,
            encoder=encoder,
            decoder=decoder,
            discriminator=discriminator,
            activation=activation
        )
        
        '''
        Encoding (= f(X, Y_hat)):
            1. encoder: x -> h = (zz, y_hat)
            2. encoder.style_y_to_style: (zz, y_hat) -> z
            (y_hat is replaced by y when y is observed)
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
        ) # x (image) -> h = (zz, y_hat) (vector); cnn

        self.encoder.style_y_to_style = nn.Linear(
            latent_dim + label_dim, latent_dim
        ) # (zz, y_hat) -> z
        
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
        ) # (z, y) -> x; cnn
        
        self.discriminator = discriminator(
            input_dim=latent_dim,
            hidden_dim=disc_size,
            layers=disc_layers,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        z, _, y_hat = self.encode(x)
        x_hat = self.decode(z, y_hat if y is None else y)
        return x_hat
    
    def encode(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        # x -> h = (zz, y_hat)
        # (zz, y_hat) -> z if y is missing
        # (zz, y) -> z if y is observed
        h = self.encoder(x)

        style, label = h.split([self.latent_dim, self.label_dim], dim=1)
        y_hat = F.softmax(label, dim=1) # probability vector
        style = (
            self.encoder.style_y_to_style(torch.cat([style, y_hat], dim=1)) # missing Y
            if y is None else
            self.encoder.style_y_to_style(torch.cat([style, y], dim=1)) # observed Y
        )
        return style, label, y_hat

    def decode(self, z: torch.Tensor, y: torch.Tensor):
        return self.decoder(torch.cat([z, y], dim=1))

    def first_operation(self,
                        labeled_x: torch.Tensor,
                        unlabeled_x: torch.Tensor = None):
        
        labeled_z, labeled_y_logit, labeled_y_hat = self.encode(labeled_x)

        if unlabeled_x is None:
            return (
                labeled_z, labeled_y_logit, labeled_y_hat,
                nn.Identity(), nn.Identity()
            )
        else:
            unlabeled_z, _, unlabeled_y_hat = self.encode(unlabeled_x)

            return (
                labeled_z, labeled_y_logit, labeled_y_hat,
                unlabeled_z, unlabeled_y_hat
            )

    def get_losses(self,
                   ingredient,
                   labeled_x: torch.Tensor,
                   labeled_y: Optional[torch.Tensor] = None,
                   unlabeled_x: Optional[torch.Tensor] = None,
                   mode: str = "",
                   valid: bool = False):
        
        labeled_loss = 0
        unlabeled_loss = 0
        
        (
            labeled_z, labeled_y_logit, labeled_y_hat,
            unlabeled_z, unlabeled_y_hat
        ) = ingredient

        # Labeled loss
        if mode == "reconstruction":
            # Reconstruction loss
            labeled_x_hat = self.decode(
                labeled_z, F.one_hot(labeled_y, num_classes=10).float()
            )
            
            labeled_loss = torch.mean(
                torch.sum((labeled_x - labeled_x_hat).pow(2), dim=[1,2,3])
            )

            if not valid:
                unlabeled_x_hat = self.decode(unlabeled_z, unlabeled_y_hat)

                # MSE loss
                unlabeled_loss = torch.mean(torch.sum(
                    (unlabeled_x - unlabeled_x_hat).pow(2), dim=[1,2,3])
                )

        elif mode == "supervised":
            # Labeled penalty
            labeled_loss = F.cross_entropy(labeled_y_logit, labeled_y)

        elif mode == "mmd":
            unlabeled_loss = mmd_penalty(
                unlabeled_y_hat,
                F.one_hot(labeled_y, num_classes=10).float()
            )
        
        elif mode == "generator": # prior matching
            labeled_q_z = self.discriminator(labeled_z)
            labeled_loss = F.binary_cross_entropy_with_logits(
                labeled_q_z, torch.ones_like(labeled_q_z)
            )
            
            if not valid:
                unlabeled_q_z = self.discriminator(unlabeled_z)
                unlabeled_loss = F.binary_cross_entropy_with_logits(
                    unlabeled_q_z, torch.ones_like(unlabeled_q_z)
                )
        
        elif mode == "hsic":
            labeled_loss = hsic_penalty(labeled_z, labeled_y_hat)

            if not valid:
                unlabeled_loss = hsic_penalty(unlabeled_z, unlabeled_y_hat)

        elif mode == "discriminator": # adversarial training
            labeled_z_real = (
                torch.randn_like(labeled_z).type_as(labeled_z)
            )

            labeled_p_z = self.discriminator(labeled_z_real)
            labeled_q_z = self.discriminator(labeled_z.detach())
            labeled_loss = (
                F.binary_cross_entropy_with_logits(
                    labeled_p_z, torch.ones_like(labeled_p_z)
                )
                + F.binary_cross_entropy_with_logits(
                    labeled_q_z, torch.zeros_like(labeled_q_z)
                )
            )
            
            if not valid:
                unlabeled_z_real = (
                    torch.randn_like(unlabeled_z)
                ).type_as(unlabeled_z)

                unlabeled_p_z = self.discriminator(unlabeled_z_real)
                unlabeled_q_z = self.discriminator(unlabeled_z.detach())
                unlabeled_loss = (
                    F.binary_cross_entropy_with_logits(
                        unlabeled_p_z, torch.ones_like(unlabeled_p_z)
                    )
                    + F.binary_cross_entropy_with_logits(
                        unlabeled_q_z, torch.zeros_like(unlabeled_q_z)
                    )
                )
        else:
            raise ValueError("Invalid mode")
        
        loss = labeled_loss + unlabeled_loss

        if not valid:
            return loss
        else:
            return loss, nn.Identity(), labeled_y_hat

    def get_optimizers(self):
        optimizer = optim.RAdam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            betas=(self.optim_beta, 0.999)
        )

        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=50,
            T_mult=1,
            eta_max=self.lr,
            T_up=25,
            gamma=0.5
        )

        optimizer_gan = optim.RAdam(
            self.discriminator.parameters(),
            lr=self.lr_gan,
            betas=(self.optim_gan_beta, 0.999)
        )

        scheduler_gan = CosineAnnealingWarmUpRestarts(
            optimizer_gan,
            T_0=50,
            T_mult=1,
            eta_max=self.lr_gan,
            T_up=25,
            gamma=0.5
        )

        return {
            "optimizers": {
                "reconstruction": optimizer,
                "supervised": optimizer,
                "mmd": optimizer,
                "generator": optimizer,
                "hsic": optimizer,
                "discriminator": optimizer_gan
            },
            "schedulers": {
                "reconstruction": scheduler,
                "supervised": scheduler,
                "mmd": scheduler,
                "generator": scheduler,
                "hsic": scheduler,
                "discriminator": scheduler_gan
            }
        }


class StrWAE_embedder(WAE_GAN):
    """
        Task: Conditional generation using embedded variables
        Data: vggface2 
            - X = image
            - Y = identity of person; fully observed but test data has new identities 
            - S = 7 attributes (gender, mouth open, etc.); partially observed
        Decoder Structure: 
            - (Z, Y, S) -> X
            - Z independent of (Y, S)
        Embedding Y and S with pretrained encoders
    """
    def __init__(self, 
                 in_channels: int, 
                 base_channels: int, 
                 hidden_size: int,
                 disc_size: int,
                 label_dim: int,
                 latent_dim: int,
                 attr_dim: int,
                 linear_bn: bool,
                 kernel_sizes: List[List[int]], # (encoder, decoder)
                 input_size: int,
                 scaling_steps: List[int], # (encoder, decoder)
                 conv_steps: List[int], # (encoder, decoder)
                 skip: List[bool], # (encoder, decoder)
                 block_layers: List[int], # (encoder, decoder)
                 fc_layers: List[int], # (encoder, decoder)
                 disc_layers: int,
                 learning_rate: float,
                 learning_rate_gan: float,
                 encoder: nn.Module = MLPBlock, # convnet2.Encoder
                 decoder: nn.Module = MLPBlock, # convnet2.Decoder
                 discriminator: nn.Module = MLPBlock,
                 activation: nn.Module = nn.LeakyReLU,
                 pretrained_path: str = "embedders/vggface2_simple_jit.pt"):

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

        '''
        Encoding (= f(X, Y_hat, S_hat)):
            1. (pretrained) encoder_ys: x -> (y_hat, s_hat)
            2. encoder: x -> z
        '''
        self.encoder = encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            linear_bn=linear_bn,
            kernel_sizes=kernel_sizes[0],
            input_size=input_size,
            scaling_steps=scaling_steps[0],
            conv_steps=conv_steps[0],
            block_layers=block_layers[0],
            fc_layers=fc_layers[0],
            skip=skip[0],
            activation=nn.ReLU,
        ) # x -> z; cnn
        '''
        Decoding (= g(Y, S, Z))
            1. (z, y, s) -> x_hat
        '''
        self.decoder = decoder(
            base_channels=base_channels,
            out_channels=in_channels,
            hidden_size=hidden_size,
            latent_dim=latent_dim + label_dim + attr_dim,
            linear_bn=linear_bn,
            kernel_sizes=kernel_sizes[1],
            output_size=input_size,
            scaling_steps=scaling_steps[1],
            conv_steps=conv_steps[1],
            block_layers=block_layers[1],
            fc_layers=fc_layers[1],
            skip=skip[1],
            activation=activation
        ) # (z, y, s) -> x; cnn + skip_connection

        self.discriminator = discriminator(
            input_dim=latent_dim,
            hidden_dim=disc_size,
            layers=disc_layers,
            activation=nn.ReLU,
        )

        self.encoder_ys = torch.jit.load(f"./checkpoints/{pretrained_path}")
        self.encoder_ys.eval() # fix pretrained (Y, S)-embedder

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            y_hat, s_hat = self.encoder_ys.encode(x)
        z = self.encoder(x)
        x_hat = self.decoder(torch.cat([z, y_hat, s_hat], dim=1))

        return x_hat
    
    def encode(self, x: torch.Tensor):
        with torch.no_grad():
            y_hat, s_hat = self.encoder_ys.encode(x)
        z = self.encoder(x)

        return z, s_hat, y_hat

    def decode(self, z: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        return self.decoder(torch.cat([z, y, s], dim=1))

    def first_operation(self, x: torch.Tensor):
        with torch.no_grad():
            y_hat, s_hat = self.encoder_ys.encode(x)
        return y_hat, s_hat

    def get_losses(self,
                   ingredient,
                   x: torch.Tensor,
                   mode: str = "",
                   valid: bool = False):
        
        y_hat, s_hat = ingredient
        z = self.encoder(x)
        
        if mode == "reconstruction":
            x_hat = self.decoder(torch.cat([z, y_hat, s_hat], dim=1))
            loss = torch.mean(
                torch.sum((x - x_hat).pow(2), dim=[1, 2, 3])
            )
        
        elif mode == "generator": # prior matching
            q_z = self.discriminator(z)
            loss = F.binary_cross_entropy_with_logits(
                q_z, torch.ones_like(q_z)
            )
        
        elif mode == "discriminator": # adversarial training
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

        elif mode == "hsic":
            loss = hsic_penalty(z, torch.cat([y_hat, s_hat], dim=1))

        if not valid:
            return loss
        else:
            return loss, s_hat, y_hat
    
    def get_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999)
        )

        scheduler = None
        
        optimizer_adv = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_gan,
            betas=(0.5, 0.999)
        )

        scheduler_adv = None
        
        return {
            "optimizers": {
                "discriminator": optimizer_adv,
                "reconstruction": optimizer,
                "generator": optimizer,
                "hsic": optimizer,
            },
            "schedulers": {
                "discriminator": scheduler_adv,
                "reconstruction": scheduler,
                "generator": scheduler,
                "hsic": scheduler,
            }
        }


class StrWAE_stack_joint(WAE_GAN):
    """
        Task: Invariant representations
        Data: Extended YaleB
            - X = image
            - Y = identity of person; fully observed
            - S = light condition; fully observed
        Decoder Structure: Stacked model, Modeling joint probability distribution of (X,Y, S).
            - (Z2, Y) -> (Z1, S) -> X
            - Y and S are independent
            - Z2 independent of (Y, S)
            - Z1 independent of S but correlated with Y (desired representation)
        
        Embedding Y with a pretrained encoder
    """
    def __init__(self,
                 in_channels: int,
                 base_channels: int,
                 hidden_size: int,
                 disc_size: int,
                 label_dim: int,
                 latent_dim: int,
                 attr_dim: int,
                 linear_bn: bool,
                 kernel_sizes: List[List[int]], # (encoder, decoder)
                 input_size: int,
                 scaling_steps: List[int], # (encoder, decoder)
                 conv_steps: List[int], # (encoder, decoder)
                 block_layers: List[int], # (encoder, decoder)
                 skip: List[bool], # (encoder, decoder)
                 fc_layers: List[int],
                 disc_layers: int,
                 learning_rate: float,
                 learning_rate_gan: float,
                 encoder: nn.Module = MLPBlock,
                 decoder: nn.Module = MLPBlock,
                 discriminator: nn.Module = MLPBlock,
                 activation: nn.Module = nn.LeakyReLU,
                 pretrained_path: str = "embedders/eyaleb_simple_jit.pt"):

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

        '''
        Encoding (= f(X, Y_hat, S)):
            1. (pretrained) encoder_y: x -> y_hat
            2. encoder: x -> z
        '''
        self.encoder = encoder(
            in_channels=in_channels,
            base_channels=base_channels//2, # 32
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            linear_bn=linear_bn,
            input_size=input_size,
            kernel_sizes=kernel_sizes[0],
            scaling_steps=scaling_steps[0],
            conv_steps=conv_steps[0],
            block_layers=block_layers[0],
            skip=skip[0],
            fc_layers=fc_layers[0],
            activation=nn.ReLU
        ) # x -> z2; cnn

        '''
        Decoding (= g_1(S, g_2(Y, Z2)))
            1. decoder_z1: (Z2, Y) -> Z1
            2. decoder: (Z1, S) -> x_hat
        '''
        self.decoder = decoder(
            base_channels=base_channels, # 64
            out_channels=in_channels,
            hidden_size=hidden_size,
            latent_dim=label_dim + latent_dim + attr_dim,
            linear_bn=linear_bn,
            output_size=input_size,
            kernel_sizes=kernel_sizes[1],
            # conv_layers=conv_layers-1,
            scaling_steps=scaling_steps[1],
            conv_steps=conv_steps[1],
            block_layers=block_layers[1],
            skip=skip[1],
            fc_layers=fc_layers[1],
            activation=activation
        ) # (z1, s) -> x; cnn

        self.decoder_z1 = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation(inplace=True),

            nn.Linear(hidden_size, latent_dim + label_dim),
        ) # (z2, y) -> z1; MLP

        self.discriminator = discriminator(
            input_dim=latent_dim,
            hidden_dim=disc_size,
            layers=disc_layers,
            activation=nn.ReLU,
        )

        self.encoder_y = torch.jit.load(f"./checkpoints/{pretrained_path}")
        self.encoder_y.eval() # fix pretrained Y-embedder

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        with torch.no_grad():
            y_hat, _ = self.encoder_y.encode(x)
        z = self.encoder(x)
        z1 = self.decoder_z1(torch.cat([z, y_hat], dim=1))
        x_hat = self.decoder(torch.cat([z1, s], dim=1))
        return x_hat
    
    def encode(self, x: torch.Tensor):
        with torch.no_grad():
            y_hat, _ = self.encoder_y.encode(x)
        z = self.encoder(x)
        return z, nn.Identity(), y_hat

    def decode(self, z: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        z1 = self.decoder_z1(torch.cat([z, y], dim=1))
        return self.decoder(torch.cat([z1, s], dim=1))
    
    def first_operation(self, x: torch.Tensor):
        with torch.no_grad():
            y_hat, _ = self.encoder_y.encode(x)
        z = self.encoder(x)
        z1 = self.decoder_z1(torch.cat([z, y_hat], dim=1))
        return z1, z, y_hat

    def get_losses(self,
                   ingredient,
                   x: torch.Tensor,
                   s: torch.Tensor,
                   mode: str = "",
                   valid: bool = False):

        z1, z, y_hat = ingredient
        
        if mode == "reconstruction":
            x_hat = self.decoder(torch.cat([z1, s], dim=1))

            loss = torch.mean(
                torch.sum((x - x_hat).pow(2), dim=[1,2,3])
            )
        
        elif mode == "generator": # prior matching
            q_z = self.discriminator(z)

            loss = F.binary_cross_entropy_with_logits(
                q_z, torch.ones_like(q_z)
            )

        elif mode == "discriminator": # adversarial training
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

        elif mode == "hsic":
            loss = dhsic_penalty([z, y_hat, s])

        elif mode == "hsic_attr":
            loss = hsic_penalty(z1, s)

        else:
            raise ValueError("Invalid mode")
        
        if not valid:
            return loss
        else:
            return loss, s, y_hat

    def get_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) \
            + list(self.decoder.parameters()) \
            + list(self.decoder_z1.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999)
        )

        optimizer_adv = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_gan,
            betas=(0.5, 0.999)
        )

        return {
            "optimizers": {
                "discriminator": optimizer_adv,
                "generator": optimizer,
                "reconstruction": optimizer,
                "hsic": optimizer,
                "hsic_attr": optimizer
            },
            "schedulers": {
                "discriminator": None,
                "generator": None,
                "reconstruction": None,
                "hsic": None,
                "hsic_attr": None
            }
        }


class StrWAE_stack_cond(WAE_GAN):
    """
        Task: fair representations (independent of S but correlated with Y); fully observed
        Data: Adult Income Dataset
            - X: 113 binary attributes (preprocessing)
            - Y: income (>50K)
            - S: gender (binary); correlated with Y!
        Decoder Structure: Stacked model, Modeling conditional distribution of (X,Y) given S.
            - (Z2, Y) -> (Z1, S) -> X
            - Y and S are correlated
            - Z2 independent of (Y, S)
            - Z1 independent of S but correlated with Y (desired representation)
    """
    def __init__(self,
                 hidden_size: int,
                 disc_size: int,
                 label_dim: int,
                 latent_dim: int,
                 linear_bn: bool,
                 input_size: int,
                 fc_layers: int,
                 disc_layers: int,
                 learning_rate: float,
                 learning_rate_gan: float,
                 attr_dim: int = 1,
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
        '''
        Encoding (= f(X,Y,S)):
            1. encoder: x -> h
            2. encoder_z2: (h, s, y) -> z2
        '''
        self.encoder = encoder(
            input_dim=input_size,
            hidden_dim=hidden_size,
            output_dim=latent_dim,
            layers=fc_layers,
            linear_bn=linear_bn,
            last_activation=True,
            activation=activation
        ) # x -> h; MLP
    
        self.encoder_z = nn.Sequential(
            nn.Linear(latent_dim + label_dim + attr_dim, hidden_size),
            nn.BatchNorm1d(hidden_size) if linear_bn else nn.Identity(),
            activation(inplace=True),
            
            nn.Linear(hidden_size, latent_dim)
        ) # (h, s, y) -> z2; MLP

        '''
        Decoding (= g_1(S, g_2(Y, Z2)))
            1. decoder_z1: (Z2, Y) -> Z1
            2. decoder: (Z1, S) -> x_hat
        '''
        self.decoder_z1 = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_size),
            nn.BatchNorm1d(hidden_size) if linear_bn else nn.Identity(),
            activation(inplace=True),
            
            nn.Linear(hidden_size, latent_dim)
        )
        
        self.decoder = decoder(
            input_dim=latent_dim + attr_dim,
            hidden_dim=hidden_size,
            output_dim=input_size,
            layers=fc_layers,
            linear_bn=linear_bn,
            last_activation=False,
            activation=activation
        )
        
        self.discriminator0 = discriminator(
            input_dim=latent_dim,
            hidden_dim=disc_size, # 4*latent_dim
            layers=disc_layers,
            activation=nn.ReLU,
        ) # prior matching when S=0
        
        self.discriminator1 = discriminator(
            input_dim=latent_dim,
            hidden_dim=disc_size, # 4*latent_dim
            layers=disc_layers,
            activation=nn.ReLU,
        ) # prior matching when S=1

    def forward(self, x: torch.Tensor, s: torch.Tensor, y: torch.Tensor):
        x_embed = self.encoder(x)
        z = self.encoder_z(torch.cat([x_embed, s, y], dim=1))
        
        z1 = self.decoder_z1(torch.cat([z, y], dim=1))
        x_hat = self.decoder(torch.cat([z1, s], dim=1))
        return x_hat
    
    def encode(self, 
               x: torch.Tensor, 
               s: torch.Tensor, 
               y: torch.Tensor):
        x_embed = self.encoder(x) # x -> h
        z = self.encoder_z(torch.cat([x_embed, s, y], dim=1)) # (h, s, y) -> z2
        return z

    def decode(self, z: torch.Tensor, s: torch.Tensor, y: torch.Tensor):
        z1 = self.decoder_z1(torch.cat([z, y], dim=1))
        return self.decoder(torch.cat([z1, s], dim=1))
    
    def first_operation(self,
                        x: torch.Tensor,
                        s: torch.Tensor,
                        y: torch.Tensor):
        z = self.encode(x, s, y)
        z1 = self.decoder_z1(torch.cat([z, y], dim=1))
        return z1, z
    
    def get_losses(self,
                   ingredient,
                   x: torch.Tensor,
                   s: torch.Tensor,
                   y: torch.Tensor,
                   mode: str = "",
                   valid: bool = False):

        z1, z = ingredient
        p_s = s.mean()

        if mode == "reconstruction":
            x_hat = self.decoder(torch.cat([z1, s], dim=1))
            
            # BCE loss
            loss = torch.mean(
                torch.sum(
                    F.binary_cross_entropy_with_logits(x_hat, x, reduction='none'),
                    dim=list(range(1, len(x.size())))
                )
            )
        
        elif mode == "generator":
            loss_s0, loss_s1 = 0.0, 0.0
            
            if p_s < 1.0: # There is a case of s=0 in the batch
                q_z0 = self.discriminator0(z[s[:,0]==0, :])
                loss_s0 = F.binary_cross_entropy_with_logits(
                    q_z0, torch.ones_like(q_z0)
                )
            
            if p_s > 0.0: # There is a case of s=1 in the batch
                q_z1 = self.discriminator1(z[s[:,0]==1, :])
                loss_s1 = F.binary_cross_entropy_with_logits(
                    q_z1, torch.ones_like(q_z1)
                )
            loss = p_s * loss_s1 + (1.-p_s) * loss_s0

        elif mode == "discriminator":
            z_prior = torch.randn_like(z)
            loss_s0, loss_s1 = 0.0, 0.0
            
            if p_s < 1.0:
                p_z0 = self.discriminator0(z_prior[s[:,0]==0., :])
                q_z0 = self.discriminator0(z[s[:,0]==0, :].detach())
                loss_s0 = (
                    F.binary_cross_entropy_with_logits(
                        p_z0, torch.ones_like(p_z0)
                    )
                    + F.binary_cross_entropy_with_logits(
                        q_z0, torch.zeros_like(q_z0)
                    )
                )
            
            if p_s > 0.0:
                p_z1 = self.discriminator1(z_prior[s[:,0]==1, :])
                q_z1 = self.discriminator1(z[s[:,0]==1, :].detach())
                loss_s1 = (
                    F.binary_cross_entropy_with_logits(
                        p_z1, torch.ones_like(p_z1)
                    )
                    + F.binary_cross_entropy_with_logits(
                        q_z1, torch.zeros_like(q_z1)
                    )
                )
            
            loss = p_s * loss_s1 + (1.-p_s) * loss_s0
        
        elif mode == "hsic":
            loss_s0, loss_s1 = 0.0, 0.0
            
            if p_s < 1.0:
                loss_s0 = hsic_penalty(z[s[:,0]==0,:], y[s[:,0]==0,:])
            
            if p_s > 0.0:
                loss_s1 = hsic_penalty(z[s[:,0]==1,:], y[s[:,0]==1,:])
            loss = p_s * loss_s1 + (1.-p_s) * loss_s0

        elif mode == "hsic_attr":
            loss = hsic_penalty(z1, s)

        else:
            raise ValueError("Invalid mode")
        
        if not valid:
            return loss
        else:
            return loss, s, y

    def get_optimizers(self):
        optimizer = optim.RAdam(
            list(self.encoder.parameters()) \
            + list(self.encoder_z.parameters()) \
            + list(self.decoder_z1.parameters()) \
            + list(self.decoder.parameters()),
            betas=(0.5, 0.999)
        )

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        
        optimizer_adv = optim.RAdam(
            list(self.discriminator0.parameters()) \
            + list(self.discriminator1.parameters()),
            lr=self.lr_gan,
            betas=(0.5, 0.999)
        )

        scheduler_adv = optim.lr_scheduler.StepLR(optimizer_adv, step_size=30, gamma=0.9)
        
        return {
            "optimizers": {
                "discriminator": optimizer_adv,
                "generator": optimizer,
                "reconstruction": optimizer,
                "hsic": optimizer,
                "hsic_attr": optimizer
            },
            "schedulers": {
                "discriminator": scheduler_adv,
                "generator": None,
                "reconstruction": scheduler,
                "hsic": None,
                "hsic_attr": None
            }
        }
