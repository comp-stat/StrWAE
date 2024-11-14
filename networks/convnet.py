"""
Simple ConvNet for MNIST, SVHN
"""
from typing import List
import torch
from torch import nn
from .base import MLPBlock


class ConvBlock(nn.Module):
    """
    Convolution block: Convolution layer + batch_norm + activation
    
    if stride == 2:
        Downscaling (transpose=False) or Upscaling (transpose=True) by two;
        (in_channels) x H x W -> (out_channels) x (H / 2) x (W / 2)
    elif stride == 1:
        output size = input size;
        (in_channels) x H x W -> (out_channels) x H x W
    """
    def __init__(self,
                 transpose: bool = False,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 kernel_size: int = 3,
                 stride: int = 2,
                 bias: bool = False,
                 activation: nn.Module = nn.LeakyReLU) -> None:
        super().__init__()

        padding = (kernel_size + 1) // 2 - 1
        conv_module = nn.Conv2d if not transpose else nn.ConvTranspose2d
        dict_param = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "bias": bias
        }
        if transpose:
            dict_param.update({"output_padding": (kernel_size % 2)})
        
        self.conv = conv_module(**dict_param)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        return self.activation(out)

class Encoder(nn.Module):
    """
        Convolutional encoder
    """
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 32, # 32 for MNIST, 64 for SVHN
                 hidden_size: int = 32, # 32 for MNIST, 64 for SVHN
                 latent_dim: int = 10, # 10 for MNIST, 20 for SVHN
                 linear_bn: bool = False,
                 kernel_size: int = 3,
                 input_size: int = 32,
                 conv_layers: int = 4,
                 fc_layers: int = 4,
                 bias: bool = False,
                 activation: nn.Module = nn.LeakyReLU) -> None:
        
        super().__init__()
        self.activation = activation(inplace=True)

        encoder_list = [
            ConvBlock(
                transpose=False, # downscaling
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=kernel_size,
                stride=2,
                bias=bias,
                activation=activation
            )
        ]
        for i in range(conv_layers-1):
            encoder_list.append(
                ConvBlock(
                    transpose=False,
                    in_channels=(2**i)*base_channels,
                    out_channels=(2**(i+1))*base_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=bias,
                    activation=activation
                )
            )

        # fc_layers = 0: add only flatten layer; output_dim=enc_dim (for embedding networks)
        # fc_layers >= 1: add fc layers; output_dim=latent_dim
        enc_dim = int(
            (2**(conv_layers-2)*base_channels * (input_size//(2**(conv_layers-1)))**2)//2
        )
        encoder_list.extend([
            nn.Flatten(),
            MLPBlock(
                input_dim=enc_dim,
                hidden_dim=hidden_size,
                output_dim=latent_dim,
                layers=fc_layers,
                linear_bn=linear_bn,
                last_activation=False,
                activation=activation
            )
        ])

        self.model = nn.Sequential(*encoder_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out


class Decoder(nn.Module):
    """
        Convolutional decoder
    """
    def __init__(self,
                 base_channels: int = 32, # 32 for MNIST, 64 for SVHN
                 out_channels: int = 1,
                 hidden_size: int = 32, # 32 for MNIST, 64 for SVHN
                 latent_dim: int = 10, # 10 for MNIST, 20 for SVHN
                 linear_bn: bool = False,
                 kernel_size: int = 4,
                 bias: bool = False,
                 output_size: int = 32,
                 conv_layers: int = 4,
                 fc_layers: int = 4,
                 activation: nn.Module = nn.LeakyReLU) -> None:
        
        super().__init__()

        initial_size = output_size // (2**conv_layers)
        initial_channels = 2**(conv_layers-1) * base_channels
        decoder_list = [
            MLPBlock(
                input_dim=latent_dim,
                hidden_dim=hidden_size,
                output_dim=initial_channels * initial_size**2,
                layers=fc_layers,
                linear_bn=linear_bn,
                last_activation=False,
                activation=activation
            ),
            nn.Unflatten(
                1, (initial_channels, initial_size, initial_size)
            )
        ]
        
        for i in range(conv_layers-1):
            decoder_list.append(
                ConvBlock(
                    transpose=True,
                    in_channels=initial_channels//(2**i),
                    out_channels=initial_channels//(2**(i+1)),
                    kernel_size=kernel_size,
                    stride=2,
                    bias=bias,
                    activation=activation,
                )
            )

        # last convolutution
        decoder_list.extend([
            nn.ConvTranspose2d(
                base_channels, out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=((kernel_size+1)//2 - 1),
                output_padding=kernel_size % 2,
                bias=bias
            ),
            nn.Sigmoid()
        ])

        self.model = nn.Sequential(*decoder_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out
