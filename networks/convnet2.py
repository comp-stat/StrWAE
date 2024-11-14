"""
Simple ConvNet for VGGFace2, EyaleB
"""
from typing import List
import torch
from torch import nn

from .base import MLPBlock
from .convnet import ConvBlock

class ResBlock(nn.Module):
    def __init__(self, 
                 base_channels: int, 
                 kernel_size: int, 
                 bias: bool = False, 
                 activation: nn.Module = nn.LeakyReLU) -> None:
        super().__init__()
        d = base_channels
        self.activation = activation(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(d, d, kernel_size = kernel_size, padding = 'same', bias=bias),
            nn.BatchNorm2d(d),
            self.activation,

            nn.Conv2d(d, d, kernel_size = kernel_size, padding = 'same', bias=bias),
            nn.BatchNorm2d(d),
        )
        self.skip = nn.Conv2d(d, d, kernel_size = 1)

    def forward(self, x):
        return self.activation(self.skip(x) + self.conv(x))

"""
Encoder: Downscaling convolution networks + fc layers
"""
class ConvBlock2(nn.Module):
    """
        Convolutional Block: (down or up) scaling x (scaling_steps) + convolution x (conv_steps)
    """
    def __init__(self, 
                 transpose: bool = False, 
                 in_channels: int = 1, 
                 base_channels: int = 1,
                 kernel_sizes: List[int] = [3, 3],
                 scaling_steps: int = 1,
                 conv_steps: int = 1,
                 skip: bool = False,
                 bias: bool = False,
                 activation: nn.Module = nn.LeakyReLU) -> None:
        
        super().__init__()
        
        assert len(kernel_sizes) == (scaling_steps + conv_steps)
        """
        1. scaling steps
            (transpose == False): (C, H, W) -> (C x 2, H / 2, W / 2)
            (transpose == True): (C, H, W) -> (C / 2, H x 2, W x 2)
        """
        layers_list = [
            ConvBlock(
                transpose=transpose,
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=kernel_sizes[0],
                stride=2,
                bias=bias,
                activation=activation
            )
        ]
        
        channel_scale = 0.5 if transpose else 2
        for i in range(scaling_steps-1):
            layers_list.append(
                ConvBlock(
                    transpose=transpose,
                    in_channels=int(base_channels * (channel_scale**i)),
                    out_channels=int(base_channels * (channel_scale**(i+1))),
                    kernel_size=kernel_sizes[i+1],
                    stride=2,
                    bias=bias,
                    activation=activation
                )
            )
        
        out_channels = int(base_channels * channel_scale**(scaling_steps-1))
        # 2. conv steps
        Conv_module = ResBlock if skip else ConvBlock
        for i in range(conv_steps):
            layers_list.append(
                ResBlock(
                    base_channels=out_channels,
                    kernel_size=kernel_sizes[scaling_steps+i],
                    bias=bias,
                    activation=activation
                ) if skip else ConvBlock(
                    transpose=False,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[scaling_steps+i],
                    stride=1,
                    bias=bias,
                    activation=activation
                )
            )
        
        self.model = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 64,
                 hidden_size: int = 32,
                 latent_dim: int = 8,
                 linear_bn: bool = False,
                 kernel_sizes: List[int] = [5, 5],
                 input_size: int = 32,
                 scaling_steps: int = 1,
                 conv_steps: int = 1,
                 block_layers: int = 1,
                 fc_layers: int = 1,
                 skip: bool = False,
                 bias: bool = False,
                 activation: nn.Module = nn.LeakyReLU) -> None:
        
        super().__init__()
        self.activation = activation(inplace=True)
        
        num_steps = scaling_steps + conv_steps
        encoder_list = [
            ConvBlock2(
                transpose=False,
                in_channels=in_channels,
                base_channels=base_channels,
                kernel_sizes=kernel_sizes[:num_steps],
                scaling_steps=scaling_steps,
                conv_steps=conv_steps,
                skip=skip,
                bias=bias,
                activation=activation
            )
        ]
        
        block_base_channels = base_channels * (2**(scaling_steps-1))
        for i in range(block_layers-1):
            encoder_list.append(
                ConvBlock2(
                    transpose=False,
                    in_channels=block_base_channels * (2**(i * scaling_steps)),
                    base_channels=block_base_channels * (2**(i * scaling_steps + 1)),
                    kernel_sizes=kernel_sizes[((i+1)*num_steps):((i+2)*num_steps)],
                    scaling_steps=scaling_steps,
                    conv_steps=conv_steps,
                    skip=skip,
                    bias=bias,
                    activation=activation
                )
            )
        scale_layers = block_layers * scaling_steps
        enc_dim = int(
            (2**(scale_layers-2)*base_channels * (input_size//(2**(scale_layers-1)))**2)//2
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

"""
A decoder designed with the (Upscaling - convolution) structure
"""
class Decoder(nn.Module):
    def __init__(self,
                 base_channels: int = 64,
                 out_channels: int = 1,
                 hidden_size: int = 32,
                 latent_dim: int = 8,
                 linear_bn: bool = False,
                 kernel_sizes: List[int] = [5, 5],
                 output_size: int = 32,
                 scaling_steps: int = 1,
                 conv_steps: int = 1,
                 block_layers: int = 1,
                 fc_layers: int = 1,
                 skip: bool = False,
                 bias: bool = False,
                 activation: nn.Module = nn.LeakyReLU) -> None:
        
        super().__init__()

        assert fc_layers >= 1
        num_steps = scaling_steps + conv_steps
        scale_layers = block_layers * scaling_steps
        initial_size = output_size // (2**scale_layers)
        initial_channels = 2**(scale_layers) * base_channels
        
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
        ]
        decoder_list.append(
            nn.Unflatten(
                1, (initial_channels, initial_size, initial_size)
            )
        )
        
        for i in range(block_layers):
            decoder_list.append(
                ConvBlock2(
                    transpose=True,
                    in_channels=initial_channels // (2**(i * scaling_steps)),
                    base_channels=initial_channels // (2**(i * scaling_steps + 1)),
                    kernel_sizes=kernel_sizes[(i*num_steps):((i+1)*num_steps)],
                    scaling_steps=scaling_steps,
                    conv_steps=conv_steps,
                    skip=skip,
                    bias=bias,
                    activation=activation
                )
            )

        decoder_list.extend([
            nn.Conv2d(
                base_channels, out_channels,
                kernel_size=kernel_sizes[-1],
                stride=1,
                padding=((kernel_sizes[-1]+1)//2 - 1),
                bias=bias
            ),
            nn.Sigmoid()
        ])


        self.model = nn.Sequential(*decoder_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out
