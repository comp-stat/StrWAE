import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Residual network for the SVHN classifier
"""
class Residual(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 bias: bool = False,
                 activation: nn.Module = nn.LeakyReLU):
        super().__init__()
    
        self._block = nn.Sequential(
            activation(inplace=True),
            
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.BatchNorm2d(hidden_channels),
            activation(inplace=True),

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=1, stride=1, bias=bias
            ),
            nn.BatchNorm2d(hidden_channels)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 32,
                 layers: int = 4,
                 bias: bool = False,
                 activation: nn.Module = nn.LeakyReLU):
        
        super().__init__()

        self._activation = activation(inplace=True)

        self._num_layers = layers
        self._layers = nn.ModuleList([
            Residual(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                bias=bias,
                activation=activation
            )
                for _ in range(self._num_layers)
            ])

    def forward(self, x):
        
        for op in self._layers:
            x = op(x)
        out = self._activation(x)
        return out


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 32,
                 latent_dim: int = 10,
                 input_size: int = 32,
                 layers: int = 4,
                 bias: bool = False,
                 activation: nn.Module = nn.LeakyReLU):

        super(Encoder, self).__init__()

        self._activation = activation(inplace=True)

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=2, padding=1, bias=bias
        )
        self._bn1 = nn.BatchNorm2d(base_channels)

        self._conv_2 = nn.Conv2d(
            in_channels=base_channels,
            out_channels=base_channels*2,
            kernel_size=3,
            stride=2, padding=1, bias=bias
        )
        self._bn2 = nn.BatchNorm2d(2*base_channels)

        self._conv_3 = nn.Conv2d(
            in_channels=base_channels*2,
            out_channels=base_channels*4,
            kernel_size=3,
            stride=2, padding=1, bias=bias
        )
        self._bn3= nn.BatchNorm2d(4*base_channels)
        
        self._conv_4 = nn.Conv2d(
            in_channels=base_channels*4,
            out_channels=base_channels*8,
            kernel_size=3,
            stride=2, padding=1, bias=bias
        )
        self._bn4 = nn.BatchNorm2d(8*base_channels)

        self._residual_stack_1 = ResidualStack(
            in_channels=base_channels*2,
            hidden_channels=base_channels*2,
            layers=layers,
        )

        self._residual_stack_2= ResidualStack(
            in_channels=base_channels*8,
            hidden_channels=base_channels*8,
            layers=layers,
        )

        self.projector = nn.Linear(
            8*base_channels * ((input_size // (2**(layers))))**2,
            latent_dim
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self._conv_1(x)
        out = self._bn1(out)
        out = self._activation(out)

        out = self._conv_2(out)
        out = self._bn2(out)
        
        out = self._residual_stack_1(out)

        out = self._conv_3(out)
        out = self._bn3(out)
        out = self._activation(out)

        out = self._conv_4(out)
        out = self._bn4(out)
        
        out = self._residual_stack_2(out)
        return out

    def forward(self, x):
        out = self.encode(x)
        out = out.view(out.size(0), -1)
        out = self.projector(out)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 base_channels: int = 32,
                 out_channels: int = 1,
                 latent_dim: int = 10,
                 output_size: int = 32,
                 layers: int = 4,
                 bias: bool = False,
                 activation: nn.Module = nn.LeakyReLU):
    
        super(Decoder, self).__init__()

        self._activation = activation(inplace=True)
        
        init_size = output_size // (2**(layers))
        self._lin_1 = nn.Linear(latent_dim, 8*base_channels * init_size**2)
        self._bn1 = nn.BatchNorm1d(8*base_channels * init_size**2)

        self._unflatten = nn.Unflatten(
            1, (8*base_channels, init_size, init_size)
        )

        self._bn2 = nn.BatchNorm2d(8*base_channels)
        self._bn3 = nn.BatchNorm2d(4*base_channels)
        self._bn4 = nn.BatchNorm2d(2*base_channels)
        self._bn5 = nn.BatchNorm2d(base_channels)

        self._conv_1 = nn.Conv2d(
            in_channels=8*base_channels,
            out_channels=8*base_channels,
            kernel_size=3,
            stride=1, padding=1, bias=bias
        )

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=8*base_channels, 
            out_channels=4*base_channels,
            kernel_size=4, 
            stride=2, padding=1, bias=bias
        )
        
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=4*base_channels, 
            out_channels=2*base_channels,
            kernel_size=4, 
            stride=2, padding=1, bias=bias
        )

        self._conv_trans_3 = nn.ConvTranspose2d(
            in_channels=2*base_channels,
            out_channels=base_channels,
            kernel_size=4, 
            stride=2, padding=1, bias=bias
        )

        self._conv_trans_4 = nn.ConvTranspose2d(
            in_channels=base_channels, 
            out_channels=out_channels,
            kernel_size=4, 
            stride=2, padding=1, bias=bias
        )

        self._residual_stack_1 = ResidualStack(
            in_channels=8*base_channels,
            hidden_channels=8*base_channels,
            layers=layers
        )

        self._residual_stack_2 = ResidualStack(
            in_channels=2*base_channels,
            hidden_channels=2*base_channels,
            layers=layers
        )

    def forward(self, x):
        out = self._lin_1(x)
        out = self._bn1(out)
        out = self._activation(out)
        out = self._unflatten(out)

        out = self._conv_1(out)
        out = self._bn2(out)
        out = self._residual_stack_1(out)
        
        out = self._conv_trans_1(out)
        out = self._bn3(out)
        out = self._activation(out)

        out = self._conv_trans_2(out)
        out = self._bn4(out)
        out = self._residual_stack_2(out)
        
        out = self._conv_trans_3(out)
        out = self._bn5(out)
        out = self._activation(out)

        out = self._conv_trans_4(out)
        out = F.sigmoid(out)

        return out
