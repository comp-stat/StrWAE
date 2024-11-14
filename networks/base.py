import torch
from torch import nn


class FCLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: nn.Module = nn.LeakyReLU) -> None:
        super().__init__()
        
        self.lin = nn.Linear(input_dim, output_dim)
        self.activation = activation(inplace=True)

    def forward(self, x):
        return self.activation(self.lin(x))


class MLPBlock(nn.Module):
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 32,
                 output_dim: int = 1,
                 layers: int = 2,
                 linear_bn: bool = False,
                 last_activation: bool = False,
                 activation: nn.Module = nn.LeakyReLU) -> None:
        super().__init__()
        
        if layers == 0:
            encoder_list = [nn.Identity()]
        elif layers == 1:
            encoder_list = [
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim) if linear_bn else nn.Identity(),
            ]
        else:
            encoder_list = [
                FCLayer(input_dim, hidden_dim, activation),
                nn.BatchNorm1d(hidden_dim) if linear_bn else nn.Identity()
            ]
            
            for _ in range(layers-2):
                encoder_list.append(
                    FCLayer(
                        hidden_dim, hidden_dim, activation
                    )
                )
                encoder_list.append(
                    nn.BatchNorm1d(hidden_dim) if linear_bn else nn.Identity()
                )
                
            encoder_list.append(
                nn.Linear(hidden_dim, output_dim),
            )
            
            encoder_list.append(
                nn.BatchNorm1d(output_dim) if linear_bn else nn.Identity()
            )
        
        if last_activation:
            encoder_list.append(
                activation()
            )
            
        self.model = nn.Sequential(*encoder_list)

    def forward(self, x):
        return self.model(x)
