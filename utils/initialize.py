from torch import nn

def init_params(model):
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_normal_(p)
            nn.init.trunc_normal_(p, std = 0.01, a = -0.02, b = 0.02)
        else:
            nn.init.uniform_(p, 0.1, 0.2)
    return