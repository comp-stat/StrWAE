import torch
from scipy.special import gamma
from typing import List

"""
    Refers to original Tensorflow implementation: https://github.com/romain-lopez/HCV
    Refers to original implementations
        - https://github.com/kacperChwialkowski/HSIC
        - https://cran.r-project.org/web/packages/dHSIC/index.html
"""

def bandwidth(d):
    gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
    return 1. / (2. * gz**2)

def knl(x, y, gam=1.):
    dist_table = (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)
    return (-gam * dist_table).exp().transpose(0,1) 

def hsic_penalty(x: torch.Tensor, y: torch.Tensor):

    dx = x.shape[1]
    dy = y.shape[1]

    xx = knl(x, x, gam=bandwidth(dx))
    yy = knl(y, y, gam=bandwidth(dy))

    res = ((xx*yy).mean()) + (xx.mean()) * (yy.mean())
    res -= 2*((xx.mean(dim=1))*(yy.mean(dim=1))).mean()
    return res.clamp(min = 1e-16)


# dHSIC
# list_variables has to be a list of tensorflow tensors
def dhsic_penalty(list_variables: List[torch.Tensor]):

    for i, z_j in enumerate(list_variables):
        k_j = knl(z_j, z_j, gam=bandwidth(z_j.shape[1]))
        if i == 0:
            term1 = k_j
            term2 = k_j.mean()
            term3 = k_j.mean(dim=0)
        else:
            term1 = term1 * k_j
            term2 = term2 * k_j.mean()
            term3 = term3 * k_j.mean(dim=0)

    return term1.mean() + term2 - 2 * term3.mean()