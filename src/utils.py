from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F

from .score.both import get_inception_and_fid_score

from typing import List

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else :
        raise ValueError('Composition of empty sequence not supported.')


def extract(v : torch.Tensor, t : torch.Tensor, x_shape : List[int]) -> torch.Tensor:
    out = torch.gather(v, index=t, dim=0).float()
    return out.contiguous().view([t.shape[0]] + [1] * (len(x_shape) -1))


def ema(source : nn.Module, target : nn.Module, decay : float) -> None:
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1-decay))
    