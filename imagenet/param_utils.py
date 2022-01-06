"""
    Functions which help to skip bias and BatchNorm
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Iterable, Optional, Callable, Tuple
from torch import nn
BN_CLS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def get_parameters_from_cls(module, cls_):
    def get_members_fn(m):
        if isinstance(m, cls_):
            return m._parameters.items()
        else:
            return dict()

    named_parameters = module._named_members(get_members_fn=get_members_fn)
    for name, param in named_parameters:
        yield param


def get_norm_parameters(module):
    return get_parameters_from_cls(module, (nn.LayerNorm, *BN_CLS))


def get_bias_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters and 'bias' in name:
            yield param


def get_norm_bias_parameters(module):
    for param in get_norm_parameters(module):
        yield param
    for param in get_bias_parameters(module, exclude_func=get_norm_parameters):
        yield param


def get_common_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters:
            yield param
