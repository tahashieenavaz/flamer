import torch
from .output_shape import output_shape


def output_shape_last(module: torch.nn.Module, dummy: torch.Tensor):
    return output_shape(module, dummy)[-1]
