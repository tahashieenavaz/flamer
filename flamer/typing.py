import torch
from typing import TypeAlias, Union

NormalizationLayer: TypeAlias = Union[
    torch.nn.LayerNorm,
    torch.nn.RMSNorm,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm3d,
]
