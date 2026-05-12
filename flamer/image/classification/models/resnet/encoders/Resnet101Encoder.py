import torch
from torchvision.models import resnet101
from typing import Type
from .ResnetEncoder import ResnetEncoder


class Resnet101Encoder(ResnetEncoder):
    def __init__(self, activation: Type[torch.nn.Module] = torch.nn.ReLU):
        super().__init__(builder=resnet101, activation=activation)
