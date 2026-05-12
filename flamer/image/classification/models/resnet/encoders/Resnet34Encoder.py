import torch
from torchvision.models import resnet34
from typing import Type
from .ResnetEncoder import ResnetEncoder


class Resnet34Encoder(ResnetEncoder):
    def __init__(self, activation: Type[torch.nn.Module] = torch.nn.ReLU):
        super().__init__(builder=resnet34, activation=activation)
