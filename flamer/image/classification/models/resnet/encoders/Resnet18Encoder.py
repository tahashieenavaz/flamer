import torch
from torchvision.models import resnet18
from typing import Type
from .ResnetEncoder import ResnetEncoder


class Resnet18Encoder(ResnetEncoder):
    def __init__(self, activation: Type[torch.nn.Module] = torch.nn.ReLU):
        super().__init__(builder=resnet18, activation=activation)
