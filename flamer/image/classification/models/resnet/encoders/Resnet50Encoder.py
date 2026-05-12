import torch
from torchvision.models import resnet50
from typing import Type
from .ResnetEncoder import ResnetEncoder


class Resnet50Encoder(ResnetEncoder):
    def __init__(self, activation: Type[torch.nn.Module] = torch.nn.ReLU):
        super().__init__(builder=resnet50, activation=activation)
