import torch
from torchvision.models import resnet152
from typing import Type
from .ResnetEncoder import ResnetEncoder


class Resnet152Encoder(ResnetEncoder):
    def __init__(self, activation: Type[torch.nn.Module] = torch.nn.ReLU):
        super().__init__(builder=resnet152, activation=activation)
