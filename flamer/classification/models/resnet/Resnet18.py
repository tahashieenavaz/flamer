import torch
from .encoders import Resnet18Encoder


class Resnet18(torch.nn.Module):
    def __init__(self, num_classes: int, activation: torch.nn.ReLU):
        self.encoder = Resnet18Encoder(activation=activation)
        self.fc = 
