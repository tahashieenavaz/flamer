import torch
from typing import Type
from flamer.image.classification.models.resnet.encoders import Resnet18Encoder
from flamer.image.classification.models.resnet import Resnet
from flamer.functions import output_shape_last


class Resnet18(Resnet):
    def __init__(
        self, num_classes: int, activation: Type[torch.nn.Module] = torch.nn.ReLU
    ):
        self.encoder = Resnet18Encoder(activation=activation)
        self.fc = torch.nn.Linear(
            output_shape_last(self.encoder, self.dummy()), num_classes
        )
