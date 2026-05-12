import torch
from torchvision.models import resnet18
from typing import Type
from .....functions import replace_modules
from .....functions import module_accepts_channels


class Resnet18Encoder(torch.nn.Module):
    def __init__(self, activation: Type[torch.nn.Module] = torch.nn.ReLU):
        super().__init__()
        self.stream = resnet18()
        self.stream.fc = torch.nn.Identity()

        if activation != torch.nn.ReLU:
            if module_accepts_channels(activation):
                channels_iterator = iter(ResnetChannelRepository.18)
            else:
                channels_iterator = None

            replace_modules(
                subject=self.stream,
                search=torch.nn.ReLU,
                replace=activation,
                channels_iterator=channels_iterator
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
