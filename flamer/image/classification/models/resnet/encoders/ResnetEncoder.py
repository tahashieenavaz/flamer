import torch
from typing import Type
from flamer.functions import replace_modules
from flamer.functions import module_accepts_channels
from flamer.repositories import ResidualNetworkChannelRepository


class ResnetEncoder(torch.nn.Module):
    def __init__(self, builder: function, activation: Type[torch.nn.Module]):
        super().__init__()
        self.stream = builder()
        self.stream.fc = torch.nn.Identity()

        if activation == torch.nn.ReLU:
            return

        if module_accepts_channels(activation):
            activation_channels = getattr(
                ResidualNetworkChannelRepository, builder.__name__
            )
            channels_iterator = iter(activation_channels)
        else:
            channels_iterator = None

        replace_modules(
            subject=self.stream,
            search=torch.nn.ReLU,
            replace=activation,
            channels_iterator=channels_iterator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
