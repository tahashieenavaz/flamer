import torch


class SqueezeExcitation(torch.nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.alpha = torch.nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.beta = torch.nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)
        self.activation = torch.nn.SiLU(inplace=True)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def __get_scale(self, x: torch.Tensor) -> float:
        scale = self.pool(x)
        scale = self.alpha(scale)
        scale = self.activation(scale)
        scale = self.beta(scale)
        return torch.nn.functional.sigmoid(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.__get_scale(x)
        return scale * x
