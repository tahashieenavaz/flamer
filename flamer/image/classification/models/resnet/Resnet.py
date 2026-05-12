import torch


class Resnet(torch.nn.Module):
    def dummy(self):
        return torch.randn(1, 3, 224, 224)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
