import torch


class SirenActivation(torch.nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)
