import torch


class RMSNorm1d(torch.nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, channels, 1))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
        return (x / rms) * self.weight
