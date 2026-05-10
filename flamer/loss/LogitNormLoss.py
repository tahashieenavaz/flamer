import torch


class LogitNormLoss(torch.nn.Module):
    def __init__(self, *, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.nn.functional.normalize(logits, p=2, dim=-1) / self.temperature
        return torch.nn.functional.cross_entropy(logits, targets)
