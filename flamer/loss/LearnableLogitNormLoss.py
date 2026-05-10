import torch


class LearnableLogitNormLoss(torch.nn.Module):
    def __init__(self, *, temperature: float = 1.0):
        super().__init__()
        self.log_temperature = torch.nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature)
        logits = torch.nn.functional.normalize(logits, p=2, dim=-1)
        logits = logits / temperature
        return torch.nn.functional.cross_entropy(logits, targets)
