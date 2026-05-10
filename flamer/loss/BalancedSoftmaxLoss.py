import torch


class BalancedSoftmaxLoss(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.register_buffer(
            "class_prior", torch.log(torch.tensor(num_classes, dtype=torch.float32))
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits + self.class_prior
        return torch.nn.functional.cross_entropy(logits, targets)
