import torch

class PrintShape(torch.nn.Module):
    def __init__(self, flush: bool = False):
        super().__init__()
        self.flush = flush

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape, flush=self.flush)
        return x
