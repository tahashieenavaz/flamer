import torch

class PrintStd(torch.nn.Module):
    def __init__(self, flush: bool = False, dim: int = -1):
        super().__init__()
        self.flush = flush
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.std(dim=self.dim), flush=self.flush)
        return x
