import torch
import math


class SinusoidalPositionalEmbeddings(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dimension = self.dimension // 2
        embeddings = math.log(10000) / (half_dimension - 1)
        embeddings = torch.exp(
            torch.arange(half_dimension, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
