import torch
from typing import Tuple

class Reshape(torch.nn.Module):
  def __init__(self, target_shape: Tuple[int]):
    super().__init__()
    self.target_shape = target_shape

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.reshape(self.target_shape)
