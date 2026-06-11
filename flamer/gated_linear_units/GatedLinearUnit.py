import torch
from typing import Type


class GatedLinearUnit(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        activation: Type[torch.nn.Module],
    ):
        super().__init__()
        self.w = torch.nn.Linear(input_dimension, output_dimension * 2)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.w(x).chunk(2, dim=-1)
        return self.activation(gate) * x
