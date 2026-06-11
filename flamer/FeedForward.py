import torch
from typing import Optional, Type


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        activation: Type[torch.nn.Module] = torch.nn.GELU,
        hidden_dimension: Optional[int] = None,
        expansion: Optional[int] = None,
    ):
        super().__init__()
        if expansion is None and hidden_dimension is None:
            raise Exception(
                "FeedForward requires either a hidden_dimension or expansion"
            )

        self.hidden_dimension = (
            hidden_dimension if hidden_dimension else input_dimension * expansion
        )
        self.activation = activation()
        self.alpha = torch.nn.Linear(input_dimension, self.hidden_dimension)
        self.beta = torch.nn.Linear(self.hidden_dimension, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alpha(x)
        x = self.activation(x)
        x = self.beta(x)
        return x
