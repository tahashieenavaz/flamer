import torch
from typing import Optional, Type
from flamer.typing import NormalizationLayer


class Stream(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        dropout: float = 0.0,
        activation: Type[torch.nn.Module] = torch.nn.GELU,
        hidden_dimension: Optional[int] = None,
        expansion: Optional[int] = None,
        normalization: bool = True,
        normalization_layer: NormalizationLayer = torch.nn.LayerNorm,
    ):
        super().__init__()
        if expansion is None and hidden_dimension is None:
            raise Exception(
                "FeedForward requires either a hidden_dimension or expansion"
            )

        self.normalization = normalization
        self.hidden_dimension = (
            hidden_dimension if hidden_dimension else input_dimension * expansion
        )
        self.activation = activation()
        self.alpha = torch.nn.Linear(input_dimension, self.hidden_dimension)
        self.beta = torch.nn.Linear(self.hidden_dimension, output_dimension)
        self.dropout = torch.nn.Dropout(dropout)

        if normalization:
            self.normalization_layer = normalization_layer(input_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization:
            x = self.normalization_layer(x)

        x = self.alpha(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.beta(x)
        x = self.dropout(x)
        return x
