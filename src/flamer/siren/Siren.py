import torch
import math
from .siren_activation import SirenActivation


class Siren(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation_coefficient: float = 30.0,
    ):
        super().__init__()
        self.activation_coefficient = activation_coefficient
        self.activation = SirenActivation(w0=activation_coefficient)
        self.alpha = torch.nn.Linear(input_dimension, hidden_dimension)
        self.beta = torch.nn.Linear(hidden_dimension, output_dimension)
        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        b1 = 1 / self.alpha.in_features
        b2 = math.sqrt(6 / self.beta.in_features) / self.activation_coefficient
        self.alpha.weight.uniform_(-b1, b1)
        self.beta.weight.uniform_(-b2, b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alpha(x)
        x = self.activation(x)
        x = self.beta(x)
        return x
