import torch
from .Stream import Stream


class SoftMaskedStream(Stream):
    def forward(self, x: torch.Tensor, soft_mask: torch.Tensor) -> torch.Tensor:
        if self.normalization:
            x = self.normalization_layer(x)
        x = self.alpha(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.beta(x)
        x = self.dropout(x)
        x = x * soft_mask
        return x
