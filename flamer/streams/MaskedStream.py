import torch
from .Stream import Stream


class MaskedStream(Stream):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.normalization:
            x = self.normalization_layer(x)

        x = self.alpha(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.beta(x)
        x = self.dropout(x)
        if mask.dtype == torch.bool:
            x = x.masked_fill(~mask, 0.0)
        else:
            x = x * mask
        return x
