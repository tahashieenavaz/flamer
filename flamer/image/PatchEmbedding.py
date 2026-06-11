import torch
from typing import TypeAlias, Union

NormalizationLayer: TypeAlias = Union[
    torch.nn.LayerNorm, torch.nn.RMSNorm, torch.nn.BatchNorm
]


class PatchEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embedding_dimension: int,
        normalization: bool = True,
        normalization_layer: NormalizationLayer = torch.nn.LayerNorm,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.conv = torch.nn.Conv2d(
            in_channels, embedding_dimension, patch_size, patch_size
        )
        self.normalization = normalization

        if self.normalization:
            self.normalization_layer = normalization_layer(embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).flatten(2).transpose(1, 2)

        if self.normalization:
            x = self.normalization_layer(x)

        return x
