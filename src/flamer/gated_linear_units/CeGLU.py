import torch
from .GatedLinearUnit import GatedLinearUnit


class CeGLU(GatedLinearUnit):
    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            activation=torch.nn.CELU,
        )
