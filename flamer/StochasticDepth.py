import torch


class StochasticDepth(torch.nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        assert probability <= 1.0 and probability >= 0
        self.keep_probability = 1 - probability

    def __skip_condition(self) -> bool:
        return not self.training or self.keep_probability == 1.0

    def __get_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        mask_size = (batch_size,) + (1,) * (x.ndim - 1)
        return (torch.rand(mask_size, device=x.device) < self.keep_probability).to(
            dtype=x.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.__skip_condition():
            return x

        mask = self.__get_mask(x)
        return x * mask / self.keep_probability
