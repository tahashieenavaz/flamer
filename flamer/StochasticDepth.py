import torch


class StochasticDepth(torch.nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        assert probability <= 1.0 and probability >= 0, "0 <= prob <= 1"
        self.keep_probability = 1 - probability

    def __intact_condition(self, x: torch.Tensor) -> bool:
        return not self.training or self.keep_probability == 1.0

    def __get_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return torch.rand(batch_size, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.__intact_condition(x):
            return x
        mask = self.__get_mask(x)
        return x * mask / self.keep_probability
