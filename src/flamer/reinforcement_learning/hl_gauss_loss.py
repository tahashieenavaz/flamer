import math
import torch


class HLGaussLoss(torch.nn.Module):
    def __init__(
        self, min_value: float, max_value: float, num_bins: int, sigma: float = None
    ):
        super().__init__()
        assert num_bins > 1, "Number of bins must be greater than 1"

        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins

        support = torch.linspace(min_value, max_value, num_bins + 1)
        self.register_buffer("support", support, persistent=False)

        centers = (support[:-1] + support[1:]) / 2
        self.register_buffer("centers", centers, persistent=False)

        mean_bin_size = (max_value - min_value) / num_bins
        self.sigma = sigma if sigma is not None else (2.0 * mean_bin_size)
        self.sigma_sqrt2 = self.sigma * math.sqrt(2.0)

    def transform_to_probs(
        self, target: torch.Tensor, eps: float = 1e-10
    ) -> torch.Tensor:
        target = target.clamp(min=self.min_value, max=self.max_value).unsqueeze(-1)
        cdf_evals = torch.special.erf((self.support - target) / self.sigma_sqrt2)
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        z = cdf_evals[..., -1:] - cdf_evals[..., :1]
        return bin_probs / z.clamp(min=eps)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor = None, reduction: str = "mean"
    ) -> torch.Tensor:
        assert (
            logits.shape[-1] == self.num_bins
        ), f"Expected {self.num_bins} logits, got {logits.shape[-1]}"

        if target is None:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return (probs * self.centers).sum(dim=-1)

        target_probs = self.transform_to_probs(target)
        logits_flat = logits.view(-1, self.num_bins)
        target_probs_flat = target_probs.view(-1, self.num_bins)
        return torch.nn.functional.cross_entropy(
            logits_flat, target_probs_flat, reduction=reduction
        )
