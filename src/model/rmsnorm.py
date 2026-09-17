import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., hidden_dim]

        variance = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x