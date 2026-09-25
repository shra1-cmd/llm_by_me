"""
RMSNorm: x / sqrt(mean(x^2) + eps) * weight, over the last dimension.

Phase 15: with fast_paths fused_rmsnorm on, the same formula runs as
one F.rms_norm kernel instead of six elementwise/reduction kernels
(pow, mean, add, rsqrt, mul, mul). Results agree to ~1e-6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import fast_paths


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

        if fast_paths.FLAGS.fused_rmsnorm:
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

        variance = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x