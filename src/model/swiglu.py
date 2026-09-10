import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_dim,
            ffn_dim,
            bias=False,
        )

        self.up_proj = nn.Linear(
            hidden_dim,
            ffn_dim,
            bias=False,
        )

        self.down_proj = nn.Linear(
            ffn_dim,
            hidden_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate = F.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        hidden = gate * up

        return self.down_proj(hidden)