"""
SwiGLU feed-forward block: down_proj(silu(gate_proj(x)) * up_proj(x)).

Phase 13: split into profiler regions mlp/gate_up_proj, mlp/act_mul
and mlp/down_proj (no-ops unless profiling is enabled, see
src/model/profiling.py).

Phase 15: with fast_paths fused_gate_up on (after prepare_fast_paths),
gate and up run as one [hidden, 2*ffn] matmul whose output is split.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import fast_paths
from src.model.profiling import region


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

        # Phase 15 fused_gate_up: set by prepare_fast_paths()
        self.register_buffer("gate_up_weight", None, persistent=False)

    def prepare_fast_paths(self):
        """Fuse gate/up weights into one [2*ffn, hidden] buffer (fast_paths.prepare)."""

        fast_paths.fuse_linears(self, "gate_up_weight", [self.gate_proj, self.up_proj])

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        with region("mlp/gate_up_proj"):
            if fast_paths.FLAGS.fused_gate_up and self.gate_up_weight is not None:
                gate, up = F.linear(x, self.gate_up_weight).chunk(2, dim=-1)
            else:
                gate = self.gate_proj(x)
                up = self.up_proj(x)

        with region("mlp/act_mul"):
            hidden = F.silu(gate) * up

        with region("mlp/down_proj"):
            return self.down_proj(hidden)