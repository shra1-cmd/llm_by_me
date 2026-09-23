import torch
import torch.nn as nn

from src.model.attention import GroupedQueryAttention
from src.model.rmsnorm import RMSNorm
from src.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(
            hidden_dim,
            eps=rms_norm_eps,
        )

        self.attention = GroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.ffn_norm = RMSNorm(
            hidden_dim,
            eps=rms_norm_eps,
        )

        self.ffn = SwiGLU(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
        layer_idx: int | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Attention sub-layer
        x = x + self.attention(
            self.attn_norm(x),
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        # FFN sub-layer
        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x