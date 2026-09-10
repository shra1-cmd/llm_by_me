import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim % num_q_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_q_heads"
            )

        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                "num_q_heads must be divisible by num_kv_heads"
            )

        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = (
            hidden_dim // num_q_heads
        )

        self.num_groups = (
            num_q_heads // num_kv_heads
        )

        self.q_proj = nn.Linear(
            hidden_dim,
            num_q_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            hidden_dim,
            num_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            hidden_dim,
            num_kv_heads * self.head_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        B, T, C = x.shape

        # --------------------------------------------------
        # QKV projections
        # --------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # --------------------------------------------------
        # Split into heads
        # --------------------------------------------------

        q = q.view(
            B,
            T,
            self.num_q_heads,
            self.head_dim,
        )

        k = k.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        )

        v = v.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        )

        # [B, T, H, D] -> [B, H, T, D]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # --------------------------------------------------
        # RoPE
        # --------------------------------------------------

        q, k = self.rope(q, k)

        # --------------------------------------------------
        # GQA
        # --------------------------------------------------
        #
        # K/V:
        #
        # [B, 2, T, 64]
        #
        # becomes:
        #
        # [B, 8, T, 64]
        #
        # Each KV head is shared by 4 Q heads.
        # --------------------------------------------------

        if self.num_groups > 1:
            k = k.repeat_interleave(
                self.num_groups,
                dim=1,
            )

            v = v.repeat_interleave(
                self.num_groups,
                dim=1,
            )

        # --------------------------------------------------
        # Causal self-attention
        # --------------------------------------------------

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        # [B, H, T, D]
        # -> [B, T, H, D]

        attn_output = attn_output.transpose(
            1,
            2,
        )

        # [B, T, H, D]
        # -> [B, T, hidden_dim]

        attn_output = attn_output.contiguous().view(
            B,
            T,
            C,
        )

        return self.out_proj(attn_output)