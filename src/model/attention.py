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
        kv_cache=None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """
        kv_cache:
            Phase 4 KVCache, or None (Phase 3 behavior, unchanged).

        layer_idx:
            which layer's slot in kv_cache this call reads/writes.
            Required whenever kv_cache is not None.
        """

        B, T, C = x.shape

        past_seq_len = (
            kv_cache.get_seq_length(layer_idx)
            if kv_cache is not None
            else 0
        )

        # --------------------------------------------------
        # QKV projections (only for the new tokens in x)
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
        #
        # Rotate only the new positions, at their true absolute
        # offset (past_seq_len). Cached k already has RoPE baked in
        # from when it was computed, so it must not be rotated again.

        q, k = self.rope(q, k, start_pos=past_seq_len)

        # --------------------------------------------------
        # KV cache
        # --------------------------------------------------
        #
        # Append this step's new k/v to whatever was cached, and use
        # the full (past + new) k/v for attention. With no cache,
        # k/v are just the new tokens' k/v, same as Phase 3.

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

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
        #
        # With no past (past_seq_len == 0), query and key lengths are
        # equal and `is_causal=True` is the standard causal mask.
        #
        # With a cache offset, query i (local index, 0-based) must be
        # allowed to see every key up to and including its own
        # absolute position (past_seq_len + i). `is_causal=True` does
        # not reliably produce this for a non-square (T_q != T_k)
        # attention, so the mask is built explicitly instead.

        if past_seq_len == 0:
            attn_mask = None
            is_causal = True
        else:
            total_len = past_seq_len + T

            query_positions = torch.arange(
                past_seq_len,
                total_len,
                device=x.device,
            ).unsqueeze(1)

            key_positions = torch.arange(
                total_len,
                device=x.device,
            ).unsqueeze(0)

            attn_mask = key_positions <= query_positions
            is_causal = False

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=is_causal,
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