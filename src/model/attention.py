"""
Grouped-query self-attention with RoPE and an optional KV cache.

    x -> q/k/v projections -> split heads -> RoPE (new positions only)
      -> append new k/v to the KV cache (Phase 4) -> repeat k/v heads
         for GQA -> scaled_dot_product_attention (causal, or an
         explicit mask for cached / padded batches, Phase 4/9)
      -> merge heads -> output projection

Phase 13: each stage runs inside a named profiler region
(attention/qkv_proj, attention/rope, attention/kv_cache,
attention/gqa_repeat, attention/mask, attention/sdpa,
attention/out_proj). Regions are no-ops unless profiling is enabled
(see src/model/profiling.py).

Phase 15 fast paths (src/model/fast_paths.py, all off by default):
fused_qkv runs q/k/v as one matmul over a fused weight (built by
prepare_fast_paths), decode_no_mask skips the mask for a single
unpadded query, and sdpa_gqa lets SDPA broadcast the KV heads instead
of materialising repeat_interleave copies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import fast_paths
from src.model.profiling import region
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

        # Phase 15 fused_qkv: set by prepare_fast_paths()
        self.register_buffer("qkv_weight", None, persistent=False)

    def prepare_fast_paths(self):
        """Fuse q/k/v weights into one [q+2*kv, hidden] buffer (fast_paths.prepare)."""

        fast_paths.fuse_linears(self, "qkv_weight", [self.q_proj, self.k_proj, self.v_proj])

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
        layer_idx: int | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        kv_cache:
            Phase 4 KVCache, or None (Phase 3 behavior, unchanged).

        layer_idx:
            which layer's slot in kv_cache this call reads/writes.
            Required whenever kv_cache is not None.

        position_ids:
            optional [B, T] per-row absolute positions for RoPE
            (Phase 9 batching). None keeps the past_seq_len offset.

        attention_mask:
            optional bool mask broadcastable to [B, H, T_q, T_k],
            True = may attend. Replaces the built-in causal mask, so
            it must already include causality (Phase 9 batching,
            where padding differs per row). None keeps the Phase 4
            behavior.
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

        with region("attention/qkv_proj"):
            if fast_paths.FLAGS.fused_qkv and self.qkv_weight is not None:
                kv_dim = self.num_kv_heads * self.head_dim
                q, k, v = F.linear(x, self.qkv_weight).split(
                    [self.hidden_dim, kv_dim, kv_dim],
                    dim=-1,
                )
            else:
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

        with region("attention/rope"):
            q, k = self.rope(
                q,
                k,
                start_pos=past_seq_len,
                position_ids=position_ids,
            )

        # --------------------------------------------------
        # KV cache
        # --------------------------------------------------
        #
        # Append this step's new k/v to whatever was cached, and use
        # the full (past + new) k/v for attention. With no cache,
        # k/v are just the new tokens' k/v, same as Phase 3.

        if kv_cache is not None:
            with region("attention/kv_cache"):
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

        use_sdpa_gqa = fast_paths.FLAGS.sdpa_gqa and self.num_groups > 1

        if self.num_groups > 1 and not use_sdpa_gqa:
            with region("attention/gqa_repeat"):
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

        with region("attention/mask"):
            if attention_mask is not None:
                attn_mask = attention_mask
                is_causal = False
            elif past_seq_len == 0:
                attn_mask = None
                is_causal = True
            elif T == 1 and fast_paths.FLAGS.decode_no_mask:
                # One new query at the last position: every key is at
                # or before it, so the causal mask would be all True.
                attn_mask = None
                is_causal = False
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

        with region("attention/sdpa"):
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
                enable_gqa=use_sdpa_gqa,
            )

        # [B, H, T, D]
        # -> [B, T, H, D]

        attn_output = attn_output.transpose(
            1,
            2,
        )

        # [B, T, H, D]
        # -> [B, T, hidden_dim]

        with region("attention/out_proj"):
            attn_output = attn_output.contiguous().view(
                B,
                T,
                C,
            )

            return self.out_proj(attn_output)