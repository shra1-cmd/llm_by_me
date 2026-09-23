import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 512,
        base: float = 10_000.0,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "cos",
            freqs.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            freqs.sin(),
            persistent=False,
        )

    @staticmethod
    def rotate_half(x: torch.Tensor):
        """
        x: [..., head_dim]
        """

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        return torch.stack(
            (-x2, x1),
            dim=-1,
        ).flatten(-2)

    def apply_rotary(
        self,
        x: torch.Tensor,
        start_pos: int,
        seq_len: int,
        position_ids: torch.Tensor | None = None,
    ):
        """
        x: [B, H, T, head_dim]

        Positions used are [start_pos, start_pos + seq_len), so a
        decode step (T=1) with a KV cache rotates the new token by
        its true absolute position rather than position 0.

        position_ids:
            optional [B, T] absolute positions, one row per sequence.
            Used by batched decode (Phase 9), where every row sits at
            a different position. Overrides start_pos when given.
        """

        if position_ids is not None:
            # [B, T, D/2]
            cos = self.cos[position_ids]
            sin = self.sin[position_ids]

            cos = torch.repeat_interleave(cos, 2, dim=-1).unsqueeze(1)
            sin = torch.repeat_interleave(sin, 2, dim=-1).unsqueeze(1)

            # [B, 1, T, D]
            return (
                x * cos
                + self.rotate_half(x) * sin
            )

        cos = self.cos[start_pos:start_pos + seq_len]
        sin = self.sin[start_pos:start_pos + seq_len]

        # [T, D/2] -> [T, D]
        cos = torch.repeat_interleave(
            cos,
            2,
            dim=-1,
        )

        sin = torch.repeat_interleave(
            sin,
            2,
            dim=-1,
        )

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return (
            x * cos
            + self.rotate_half(x) * sin
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
    ):
        seq_len = q.size(-2)

        q = self.apply_rotary(
            q,
            start_pos,
            seq_len,
            position_ids=position_ids,
        )

        k = self.apply_rotary(
            k,
            start_pos,
            seq_len,
            position_ids=position_ids,
        )

        return q, k