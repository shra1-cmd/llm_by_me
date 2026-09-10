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
        seq_len: int,
    ):
        """
        x: [B, H, T, head_dim]
        """

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

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
    ):
        seq_len = q.size(-2)

        q = self.apply_rotary(
            q,
            seq_len,
        )

        k = self.apply_rotary(
            k,
            seq_len,
        )

        return q, k