"""
V1 decoder-only language model.

    input_ids -> token embedding -> N x TransformerBlock (GQA attention
    + SwiGLU MLP, pre-RMSNorm, RoPE) -> final RMSNorm
    -> tied LM head (hidden @ embedding.T) -> logits

Optional inputs added by later phases: kv_cache (Phase 4),
position_ids / attention_mask for padded batches (Phase 9).

Phase 13: forward is split into profiler regions "embedding",
"layer_<i>", "rmsnorm" (final norm), "lm_head" and "loss" (no-ops
unless profiling is enabled, see src/model/profiling.py).

Phase 15: with fast_paths last_token_logits on, a forward that has no
targets and no padding (attention_mask / position_ids) runs the final
norm + LM head on the last position only and returns logits
[B, 1, vocab]. Prefill/decode only ever read that position; skipping
the rest removes (T-1)/T of the LM-head matmul (the biggest one).
"""

import torch
import torch.nn as nn

from configs.v1 import ModelConfig
from src.model import fast_paths
from src.model.block import TransformerBlock
from src.model.profiling import region
from src.model.rmsnorm import RMSNorm


class V1LanguageModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_dim,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_q_heads=config.num_q_heads,
                    num_kv_heads=config.num_kv_heads,
                    ffn_dim=config.ffn_dim,
                    max_seq_len=config.max_seq_len,
                    rms_norm_eps=config.rms_norm_eps,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            config.hidden_dim,
            eps=config.rms_norm_eps,
        )

        # No separate LM-head parameter matrix.
        #
        # logits = hidden @ embedding.T
        #
        # This ties input and output embeddings.

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache=None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        input_ids:
            [B, T]
            With a kv_cache, T is just the new tokens for this step
            (e.g. T=1 during decode); past tokens live in the cache.

        targets:
            [B, T]

        kv_cache:
            Phase 4 KVCache, or None (Phase 3 behavior: every call
            recomputes attention over the full input_ids, unchanged).
            When given, it is mutated in place with this step's k/v.

        position_ids:
            optional [B, T] per-row absolute positions (Phase 9
            batched decode, where rows sit at different positions).

        attention_mask:
            optional bool mask broadcastable to [B, 1, T, T_k],
            True = may attend. Must already encode causality; replaces
            the default causal mask (Phase 9 padded batches).

        logits:
            [B, T, vocab_size]
        """

        B, T = input_ids.shape

        past_seq_len = (
            kv_cache.get_seq_length()
            if kv_cache is not None
            else 0
        )

        if past_seq_len + T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {past_seq_len + T} exceeds "
                f"maximum {self.config.max_seq_len}"
            )

        if (
            position_ids is not None
            and position_ids.max().item() >= self.config.max_seq_len
        ):
            raise ValueError(
                f"Position {position_ids.max().item()} exceeds "
                f"maximum {self.config.max_seq_len - 1}"
            )

        # --------------------------------------------------
        # Token embedding
        # --------------------------------------------------

        with region("embedding"):
            x = self.token_embedding(input_ids)

        # [B, T]
        # ->
        # [B, T, hidden_dim]

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        for layer_idx, block in enumerate(self.blocks):
            with region(f"layer_{layer_idx}"):
                x = block(
                    x,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------

        if (
            fast_paths.FLAGS.last_token_logits
            and targets is None
            and attention_mask is None
            and position_ids is None
        ):
            x = x[:, -1:, :]

        with region("rmsnorm"):
            x = self.final_norm(x)

        # --------------------------------------------------
        # Tied LM head
        # --------------------------------------------------

        with region("lm_head"):
            logits = x @ self.token_embedding.weight.T

        # [B, T, hidden]
        # ->
        # [B, T, vocab]

        loss = None

        if targets is not None:

            with region("loss"):
                loss = nn.functional.cross_entropy(
                    logits.view(
                        -1,
                        self.config.vocab_size,
                    ),
                    targets.view(-1),
                )

        return logits, loss


def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":

    config = ModelConfig()

    # Use actual tokenizer vocabulary if available.
    #
    # This keeps model vocabulary synchronized with tokenizer.
    from src.tokenizer.tokenizer import BPETokenizer

    tokenizer = BPETokenizer()

    config.vocab_size = tokenizer.vocab_size

    model = V1LanguageModel(config)

    num_params = count_parameters(model)

    print("V1 Language Model")
    print("=" * 50)

    print(f"Vocabulary : {config.vocab_size:,}")
    print(f"Hidden dim : {config.hidden_dim}")
    print(f"Layers     : {config.num_layers}")
    print(f"Q heads    : {config.num_q_heads}")
    print(f"KV heads   : {config.num_kv_heads}")
    print(f"Head dim   : {config.hidden_dim // config.num_q_heads}")
    print(f"FFN dim    : {config.ffn_dim}")

    print(f"\nParameters: {num_params:,}")
    print(f"Parameters: {num_params / 1e6:.2f}M")

    # Forward-pass test

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, config.max_seq_len),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, config.max_seq_len),
    )

    logits, loss = model(
        input_ids,
        targets,
    )

    print(f"\nInput shape : {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss        : {loss.item():.4f}")