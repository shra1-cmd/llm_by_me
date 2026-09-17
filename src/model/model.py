import torch
import torch.nn as nn

from configs.v1 import ModelConfig
from src.model.block import TransformerBlock
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
    ):
        """
        input_ids:
            [B, T]

        targets:
            [B, T]

        logits:
            [B, T, vocab_size]
        """

        B, T = input_ids.shape

        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"maximum {self.config.max_seq_len}"
            )

        # --------------------------------------------------
        # Token embedding
        # --------------------------------------------------

        x = self.token_embedding(input_ids)

        # [B, T]
        # ->
        # [B, T, hidden_dim]

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        for block in self.blocks:
            x = block(x)

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------

        x = self.final_norm(x)

        # --------------------------------------------------
        # Tied LM head
        # --------------------------------------------------

        logits = x @ self.token_embedding.weight.T

        # [B, T, hidden]
        # ->
        # [B, T, vocab]

        loss = None

        if targets is not None:

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