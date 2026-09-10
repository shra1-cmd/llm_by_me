import torch

from configs.v1 import ModelConfig
from src.model.model import V1LanguageModel, count_parameters
from src.tokenizer.tokenizer import BPETokenizer


def count_module_parameters(module):
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )


def check_nan_inf(tensor, name):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )


def main():

    print("=" * 60)
    print("V1 MODEL INSPECTION")
    print("=" * 60)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    tokenizer = BPETokenizer()

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
    )

    print("\nConfiguration")
    print("-" * 60)
    print(f"Vocabulary size : {config.vocab_size:,}")
    print(f"Context length  : {config.max_seq_len}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Layers          : {config.num_layers}")
    print(f"Q heads         : {config.num_q_heads}")
    print(f"KV heads        : {config.num_kv_heads}")
    print(
        f"Head dimension  : "
        f"{config.hidden_dim // config.num_q_heads}"
    )
    print(f"FFN dimension   : {config.ffn_dim}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = V1LanguageModel(config)

    total_params = count_parameters(model)

    print("\nParameter Breakdown")
    print("-" * 60)

    embedding_params = count_module_parameters(
        model.token_embedding
    )

    block_params = [
        count_module_parameters(block)
        for block in model.blocks
    ]

    final_norm_params = count_module_parameters(
        model.final_norm
    )

    print(
        f"Token embedding : "
        f"{embedding_params:,}"
    )

    for i, params in enumerate(block_params):
        print(
            f"Transformer {i + 1:2d} : "
            f"{params:,}"
        )

    print(
        f"Final RMSNorm   : "
        f"{final_norm_params:,}"
    )

    print("-" * 60)

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Total parameters: "
        f"{total_params / 1e6:.3f}M"
    )

    # --------------------------------------------------
    # Parameter percentages
    # --------------------------------------------------

    print("\nParameter Percentages")
    print("-" * 60)

    print(
        f"Embedding: "
        f"{100 * embedding_params / total_params:.2f}%"
    )

    total_block_params = sum(block_params)

    print(
        f"Transformer blocks: "
        f"{100 * total_block_params / total_params:.2f}%"
    )

    print(
        f"Final norm: "
        f"{100 * final_norm_params / total_params:.2f}%"
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    print("\nForward Pass")
    print("-" * 60)

    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
    )

    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
    )

    print(f"Input shape     : {input_ids.shape}")
    print(f"Target shape    : {targets.shape}")

    logits, loss = model(
        input_ids,
        targets,
    )

    print(f"Logits shape    : {logits.shape}")
    print(f"Loss            : {loss.item():.6f}")

    check_nan_inf(
        logits,
        "Logits",
    )

    check_nan_inf(
        loss,
        "Loss",
    )

    # --------------------------------------------------
    # Backward pass
    # --------------------------------------------------

    print("\nBackward Pass")
    print("-" * 60)

    model.zero_grad()

    loss.backward()

    print("Backward pass completed.")

    # --------------------------------------------------
    # Gradient inspection
    # --------------------------------------------------

    parameters_with_grad = 0
    parameters_without_grad = 0

    total_gradient_norm_sq = 0.0

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            parameters_without_grad += 1

            print(
                f"WARNING: no gradient -> {name}"
            )

            continue

        parameters_with_grad += 1

        check_nan_inf(
            parameter.grad,
            f"Gradient: {name}",
        )

        gradient_norm = (
            parameter.grad.detach()
            .norm()
            .item()
        )

        total_gradient_norm_sq += (
            gradient_norm ** 2
        )

    total_gradient_norm = (
        total_gradient_norm_sq ** 0.5
    )

    print(
        f"Parameters with gradients    : "
        f"{parameters_with_grad}"
    )

    print(
        f"Parameters without gradients : "
        f"{parameters_without_grad}"
    )

    print(
        f"Total gradient norm          : "
        f"{total_gradient_norm:.6f}"
    )

    print("\n" + "=" * 60)
    print("MODEL SANITY CHECK PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()