import random

import torch

from configs.v1 import ModelConfig
from configs.train import TrainingConfig

from src.data.dataloader import create_dataloader
from src.model.model import V1LanguageModel

from src.training.optimizer import build_optimizer
from src.training.scheduler import update_learning_rate

from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    # ==================================================
    # Smoke-test configuration
    # ==================================================

    model_config = ModelConfig()
    train_config = TrainingConfig()

    tokenizer_vocab_size = 15_485

    model_config.vocab_size = (
        tokenizer_vocab_size
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    set_seed(train_config.seed)

    print("=" * 60)
    print("V1 TRAINING SMOKE TEST")
    print("=" * 60)

    print(f"Device: {device}")

    # ==================================================
    # Model
    # ==================================================

    model = V1LanguageModel(
        model_config
    ).to(device)

    # ==================================================
    # Data
    # ==================================================

    train_loader = create_dataloader(
        split="train",
        seq_len=512,
        batch_size=2,
        shuffle=True,
    )

    # ==================================================
    # Optimizer
    # ==================================================

    optimizer = build_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    # ==================================================
    # Training
    # ==================================================

    model.train()

    iterator = iter(train_loader)

    losses = []

    num_steps = 10

    for step in range(num_steps):

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = 0.0

        # ----------------------------------------------
        # Gradient accumulation
        # ----------------------------------------------

        for micro_step in range(2):

            try:
                input_ids, targets = next(
                    iterator
                )

            except StopIteration:

                iterator = iter(
                    train_loader
                )

                input_ids, targets = next(
                    iterator
                )

            input_ids = input_ids.to(
                device
            )

            targets = targets.to(
                device
            )

            bf16_enabled = (
                device == "cuda"
                and torch.cuda.is_bf16_supported()
            )

            with torch.autocast(
                device_type="cuda"
                if device == "cuda"
                else "cpu",
                dtype=torch.bfloat16,
                enabled=bf16_enabled,
            ):

                _, loss = model(
                    input_ids,
                    targets,
                )

                loss_for_backward = (
                    loss / 2
                )

            loss_for_backward.backward()

            total_loss += loss.item()

        # ----------------------------------------------
        # Gradient clipping
        # ----------------------------------------------

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        # ----------------------------------------------
        # Learning rate
        # ----------------------------------------------

        lr = update_learning_rate(
            optimizer=optimizer,
            step=step,
            max_steps=num_steps,
            warmup_steps=2,
            learning_rate=3e-4,
            min_learning_rate=3e-5,
        )

        # ----------------------------------------------
        # Optimizer update
        # ----------------------------------------------

        optimizer.step()

        average_loss = total_loss / 2

        losses.append(
            average_loss
        )

        print(
            f"step={step:02d} "
            f"loss={average_loss:.4f} "
            f"lr={lr:.2e} "
            f"grad_norm={grad_norm:.4f}"
        )

    # ==================================================
    # Verify training changed parameters
    # ==================================================

    if not all(
        torch.isfinite(
            torch.tensor(loss)
        )
        for loss in losses
    ):
        raise RuntimeError(
            "Loss contains NaN or Inf."
        )

    # ==================================================
    # Save checkpoint
    # ==================================================

    checkpoint_path = (
        "checkpoints/v1/smoke_test.pt"
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=num_steps,
        loss=losses[-1],
        config=model_config,
        training_config=train_config,
    )

    # ==================================================
    # Reload checkpoint
    # ==================================================

    new_model = V1LanguageModel(
        model_config
    ).to(device)

    new_optimizer = build_optimizer(
        model=new_model,
        learning_rate=3e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
        device=device,
    )

    print()
    print(
        f"Checkpoint step: "
        f"{checkpoint['step']}"
    )

    print(
        f"Checkpoint loss: "
        f"{checkpoint['loss']:.4f}"
    )

    # ==================================================
    # Verify reloaded model
    # ==================================================

    new_model.eval()

    input_ids, targets = next(
        iter(train_loader)
    )

    input_ids = input_ids.to(device)
    targets = targets.to(device)

    with torch.no_grad():

        logits, reloaded_loss = new_model(
            input_ids,
            targets,
        )

    assert torch.isfinite(
        reloaded_loss
    )

    assert logits.shape == (
        2,
        512,
        model_config.vocab_size,
    )

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()