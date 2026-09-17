import math

from torch.optim import Optimizer


def get_learning_rate(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
):
    # --------------------------------------------------
    # Linear warmup
    # --------------------------------------------------

    if step < warmup_steps:

        return learning_rate * (
            (step + 1) / warmup_steps
        )

    # --------------------------------------------------
    # Cosine decay
    # --------------------------------------------------

    progress = (
        step - warmup_steps
    ) / max(
        1,
        max_steps - warmup_steps,
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine = 0.5 * (
        1.0 + math.cos(
            math.pi * progress
        )
    )

    return (
        min_learning_rate
        + cosine
        * (
            learning_rate
            - min_learning_rate
        )
    )


def update_learning_rate(
    optimizer: Optimizer,
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
):
    lr = get_learning_rate(
        step=step,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr