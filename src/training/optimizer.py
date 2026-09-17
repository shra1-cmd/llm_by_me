import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
):
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        # Biases and normalization parameters should not
        # receive weight decay.
        if (
            parameter.ndim < 2
            or "norm" in name.lower()
        ):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )

    return optimizer