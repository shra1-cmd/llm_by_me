from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    loss,
    config,
    training_config,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "step": step,
        "loss": loss,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "model_config": config,

        "training_config": training_config,

        # CPU RNG state.
        "torch_rng_state": torch.get_rng_state().clone(),
    }

    # CUDA RNG state for every CUDA device.
    if torch.cuda.is_available():
        checkpoint[
            "cuda_rng_state"
        ] = [
            state.clone()
            for state in torch.cuda.get_rng_state_all()
        ]

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"Checkpoint saved: {path}"
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    # --------------------------------------------------
    # CPU RNG
    # --------------------------------------------------

    if "torch_rng_state" in checkpoint:

        rng_state = checkpoint[
            "torch_rng_state"
        ]

        # torch.set_rng_state requires ByteTensor.
        rng_state = torch.as_tensor(
            rng_state,
            dtype=torch.uint8,
            device="cpu",
        )

        torch.set_rng_state(
            rng_state
        )

    # --------------------------------------------------
    # CUDA RNG
    # --------------------------------------------------

    if (
        torch.cuda.is_available()
        and "cuda_rng_state" in checkpoint
    ):

        cuda_states = []

        for state in checkpoint[
            "cuda_rng_state"
        ]:

            state = torch.as_tensor(
                state,
                dtype=torch.uint8,
                device="cpu",
            )

            cuda_states.append(state)

        torch.cuda.set_rng_state_all(
            cuda_states
        )

    return checkpoint