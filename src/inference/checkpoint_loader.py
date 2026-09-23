"""
Phase 0: checkpoint -> model reconstruction.

checkpoint (.pt)
    -> model_config (stored inside checkpoint)
    -> V1LanguageModel(config)
    -> load_state_dict
    -> loaded model + metadata
"""

from pathlib import Path

import torch

from src.model.model import V1LanguageModel


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path:
    """
    checkpoints/v1/step_9550.pt
    checkpoints/v1/step_9700.pt
    checkpoints/v1/step_9950.pt
        ->
    checkpoints/v1/step_9950.pt   (highest step number)
    """

    checkpoint_dir = Path(checkpoint_dir)

    checkpoints = list(checkpoint_dir.glob("step_*.pt"))

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in: {checkpoint_dir}"
        )

    def step_number(path: Path) -> int:
        return int(path.stem.split("_")[1])

    checkpoints.sort(key=step_number)

    return checkpoints[-1]


def load_inference_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cuda",
):
    """
    checkpoint
        ->
    model (eval mode, on device) + metadata dict
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["model_config"]

    model = V1LanguageModel(config)

    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint state_dict does not match model exactly.\n"
            f"Missing keys: {load_result.missing_keys}\n"
            f"Unexpected keys: {load_result.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "step": checkpoint.get("step"),
        "loss": checkpoint.get("loss"),
        "model_config": config,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }

    return model, metadata
