"""
Phase 0 — Checkpoint & Model Sanity Check.

.pt -> model -> CUDA -> logits

Usage:
    PYTHONPATH="$(pwd)" python scripts/check_inference.py
    PYTHONPATH="$(pwd)" python scripts/check_inference.py --checkpoint checkpoints/v1/step_9950.pt
    PYTHONPATH="$(pwd)" python scripts/check_inference.py --prompt "Hello, my name is"
"""

import argparse

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.tokenizer.tokenizer import BPETokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint. Defaults to the latest checkpoint in --checkpoint-dir.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/v1",
        help="Directory to search for the latest checkpoint when --checkpoint is not given.",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer/tokenizer.json",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("V1 INFERENCE CHECK")
    print("=" * 60)

    # --------------------------------------------------
    # Checkpoint selection
    # --------------------------------------------------

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        print(f"\nNo --checkpoint given. Selected latest: {checkpoint_path}")

    print(f"\nCheckpoint:\n    {checkpoint_path}")

    # --------------------------------------------------
    # Load checkpoint + reconstruct model
    # --------------------------------------------------

    model, metadata = load_inference_checkpoint(
        checkpoint_path,
        device=args.device,
    )

    num_params = sum(p.numel() for p in model.parameters())

    print(f"\nModel:\n    V1LanguageModel")
    print(f"\nStep:\n    {metadata['step']}")
    print(f"\nTraining loss (at save time):\n    {metadata['loss']}")
    print(f"\nParameters:\n    {num_params:,}")
    print(f"\nDevice:\n    {args.device}")
    print(f"\nstate_dict missing keys: {metadata['missing_keys']}")
    print(f"state_dict unexpected keys: {metadata['unexpected_keys']}")

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = BPETokenizer(args.tokenizer_path)

    print(f"\nTokenizer:\n    loaded ({args.tokenizer_path})")

    model_vocab_size = metadata["model_config"].vocab_size
    tokenizer_vocab_size = tokenizer.vocab_size

    print(f"\nVocabulary:")
    print(f"    tokenizer = {tokenizer_vocab_size}")
    print(f"    model     = {model_vocab_size}")

    vocab_match = tokenizer_vocab_size == model_vocab_size

    print(f"    MATCH {'✓' if vocab_match else '✗ MISMATCH'}")

    if not vocab_match:
        raise RuntimeError(
            "Tokenizer vocab size does not match model vocab size."
        )

    # --------------------------------------------------
    # Encode prompt
    # --------------------------------------------------

    print(f"\nPrompt:\n    \"{args.prompt}\"")

    token_ids = tokenizer.encode(args.prompt)

    print(f"\nToken IDs:\n    {token_ids}")

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=args.device,
    )

    print(f"\nInput shape:\n    {list(input_ids.shape)}")

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    with torch.inference_mode():
        logits, _ = model(input_ids)

    print(f"\nForward pass:\n    SUCCESS ✓")

    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------

    expected_shape = (1, len(token_ids), model_vocab_size)
    shape_ok = tuple(logits.shape) == expected_shape

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()

    print(f"\nLogits:")
    print(f"    shape    = {list(logits.shape)} (expected {list(expected_shape)}) {'✓' if shape_ok else '✗'}")
    print(f"    dtype    = {logits.dtype}")
    print(f"    min      = {logits.min().item():.4f}")
    print(f"    max      = {logits.max().item():.4f}")
    print(f"    mean     = {logits.mean().item():.4f}")

    print(f"\nNaN:\n    {has_nan} {'✗' if has_nan else '✓'}")
    print(f"\nInf:\n    {has_inf} {'✗' if has_inf else '✓'}")

    if not shape_ok or has_nan or has_inf:
        raise RuntimeError("Logits failed sanity checks.")

    print("\n" + "=" * 60)
    print("PHASE 0 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
