"""
Phase 2 — export a trained V1 checkpoint into an HF-compatible directory.

step_9950.pt
     |
load our V1 checkpoint (Phase 0 loader)
     |
build V1Config from the checkpoint's model_config
     |
build V1ForCausalLM, load remapped state_dict
     |
compare logits: original V1LanguageModel vs HF V1ForCausalLM
     |
save_pretrained(...)

Usage:
    PYTHONPATH="$(pwd)" python scripts/export_hf.py
    PYTHONPATH="$(pwd)" python scripts/export_hf.py --checkpoint checkpoints/v1/step_9950.pt --out models/v1-hf
"""

import argparse
import shutil
from pathlib import Path

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.model.configuration_v1 import V1Config
from src.model.modeling_v1 import V1ForCausalLM
from src.tokenizer.tokenizer import BPETokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--out", type=str, default="models/v1-hf")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--atol", type=float, default=1e-5)

    return parser.parse_args()


def build_hf_model(original_model, model_config, device):
    """
    original_model.state_dict() keys look like:
        token_embedding.weight
        blocks.0.attn_norm.weight
        ...

    V1ForCausalLM wraps V1LanguageModel as `self.model`, so the same
    keys need the "model." prefix to load into the HF wrapper.
    """

    hf_config = V1Config.from_model_config(model_config)

    hf_model = V1ForCausalLM(hf_config)

    remapped_state_dict = {
        f"model.{key}": value
        for key, value in original_model.state_dict().items()
    }

    load_result = hf_model.load_state_dict(remapped_state_dict, strict=False)

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "HF wrapper state_dict does not match exactly.\n"
            f"Missing keys: {load_result.missing_keys}\n"
            f"Unexpected keys: {load_result.unexpected_keys}"
        )

    hf_model.to(device)
    hf_model.eval()

    return hf_model


def main():
    args = parse_args()

    print("=" * 60)
    print("V1 -> HUGGING FACE EXPORT")
    print("=" * 60)

    # --------------------------------------------------
    # Checkpoint selection + original model
    # --------------------------------------------------

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)

    print(f"\nCheckpoint:\n    {checkpoint_path}")

    original_model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)

    model_config = metadata["model_config"]

    print(f"\nOriginal model:\n    V1LanguageModel")
    print(f"    step = {metadata['step']}")

    # --------------------------------------------------
    # Build HF wrapper with the exact same weights
    # --------------------------------------------------

    hf_model = build_hf_model(original_model, model_config, args.device)

    print(f"\nHF wrapper:\n    V1ForCausalLM")
    print(f"    state_dict transferred with 0 missing / 0 unexpected keys")

    # --------------------------------------------------
    # Correctness comparison
    # --------------------------------------------------

    tokenizer = BPETokenizer(args.tokenizer_path)

    prompts = [
        "Hello, my name is",
        "The quick brown fox",
    ]

    print("\nCorrectness comparison (original vs HF wrapper):")

    all_match = True

    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)

        with torch.inference_mode():
            logits_original, _ = original_model(input_ids)
            logits_hf = hf_model(input_ids).logits

        match = torch.allclose(logits_original, logits_hf, atol=args.atol)

        max_diff = (logits_original - logits_hf).abs().max().item()

        all_match = all_match and match

        print(
            f"    \"{prompt}\" -> shape {list(logits_hf.shape)}, "
            f"max_diff={max_diff:.2e}, MATCH={'✓' if match else '✗'}"
        )

    if not all_match:
        raise RuntimeError("HF wrapper logits diverge from the original model.")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_model.save_pretrained(out_dir)

    # Custom architecture: ship the class definitions alongside the
    # weights so `trust_remote_code=True` can load this directory
    # with AutoConfig / AutoModelForCausalLM.
    hf_model.config.auto_map = {
        "AutoConfig": "configuration_v1.V1Config",
        "AutoModelForCausalLM": "modeling_v1.V1ForCausalLM",
    }
    hf_model.config.save_pretrained(out_dir)

    this_dir = Path(__file__).resolve().parent.parent
    shutil.copy(this_dir / "src/model/configuration_v1.py", out_dir / "configuration_v1.py")
    shutil.copy(this_dir / "src/model/modeling_v1.py", out_dir / "modeling_v1.py")
    shutil.copy(args.tokenizer_path, out_dir / "tokenizer.json")

    print(f"\nSaved HF-compatible model to:\n    {out_dir}/")
    print(f"    config.json")
    print(f"    model.safetensors")
    print(f"    configuration_v1.py")
    print(f"    modeling_v1.py")
    print(f"    tokenizer.json")

    print("\n" + "=" * 60)
    print("PHASE 2 EXPORT PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
