"""
Phase 4 — KV cache sanity check + naive-vs-cached speed comparison.

This is a quick standalone sanity check, not the formal correctness
gate (that's Phase 5). It exercises the real checkpoint/tokenizer and
prints a rough speed comparison against the Phase 3 baseline.

Usage:
    PYTHONPATH="$(pwd)" python scripts/check_kv_cache.py
    PYTHONPATH="$(pwd)" python scripts/check_kv_cache.py --prompt "Once upon a time" --max-new-tokens 100
"""

import argparse
import time

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.kv_cache import KVCache
from src.inference.sampler import Sampler
from src.tokenizer.tokenizer import BPETokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


@torch.inference_mode()
def generate_naive(model, tokenizer, sampler, prompt, max_new_tokens, device):
    """Phase 3 baseline: recompute the full sequence every step."""

    generated = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    start = time.perf_counter()

    for _ in range(max_new_tokens):
        logits, _ = model(generated)
        next_token = sampler.sample(logits[:, -1, :], generated).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)

    elapsed = time.perf_counter() - start

    return tokenizer.decode(generated[0].tolist()), elapsed


@torch.inference_mode()
def generate_cached(model, tokenizer, sampler, prompt, max_new_tokens, device):
    """Phase 4: prefill once, then decode one new token per step."""

    prompt_ids = tokenizer.encode(prompt)

    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    cache = KVCache(num_layers=model.config.num_layers)

    start = time.perf_counter()

    logits, _ = model(generated, kv_cache=cache)
    next_token = sampler.sample(logits[:, -1, :], generated).unsqueeze(-1)
    generated = torch.cat([generated, next_token], dim=-1)

    for _ in range(max_new_tokens - 1):
        logits, _ = model(next_token, kv_cache=cache)
        next_token = sampler.sample(logits[:, -1, :], generated).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)

    elapsed = time.perf_counter() - start

    return tokenizer.decode(generated[0].tolist()), elapsed


def main():
    args = parse_args()

    print("=" * 60)
    print("PHASE 4 — KV CACHE SANITY CHECK")
    print("=" * 60)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)

    model, _ = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    sampler = Sampler(greedy=True)

    print(f"\nPrompt:\n    \"{args.prompt}\"")
    print(f"\nGenerating {args.max_new_tokens} tokens, greedy, naive vs cached...")

    naive_text, naive_time = generate_naive(
        model, tokenizer, sampler, args.prompt, args.max_new_tokens, args.device
    )
    cached_text, cached_time = generate_cached(
        model, tokenizer, sampler, args.prompt, args.max_new_tokens, args.device
    )

    print(f"\nNaive (Phase 3):")
    print(f"    time  = {naive_time:.4f} s")
    print(f"    text  = \"{naive_text}\"")

    print(f"\nCached (Phase 4):")
    print(f"    time  = {cached_time:.4f} s")
    print(f"    text  = \"{cached_text}\"")

    match = naive_text == cached_text

    print(f"\nOutput match (greedy, same seed path): {'YES' if match else 'NO'}")
    print(f"Speedup: {naive_time / cached_time:.2f}x")

    if not match:
        raise RuntimeError(
            "Naive and cached greedy generations diverged. "
            "This should not happen for greedy decoding — investigate before Phase 5."
        )

    print("\n" + "=" * 60)
    print("PHASE 4 SANITY CHECK PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
