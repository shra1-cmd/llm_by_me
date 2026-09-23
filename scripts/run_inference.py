"""
Phase 3 — naive autoregressive inference end-to-end.

prompt -> ModelRunner -> generated text

Usage:
    PYTHONPATH="$(pwd)" python scripts/run_inference.py
    PYTHONPATH="$(pwd)" python scripts/run_inference.py --prompt "Once upon a time" --max-new-tokens 100
    PYTHONPATH="$(pwd)" python scripts/run_inference.py --greedy
"""

import argparse

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.model_runner import ModelRunner
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

    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--repeat-ngram-size", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("V1 NAIVE INFERENCE (Phase 3)")
    print("=" * 60)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)

    print(f"\nCheckpoint:\n    {checkpoint_path}")

    model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    sampler = Sampler(
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        repeat_ngram_size=args.repeat_ngram_size,
    )

    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        device=args.device,
    )

    print(f"\nPrompt:\n    \"{args.prompt}\"")
    print(f"\nSampling:")
    print(f"    greedy               = {args.greedy}")
    print(f"    temperature          = {args.temperature}")
    print(f"    top_k                = {args.top_k}")
    print(f"    top_p                = {args.top_p}")
    print(f"    repetition_penalty   = {args.repetition_penalty}")
    print(f"    repeat_ngram_size    = {args.repeat_ngram_size}")

    result = runner.generate(args.prompt, max_new_tokens=args.max_new_tokens)

    print(f"\nGenerated text:\n    \"{result['text']}\"")

    stats = result["stats"]

    print(f"\nBaseline stats (naive, no KV cache):")
    print(f"    prompt tokens     = {stats['prompt_tokens']}")
    print(f"    generated tokens  = {stats['generated_tokens']}")
    print(f"    total tokens      = {stats['total_tokens']}")
    print(f"    elapsed           = {stats['elapsed_seconds']:.4f} s")
    print(f"    tokens/sec        = {stats['tokens_per_second']:.2f}")

    if args.device == "cuda" and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"    peak GPU memory   = {peak_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("PHASE 3 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
