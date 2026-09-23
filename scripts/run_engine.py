"""
Phase 6 — prefill + decode inference engine, end-to-end.

prompt -> InferenceEngine -> ModelRunner.prefill / decode -> generated text

Also runs the Phase 3 naive ModelRunner.generate on the same prompt
and checks the token sequences are identical (use --greedy, the
default here, for a deterministic comparison).

Usage:
    PYTHONPATH="$(pwd)" python scripts/run_engine.py
    PYTHONPATH="$(pwd)" python scripts/run_engine.py --prompt "Once upon a time" --max-new-tokens 100
"""

import argparse

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.inference_engine import InferenceEngine
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

    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--repeat-ngram-size", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("V1 PREFILL + DECODE INFERENCE (Phase 6)")
    print("=" * 60)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)

    print(f"\nCheckpoint:\n    {checkpoint_path}")

    model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    sampler = Sampler(
        greedy=True,
        repetition_penalty=args.repetition_penalty,
        repeat_ngram_size=args.repeat_ngram_size,
    )

    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        device=args.device,
    )

    engine = InferenceEngine(
        runner=runner,
        tokenizer=tokenizer,
        sampler=sampler,
        device=args.device,
    )

    print(f"\nPrompt:\n    \"{args.prompt}\"")

    # Warm-up so CUDA init / kernel selection doesn't land in the timing.
    engine.generate(args.prompt, max_new_tokens=2)

    result = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    stats = result["stats"]

    print(f"\nPrefill:")
    print(f"    prompt tokens       = {stats['prompt_tokens']}")
    print(f"    KV cache length     = {stats['prefill_kv_length']}")
    print(f"    elapsed             = {stats['prefill_seconds'] * 1000:.2f} ms")

    print(f"\nDecode:")
    print(f"    generated tokens    = {stats['generated_tokens']}")
    print(f"    final KV length     = {stats['final_kv_length']}"
          f"  (last sampled token not fed back)")
    print(f"    elapsed             = {stats['decode_seconds'] * 1000:.2f} ms")

    print(f"\nGeneration:")
    print(f"    elapsed             = {stats['elapsed_seconds']:.4f} s")
    print(f"    tokens/sec          = {stats['tokens_per_second']:.2f}")

    print(f"\nGenerated text:\n    \"{result['text']}\"")

    naive = runner.generate(args.prompt, max_new_tokens=args.max_new_tokens)

    token_match = naive["token_ids"] == result["token_ids"]

    print(f"\nCorrectness:")
    print(f"    naive tokens        = {naive['stats']['generated_tokens']}")
    print(f"    cached tokens       = {stats['generated_tokens']}")
    print(f"    token match         = {token_match}")
    print(f"    naive tokens/sec    = {naive['stats']['tokens_per_second']:.2f}")

    if not token_match:
        print(f"    naive ids           = {naive['token_ids']}")
        print(f"    cached ids          = {result['token_ids']}")

    print("\n" + "=" * 60)

    if not token_match:
        print("PHASE 6 FAILED")
        print("=" * 60)
        raise RuntimeError("Engine tokens differ from naive runner tokens.")

    print("PHASE 6 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
