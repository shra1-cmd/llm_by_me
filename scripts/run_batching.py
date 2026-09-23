"""
Phase 9 — static batching: correctness + sequential vs batched.

Runs the same set of requests:
    sequential  : Phase 8 path, one request at a time
    batched     : Phase 9 path, up to --batch-size requests per forward

and checks every request produces identical tokens both ways
(greedy), then reports latency, throughput and peak GPU memory.

Usage:
    PYTHONPATH="$(pwd)" python scripts/run_batching.py
    PYTHONPATH="$(pwd)" python scripts/run_batching.py --batch-sizes 1 2 4 8 --max-new-tokens 100
"""

import argparse
import time

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.inference_engine import InferenceEngine
from src.inference.model_runner import ModelRunner
from src.inference.request import SamplingParams
from src.tokenizer.tokenizer import BPETokenizer

DEFAULT_PROMPTS = [
    "Hello, my name is",
    "Once upon a time",
    "The little dog",
    "One day, a girl named Sue went to the park and",
    "Tom had a red ball",
    "The sun was shining and the birds were singing. Lily wanted to",
    "Mom said",
    "There was a big tree in the garden",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--prompts", type=str, nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def is_cuda(device):
    return str(device).startswith("cuda") and torch.cuda.is_available()


def gpu_utilization(device):
    """Instantaneous GPU util %, or None if pynvml isn't installed."""

    if not is_cuda(device):
        return None

    try:
        return torch.cuda.utilization()
    except Exception:
        return None


def run(engine, prompts, max_new_tokens, max_batch_size, device):
    """Submit all prompts, drain the queue, return (results, metrics)."""

    requests = [
        engine.submit(engine.create_request(p, max_new_tokens=max_new_tokens))
        for p in prompts
    ]

    if is_cuda(device):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    results = engine.run_until_complete(max_batch_size=max_batch_size)
    util = gpu_utilization(device)

    if is_cuda(device):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    assert [r["request"] for r in results] == requests

    generated = sum(r["request"].num_generated for r in results)

    metrics = {
        "elapsed": elapsed,
        "generated": generated,
        "tokens_per_second": generated / elapsed,
        "peak_mb": (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if is_cuda(device) else None
        ),
        "gpu_util": util,
    }

    return results, metrics


def fmt_optional(value, fmt, suffix=""):
    return "n/a" if value is None else f"{value:{fmt}}{suffix}"


def main():
    args = parse_args()

    print("=" * 60)
    print("V1 STATIC BATCHING (Phase 9)")
    print("=" * 60)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)

    print(f"\nCheckpoint:\n    {checkpoint_path}")

    model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    sampler = SamplingParams(greedy=True).to_sampler()

    runner = ModelRunner(model=model, tokenizer=tokenizer, sampler=sampler, device=args.device)
    engine = InferenceEngine(runner=runner, tokenizer=tokenizer, sampler=sampler, device=args.device)

    print(f"\nRequests: {len(args.prompts)}  x  max_new_tokens={args.max_new_tokens}  (greedy)")

    for prompt in args.prompts:
        print(f"    [{len(tokenizer.encode(prompt)):>2} tok] \"{prompt}\"")

    # Warm-up both paths so CUDA init / kernel selection isn't timed.
    run(engine, args.prompts[:2], 4, 1, args.device)
    run(engine, args.prompts[:2], 4, 2, args.device)

    sequential_results, sequential = run(
        engine, args.prompts, args.max_new_tokens, 1, args.device
    )
    reference = [r["token_ids"] for r in sequential_results]

    rows = [("sequential", sequential, True)]

    all_match = True

    for batch_size in args.batch_sizes:
        results, metrics = run(engine, args.prompts, args.max_new_tokens, batch_size, args.device)

        match = [r["token_ids"] for r in results] == reference
        all_match &= match

        if not match:
            for ref, res in zip(reference, results):
                if ref != res["token_ids"]:
                    print(f"\n    MISMATCH {res['request'].request_id} (batch={batch_size})")
                    print(f"        sequential = {ref}")
                    print(f"        batched    = {res['token_ids']}")

        rows.append((f"batch={batch_size}", metrics, match))

    print(f"\nThroughput:")
    print(f"    {'mode':<12} {'latency':>9} {'tok/s':>9} {'speedup':>8} "
          f"{'peak MB':>9} {'GPU util':>9}  tokens match")

    for name, metrics, match in rows:
        print(
            f"    {name:<12} {metrics['elapsed']:>8.3f}s "
            f"{metrics['tokens_per_second']:>9.1f} "
            f"{sequential['elapsed'] / metrics['elapsed']:>7.2f}x "
            f"{fmt_optional(metrics['peak_mb'], '9.1f')} "
            f"{fmt_optional(metrics['gpu_util'], '8d', '%'):>9}  {match}"
        )

    print(f"\nSample outputs (sequential == batched):")

    for result in sequential_results[:3]:
        print(f"    {result['request'].request_id}: \"{result['text']}\"")

    print("\n" + "=" * 60)

    if not all_match:
        print("PHASE 9 FAILED")
        print("=" * 60)
        raise RuntimeError("Batched tokens differ from sequential tokens.")

    print("PHASE 9 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
