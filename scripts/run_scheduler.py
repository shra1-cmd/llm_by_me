"""
Phase 8 — FIFO scheduler, end-to-end.

Submit several requests -> Scheduler picks them in FIFO order ->
InferenceEngine executes each one (prefill + decode) sequentially.

Each request's tokens are checked against the Phase 3 naive
ModelRunner.generate on the same prompt.

Usage:
    PYTHONPATH="$(pwd)" python scripts/run_scheduler.py
    PYTHONPATH="$(pwd)" python scripts/run_scheduler.py --max-new-tokens 30 \\
        --prompts "Once upon a time" "The cat" "Hello, my name is"
"""

import argparse

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
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--prompts", type=str, nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("V1 FIFO SCHEDULER (Phase 8)")
    print("=" * 60)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)

    print(f"\nCheckpoint:\n    {checkpoint_path}")

    model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    sampling_params = SamplingParams(greedy=True)
    sampler = sampling_params.to_sampler()

    runner = ModelRunner(model=model, tokenizer=tokenizer, sampler=sampler, device=args.device)
    engine = InferenceEngine(runner=runner, tokenizer=tokenizer, sampler=sampler, device=args.device)

    # Warm-up so CUDA init / kernel selection doesn't land in the timing.
    engine.generate(args.prompts[0], max_new_tokens=2)

    print(f"\nSubmitted:")

    requests = []

    for prompt in args.prompts:
        request = engine.submit(
            engine.create_request(prompt, max_new_tokens=args.max_new_tokens)
        )
        requests.append(request)
        print(f"    {request.request_id}  \"{prompt}\"")

    scheduler = engine.scheduler

    print(f"\nQueue:")
    print(f"    waiting             = {[r.request_id for r in scheduler.waiting]}")
    print(f"    running             = {list(scheduler.running)}")

    print(f"\nExecution (FIFO):")

    results = []

    while scheduler.has_waiting():
        result = engine.step()
        request = result["request"]
        stats = result["stats"]

        results.append(engine.pop_result(request.request_id))

        print(
            f"    {request.request_id}  {request.status.name:<9} "
            f"finish={stats['finish_reason']:<15} "
            f"generated={stats['generated_tokens']:<4} "
            f"tok/s={stats['tokens_per_second']:.1f}  "
            f"waiting_left={scheduler.num_waiting}"
        )

    execution_order = [r["request"].request_id for r in results]
    submit_order = [r.request_id for r in requests]
    fifo_ok = execution_order == submit_order
    queue_empty = not scheduler.has_work()

    print(f"\nOutputs:")

    for result in results:
        print(f"    {result['request'].request_id}: \"{result['text']}\"")

    print(f"\nCorrectness:")

    all_match = True

    for result in results:
        request = result["request"]
        naive = runner.generate(request.prompt, max_new_tokens=args.max_new_tokens)
        match = naive["token_ids"] == result["token_ids"]
        all_match &= match
        print(f"    {request.request_id}  token match = {match}")

    print(f"    FIFO order          = {fifo_ok}")
    print(f"    queue drained       = {queue_empty}")

    passed = all_match and fifo_ok and queue_empty

    print("\n" + "=" * 60)

    if not passed:
        print("PHASE 8 FAILED")
        print("=" * 60)
        raise RuntimeError("Scheduler checks failed. See report above.")

    print("PHASE 8 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
