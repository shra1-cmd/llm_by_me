"""
Phase 7 — request abstraction, end-to-end.

prompt -> InferenceRequest -> InferenceEngine (prefill / decode steps)
       -> finished request

Prints the request's state as it moves through its lifecycle, then
checks the tokens against the Phase 3 naive ModelRunner.generate.

Usage:
    PYTHONPATH="$(pwd)" python scripts/run_request.py
    PYTHONPATH="$(pwd)" python scripts/run_request.py --prompt "Once upon a time" --max-new-tokens 100
"""

import argparse

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.inference_engine import InferenceEngine
from src.inference.model_runner import ModelRunner
from src.inference.request import RequestStatus, SamplingParams
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


def describe(request, label):
    print(
        f"    {label:<14} status={request.status.name:<10} "
        f"position={request.position:<4} "
        f"generated={request.num_generated:<4} "
        f"kv_len={request.num_cached_tokens}"
    )


def main():
    args = parse_args()

    print("=" * 60)
    print("V1 REQUEST ABSTRACTION (Phase 7)")
    print("=" * 60)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)

    print(f"\nCheckpoint:\n    {checkpoint_path}")

    model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    sampling_params = SamplingParams(
        greedy=True,
        repetition_penalty=args.repetition_penalty,
        repeat_ngram_size=args.repeat_ngram_size,
    )
    sampler = sampling_params.to_sampler()

    runner = ModelRunner(model=model, tokenizer=tokenizer, sampler=sampler, device=args.device)
    engine = InferenceEngine(runner=runner, tokenizer=tokenizer, device=args.device)

    # Warm-up so CUDA init / kernel selection doesn't land in the timing.
    engine.generate(args.prompt, max_new_tokens=2, sampling_params=sampling_params)

    request = engine.create_request(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        sampling_params=sampling_params,
    )

    print(f"\nRequest:")
    print(f"    request_id          = {request.request_id}")
    print(f"    prompt              = \"{request.prompt}\"")
    print(f"    input_tokens        = {request.input_tokens}")
    print(f"    max_new_tokens      = {request.max_new_tokens}")
    print(f"    sampling_params     = {request.sampling_params}")

    print(f"\nLifecycle:")
    describe(request, "created")

    engine.prefill(request)
    describe(request, "after prefill")

    steps = 0

    while request.status == RequestStatus.DECODING:
        engine.decode(request)
        steps += 1

        if steps in (1, 2) or request.is_finished:
            describe(request, f"decode #{steps}")
        elif steps == 3:
            print("    ...")

    print(f"\nFinished:")
    print(f"    status              = {request.status.name}")
    print(f"    finish_reason       = {request.finish_reason.value}")
    print(f"    generated tokens    = {request.num_generated}")
    print(f"    final KV length     = {request.num_cached_tokens}"
          f"  (last sampled token not fed back)")

    text = tokenizer.decode(request.all_tokens)
    print(f"\nGenerated text:\n    \"{text}\"")

    # Timed run through the convenience wrapper, same lifecycle.
    timed = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens,
                            sampling_params=sampling_params)
    stats = timed["stats"]

    print(f"\nGeneration:")
    print(f"    elapsed             = {stats['elapsed_seconds']:.4f} s")
    print(f"    tokens/sec          = {stats['tokens_per_second']:.2f}")

    naive = runner.generate(args.prompt, max_new_tokens=args.max_new_tokens)

    token_match = (
        naive["token_ids"] == request.all_tokens == timed["token_ids"]
    )

    print(f"\nCorrectness:")
    print(f"    naive tokens        = {naive['stats']['generated_tokens']}")
    print(f"    request tokens      = {request.num_generated}")
    print(f"    token match         = {token_match}")

    print("\n" + "=" * 60)

    if not token_match:
        print("PHASE 7 FAILED")
        print("=" * 60)
        raise RuntimeError("Request tokens differ from naive runner tokens.")

    print("PHASE 7 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
