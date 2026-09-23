"""
Phase 5 — KV-cache correctness report, against the real checkpoint.

Full forward pass vs prefill+decode through the KV cache, compared
across sequence lengths and decode-step counts, plus end-to-end
greedy generation equivalence. This is the human-readable companion
to tests/test_kv_cache_correctness.py, which is the actual gate.

Usage:
    PYTHONPATH="$(pwd)" python scripts/check_kv_cache_correctness.py
    PYTHONPATH="$(pwd)" python scripts/check_kv_cache_correctness.py --prompt "Once upon a time"
"""

import argparse

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.kv_cache import KVCache
from src.inference.sampler import Sampler
from src.tokenizer.tokenizer import BPETokenizer

ATOL = 1e-3
RTOL = 1e-3

SEQUENCE_LENGTHS = [1, 2, 4, 8, 16, 32]
DECODE_STEP_COUNTS = [1, 2, 4, 8]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def full_forward(model, input_ids):
    with torch.no_grad():
        logits, _ = model(input_ids)
    return logits


def cached_forward(model, input_ids, split):
    cache = KVCache(num_layers=model.config.num_layers)
    logits_chunks = []

    with torch.no_grad():
        prompt = input_ids[:, :split]

        if prompt.shape[1] > 0:
            prefill_logits, _ = model(prompt, kv_cache=cache)
            logits_chunks.append(prefill_logits)

        for position in range(split, input_ids.shape[1]):
            token = input_ids[:, position:position + 1]
            step_logits, _ = model(token, kv_cache=cache)
            logits_chunks.append(step_logits)

    return torch.cat(logits_chunks, dim=1), cache


def error_report(full_logits, cached_logits):
    diff = (full_logits - cached_logits).abs()
    return diff.max().item(), diff.mean().item()


def greedy_generate_naive(model, sampler, prompt_ids, max_new_tokens, device, eos_token_id=None):
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_token = sampler.sample(logits[:, -1, :], generated)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return generated[0].tolist()


def greedy_generate_cached(model, sampler, prompt_ids, max_new_tokens, device, eos_token_id=None):
    cache = KVCache(num_layers=model.config.num_layers)
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(generated, kv_cache=cache)
        next_token = sampler.sample(logits[:, -1, :], generated)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

        steps_left = max_new_tokens - 1
        stopped = eos_token_id is not None and next_token.item() == eos_token_id

        while steps_left > 0 and not stopped:
            logits, _ = model(next_token.unsqueeze(-1), kv_cache=cache)
            next_token = sampler.sample(logits[:, -1, :], generated)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            steps_left -= 1
            stopped = eos_token_id is not None and next_token.item() == eos_token_id

    return generated[0].tolist()


def main():
    args = parse_args()

    print("=" * 60)
    print("PHASE 5 — KV CACHE CORRECTNESS")
    print("=" * 60)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    vocab_size = metadata["model_config"].vocab_size

    all_passed = True
    max_error_seen = 0.0
    sum_error = 0.0
    num_error_samples = 0

    # --------------------------------------------------
    # Sequence length sweep
    # --------------------------------------------------

    print("\nSequence length tests:")

    for seq_len in SEQUENCE_LENGTHS:
        torch.manual_seed(1000 + seq_len)
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=args.device)

        full_logits = full_forward(model, input_ids)
        cached_logits, _ = cached_forward(model, input_ids, split=max(seq_len - 1, 0))

        max_err, mean_err = error_report(full_logits, cached_logits)
        passed = torch.allclose(full_logits, cached_logits, atol=ATOL, rtol=RTOL)

        all_passed &= passed
        max_error_seen = max(max_error_seen, max_err)
        sum_error += mean_err
        num_error_samples += 1

        print(f"    {seq_len:<3} {'PASS' if passed else 'FAIL'}  (max_err={max_err:.3e}, mean_err={mean_err:.3e})")

    # --------------------------------------------------
    # Decode step sweep
    # --------------------------------------------------

    print("\nDecode steps:")

    prefill_len = 8

    for num_decode_steps in DECODE_STEP_COUNTS:
        torch.manual_seed(2000 + num_decode_steps)
        seq_len = prefill_len + num_decode_steps
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=args.device)

        full_logits = full_forward(model, input_ids)
        cached_logits, cache = cached_forward(model, input_ids, split=prefill_len)

        max_err, mean_err = error_report(full_logits, cached_logits)
        passed = torch.allclose(full_logits, cached_logits, atol=ATOL, rtol=RTOL)
        passed = passed and cache.get_seq_length() == seq_len

        all_passed &= passed
        max_error_seen = max(max_error_seen, max_err)
        sum_error += mean_err
        num_error_samples += 1

        print(f"    {num_decode_steps:<3} {'PASS' if passed else 'FAIL'}  (max_err={max_err:.3e}, mean_err={mean_err:.3e})")

    mean_error_seen = sum_error / num_error_samples

    print(f"\nMaximum absolute error : {max_error_seen:.6e}")
    print(f"Mean absolute error    : {mean_error_seen:.6e}")

    # --------------------------------------------------
    # Generation equivalence
    # --------------------------------------------------

    print("\nGeneration equivalence:")

    sampler = Sampler(greedy=True)
    prompt_ids = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.token_to_id("<eos>")

    naive_tokens = greedy_generate_naive(
        model, sampler, prompt_ids, args.max_new_tokens, args.device, eos_token_id=eos_token_id
    )
    cached_tokens = greedy_generate_cached(
        model, sampler, prompt_ids, args.max_new_tokens, args.device, eos_token_id=eos_token_id
    )

    generation_match = naive_tokens == cached_tokens
    all_passed &= generation_match

    print(f"    uncached tokens = {naive_tokens}")
    print(f"    cached tokens   = {cached_tokens}")
    print(f"    MATCH           = {generation_match}")

    print(f"    uncached text   = \"{tokenizer.decode(naive_tokens)}\"")
    print(f"    cached text     = \"{tokenizer.decode(cached_tokens)}\"")

    print("\n" + "=" * 60)

    if not all_passed:
        print("PHASE 5 FAILED")
        print("=" * 60)
        raise RuntimeError("KV cache correctness checks failed. See report above.")

    print("PHASE 5 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
