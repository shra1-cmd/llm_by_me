"""
Phase 1 — standalone sampler inspection.

Uses artificial logits only. No model, no checkpoint, no tokenizer.

Usage:
    PYTHONPATH="$(pwd)" python scripts/test_sampler.py
"""

import torch

from src.inference.sampler import Sampler


def section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    torch.manual_seed(0)

    # --------------------------------------------------
    # Greedy
    # --------------------------------------------------

    section("GREEDY")

    logits = torch.tensor([[1.0, 2.0, 10.0, 3.0]])

    sampler = Sampler(greedy=True)

    next_token = sampler.sample(logits)

    print(f"logits          : {logits.tolist()}")
    print(f"next_token      : {next_token.tolist()} (expected [2])")

    # --------------------------------------------------
    # Temperature
    # --------------------------------------------------

    section("TEMPERATURE")

    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

    for temperature in (0.1, 1.0, 5.0):
        sampler = Sampler(temperature=temperature)

        samples = torch.cat([sampler.sample(logits) for _ in range(200)])

        counts = torch.bincount(samples, minlength=4)

        print(f"temperature={temperature:<4} -> token counts over 200 draws: {counts.tolist()}")

    # --------------------------------------------------
    # Top-K
    # --------------------------------------------------

    section("TOP-K")

    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])

    sampler = Sampler(top_k=2, temperature=1.0)

    samples = torch.cat([sampler.sample(logits) for _ in range(200)])

    counts = torch.bincount(samples, minlength=5)

    print(f"logits          : {logits.tolist()}")
    print(f"top_k=2 counts  : {counts.tolist()} (only indices 0,1 should be nonzero)")

    # --------------------------------------------------
    # Top-P
    # --------------------------------------------------

    section("TOP-P")

    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])

    sampler = Sampler(top_p=0.5, temperature=1.0)

    samples = torch.cat([sampler.sample(logits) for _ in range(200)])

    counts = torch.bincount(samples, minlength=5)

    print(f"logits          : {logits.tolist()}")
    print(f"top_p=0.5 counts: {counts.tolist()} (should be dominated by index 0)")

    # --------------------------------------------------
    # Repetition penalty
    # --------------------------------------------------

    section("REPETITION PENALTY")

    logits = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    generated_tokens = torch.tensor([[0]])

    sampler_no_penalty = Sampler(repetition_penalty=1.0, temperature=1.0)
    sampler_with_penalty = Sampler(repetition_penalty=2.0, temperature=1.0)

    samples_no_penalty = torch.cat(
        [sampler_no_penalty.sample(logits, generated_tokens) for _ in range(200)]
    )
    samples_with_penalty = torch.cat(
        [sampler_with_penalty.sample(logits, generated_tokens) for _ in range(200)]
    )

    print(f"generated so far        : {generated_tokens.tolist()}")
    print(f"no penalty, token 0 rate  : {(samples_no_penalty == 0).float().mean().item():.2f} (expected ~0.25)")
    print(f"penalty=2.0, token 0 rate: {(samples_with_penalty == 0).float().mean().item():.2f} (expected lower)")

    # --------------------------------------------------
    # Repeat n-gram blocking
    # --------------------------------------------------

    section("REPEAT N-GRAM BLOCKING")

    generated_tokens = torch.tensor([[0, 1, 2, 0, 1]])
    logits = torch.tensor([[1.0, 1.0, 10.0, 1.0]])

    sampler = Sampler(greedy=True, repeat_ngram_size=3)

    next_token = sampler.sample(logits, generated_tokens)

    print(f"generated so far : {generated_tokens.tolist()} (contains ngram [0,1,2])")
    print(f"logits           : {logits.tolist()} (argmax would be token 2)")
    print(f"next_token       : {next_token.tolist()} (token 2 must be blocked)")

    print("\nAll sampler mechanisms exercised.\n")


if __name__ == "__main__":
    main()
