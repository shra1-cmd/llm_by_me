"""
Phase 5 — KV-cache correctness validation.

tests/test_kv_cache.py answers "does the cache implementation work?"
(shapes, growth, layer independence, one prefill+decode example).

This file answers the harder question: "does using the cache change
the model's answer?" It is the hard gate before Phase 6 treats the
cache as an inference-system primitive instead of just a model
feature.
"""

import pytest
import torch

from configs.v1 import ModelConfig
from src.inference.kv_cache import KVCache
from src.inference.sampler import Sampler
from src.model.model import V1LanguageModel

ATOL = 1e-3
RTOL = 1e-3


def make_model(seed: int = 0, **config_overrides):
    torch.manual_seed(seed)

    config = ModelConfig(
        vocab_size=64,
        max_seq_len=64,
        hidden_dim=32,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        ffn_dim=64,
    )

    for key, value in config_overrides.items():
        setattr(config, key, value)

    model = V1LanguageModel(config)
    model.eval()

    return model, config


def full_forward(model, input_ids):
    with torch.no_grad():
        logits, _ = model(input_ids)

    return logits


def cached_forward(model, input_ids, split):
    """
    Prefill input_ids[:, :split] into a fresh cache, then decode the
    remaining tokens one at a time. Returns logits over the whole
    sequence (same shape as a full, uncached forward pass) plus the
    cache used, so callers can also inspect its final length.
    """

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


# --------------------------------------------------
# 1 & 2. Numerical correctness across sequence lengths
# --------------------------------------------------

SEQUENCE_LENGTHS = [1, 2, 4, 8, 16, 32]


@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
def test_logits_match_across_sequence_lengths(seq_len):
    model, config = make_model()

    torch.manual_seed(123)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    full_logits = full_forward(model, input_ids)

    # Prefill everything but the last token, then decode that last
    # token through the cache.
    split = max(seq_len - 1, 0)
    cached_logits, _ = cached_forward(model, input_ids, split)

    max_err, mean_err = error_report(full_logits, cached_logits)
    print(f"\nseq_len={seq_len:<3} max_err={max_err:.3e} mean_err={mean_err:.3e}")

    assert torch.allclose(full_logits, cached_logits, atol=ATOL, rtol=RTOL)


# --------------------------------------------------
# 3. Numerical correctness across multiple decode steps
# --------------------------------------------------

DECODE_STEP_COUNTS = [1, 2, 4, 8]


@pytest.mark.parametrize("num_decode_steps", DECODE_STEP_COUNTS)
def test_logits_match_across_decode_step_counts(num_decode_steps):
    model, config = make_model()

    prefill_len = 5
    seq_len = prefill_len + num_decode_steps

    torch.manual_seed(456)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    full_logits = full_forward(model, input_ids)
    cached_logits, cache = cached_forward(model, input_ids, prefill_len)

    max_err, mean_err = error_report(full_logits, cached_logits)
    print(f"\ndecode_steps={num_decode_steps:<3} max_err={max_err:.3e} mean_err={mean_err:.3e}")

    assert torch.allclose(full_logits, cached_logits, atol=ATOL, rtol=RTOL)
    assert cache.get_seq_length() == seq_len


def test_every_position_matches_individually():
    """
    Aggregate allclose can hide a single bad position if the rest of
    the tensor is large. Check every position on its own.
    """

    model, config = make_model()

    torch.manual_seed(789)
    input_ids = torch.randint(0, config.vocab_size, (1, 12))

    full_logits = full_forward(model, input_ids)
    cached_logits, _ = cached_forward(model, input_ids, split=3)

    for position in range(input_ids.shape[1]):
        assert torch.allclose(
            full_logits[:, position, :],
            cached_logits[:, position, :],
            atol=ATOL,
            rtol=RTOL,
        ), f"logits mismatch at position {position}"


# --------------------------------------------------
# 4. Generation equivalence
# --------------------------------------------------


def greedy_generate_naive(model, sampler, prompt_ids, max_new_tokens, eos_token_id=None):
    generated = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)

            next_token = sampler.sample(logits[:, -1, :], generated)

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return generated[0].tolist()


def greedy_generate_cached(model, sampler, prompt_ids, max_new_tokens, eos_token_id=None):
    cache = KVCache(num_layers=model.config.num_layers)

    generated = torch.tensor([prompt_ids], dtype=torch.long)

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


def test_greedy_generation_matches_without_eos():
    model, config = make_model()
    sampler = Sampler(greedy=True)

    torch.manual_seed(1)
    prompt_ids = torch.randint(0, config.vocab_size, (4,)).tolist()

    naive_tokens = greedy_generate_naive(model, sampler, prompt_ids, max_new_tokens=20)
    cached_tokens = greedy_generate_cached(model, sampler, prompt_ids, max_new_tokens=20)

    print(f"\nnaive : {naive_tokens}")
    print(f"cached: {cached_tokens}")

    assert naive_tokens == cached_tokens


def test_greedy_generation_matches_with_eos_stopping():
    """
    EOS behavior must remain correct under the cache: designate a
    token from an unconstrained reference trace as EOS, so that
    stopping actually triggers deterministically, then check the
    naive and cached loops stop at the exact same point.
    """

    model, config = make_model()
    sampler = Sampler(greedy=True)

    torch.manual_seed(2)
    prompt_ids = torch.randint(0, config.vocab_size, (4,)).tolist()

    reference = greedy_generate_naive(model, sampler, prompt_ids, max_new_tokens=15)

    eos_index = len(prompt_ids) + 5
    eos_token_id = reference[eos_index]

    # Greedy decoding on an untrained model can degenerate into
    # repeating a token, so `eos_token_id` may first reappear earlier
    # than `eos_index` once it's treated as a stop condition. Compare
    # against its true first occurrence among generated tokens
    # (skipping the prompt, which the loops never check for EOS).
    first_occurrence = reference.index(eos_token_id, len(prompt_ids))

    naive_tokens = greedy_generate_naive(
        model, sampler, prompt_ids, max_new_tokens=15, eos_token_id=eos_token_id
    )
    cached_tokens = greedy_generate_cached(
        model, sampler, prompt_ids, max_new_tokens=15, eos_token_id=eos_token_id
    )

    assert naive_tokens == cached_tokens
    assert naive_tokens[-1] == eos_token_id
    assert naive_tokens == reference[:first_occurrence + 1]


# --------------------------------------------------
# CUDA path
# --------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_logits_match_on_cuda():
    model, config = make_model()
    model = model.to("cuda")

    torch.manual_seed(321)
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

    full_logits = full_forward(model, input_ids)
    cached_logits, cache = cached_forward(model, input_ids, split=6)

    max_err, mean_err = error_report(full_logits, cached_logits)
    print(f"\n[CUDA] max_err={max_err:.3e} mean_err={mean_err:.3e}")

    assert torch.allclose(full_logits, cached_logits, atol=ATOL, rtol=RTOL)
    assert cache.get_seq_length() == 16
