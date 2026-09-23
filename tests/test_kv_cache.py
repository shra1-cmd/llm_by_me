import torch

from configs.v1 import ModelConfig
from src.inference.kv_cache import KVCache
from src.model.model import V1LanguageModel


def get_model():
    config = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        hidden_dim=32,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        ffn_dim=64,
    )

    model = V1LanguageModel(config)
    model.eval()

    return model, config


# --------------------------------------------------
# KVCache unit tests
# --------------------------------------------------


def test_kv_cache_starts_empty():
    cache = KVCache(num_layers=2)

    assert cache.get_seq_length() == 0
    assert cache.get(0) is None


def test_kv_cache_update_stores_and_grows():
    cache = KVCache(num_layers=1)

    k1 = torch.randn(1, 2, 3, 8)
    v1 = torch.randn(1, 2, 3, 8)

    full_k, full_v = cache.update(0, k1, v1)

    assert full_k.shape == (1, 2, 3, 8)
    assert cache.get_seq_length(0) == 3

    k2 = torch.randn(1, 2, 1, 8)
    v2 = torch.randn(1, 2, 1, 8)

    full_k, full_v = cache.update(0, k2, v2)

    assert full_k.shape == (1, 2, 4, 8)
    assert cache.get_seq_length(0) == 4
    assert torch.equal(full_k[:, :, :3], k1)
    assert torch.equal(full_k[:, :, 3:], k2)


def test_kv_cache_layers_are_independent():
    cache = KVCache(num_layers=2)

    cache.update(0, torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))

    assert cache.get_seq_length(0) == 5
    assert cache.get_seq_length(1) == 0
    assert cache.get(1) is None


# --------------------------------------------------
# Model + KVCache integration
# --------------------------------------------------


def test_forward_without_cache_is_unchanged():
    """Passing kv_cache=None must reproduce Phase 3 behavior exactly."""

    torch.manual_seed(0)
    model, config = get_model()

    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    with torch.no_grad():
        logits_a, _ = model(input_ids)
        logits_b, _ = model(input_ids, kv_cache=None)

    assert torch.equal(logits_a, logits_b)


def test_prefill_then_decode_matches_full_forward():
    """
    Prefill [A B C] into the cache, then decode D and E one at a
    time using the cache, must produce the same next-token logits as
    a single full-sequence forward pass with no cache.
    """

    torch.manual_seed(0)
    model, config = get_model()

    full_sequence = torch.randint(0, config.vocab_size, (1, 5))
    prompt, rest = full_sequence[:, :3], full_sequence[:, 3:]

    with torch.no_grad():
        full_logits, _ = model(full_sequence)

        cache = KVCache(num_layers=config.num_layers)

        prefill_logits, _ = model(prompt, kv_cache=cache)

        decode_logits = [prefill_logits[:, -1:, :]]

        for step in range(rest.shape[1]):
            token = rest[:, step:step + 1]
            step_logits, _ = model(token, kv_cache=cache)
            decode_logits.append(step_logits)

    assert cache.get_seq_length() == 5

    assert torch.allclose(
        prefill_logits,
        full_logits[:, :3, :],
        atol=1e-4,
    )

    for i, token_logits in enumerate(decode_logits[1:]):
        position = 3 + i
        assert torch.allclose(
            token_logits[:, 0, :],
            full_logits[:, position, :],
            atol=1e-4,
        )


def test_decode_step_produces_single_position_logits():
    torch.manual_seed(0)
    model, config = get_model()

    prompt = torch.randint(0, config.vocab_size, (1, 4))
    cache = KVCache(num_layers=config.num_layers)

    with torch.no_grad():
        model(prompt, kv_cache=cache)

        next_token = torch.randint(0, config.vocab_size, (1, 1))
        logits, _ = model(next_token, kv_cache=cache)

    assert logits.shape == (1, 1, config.vocab_size)
    assert cache.get_seq_length() == 5
