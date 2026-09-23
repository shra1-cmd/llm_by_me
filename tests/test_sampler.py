import pytest
import torch

from src.inference.sampler import Sampler


def test_greedy_sampling():
    sampler = Sampler(greedy=True)

    logits = torch.tensor([[1.0, 2.0, 10.0, 3.0]])

    next_token = sampler.sample(logits)

    assert next_token.item() == 2


def test_greedy_sampling_batch():
    sampler = Sampler(greedy=True)

    logits = torch.tensor(
        [
            [1.0, 2.0, 10.0, 3.0],
            [5.0, 1.0, 0.0, 0.0],
        ]
    )

    next_token = sampler.sample(logits)

    assert next_token.tolist() == [2, 0]


def test_temperature_changes_distribution_not_argmax_identity():
    torch.manual_seed(0)

    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

    low_temp_sampler = Sampler(temperature=0.1)
    high_temp_sampler = Sampler(temperature=5.0)

    low_temp_samples = torch.stack(
        [low_temp_sampler.sample(logits) for _ in range(50)]
    )

    high_temp_samples = torch.stack(
        [high_temp_sampler.sample(logits) for _ in range(50)]
    )

    # Low temperature should almost always pick the dominant token.
    assert (low_temp_samples == 0).float().mean() > 0.95

    # High temperature should occasionally pick other tokens.
    assert (high_temp_samples != 0).any()


def test_top_k_restricts_candidates():
    torch.manual_seed(0)

    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])

    sampler = Sampler(top_k=2, temperature=1.0)

    samples = torch.cat([sampler.sample(logits) for _ in range(100)])

    assert set(samples.tolist()).issubset({0, 1})


def test_top_p_restricts_candidates():
    torch.manual_seed(0)

    # After softmax, token 0 dominates heavily.
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])

    sampler = Sampler(top_p=0.5, temperature=1.0)

    samples = torch.cat([sampler.sample(logits) for _ in range(50)])

    assert set(samples.tolist()) == {0}


def test_repetition_penalty_reduces_repeat_probability():
    torch.manual_seed(0)

    logits = torch.tensor([[5.0, 5.0, 5.0, 5.0]])

    generated_tokens = torch.tensor([[0]])

    sampler = Sampler(repetition_penalty=2.0, temperature=1.0)

    samples = torch.stack(
        [
            sampler.sample(logits, generated_tokens)
            for _ in range(200)
        ]
    )

    fraction_repeated = (samples == 0).float().mean().item()

    # Without penalty this would be ~0.25 (uniform over 4 tokens).
    assert fraction_repeated < 0.20


def test_repeat_ngram_blocking_prevents_exact_repeat():
    sampler = Sampler(greedy=True, repeat_ngram_size=3)

    # Sequence: A B C A B -> next greedy pick would be C (highest logit),
    # but that recreates ngram "A B C" which already occurred.
    generated_tokens = torch.tensor([[0, 1, 2, 0, 1]])

    logits = torch.tensor([[1.0, 1.0, 10.0, 1.0]])

    next_token = sampler.sample(logits, generated_tokens)

    assert next_token.item() != 2


def test_invalid_temperature_raises():
    with pytest.raises(ValueError):
        Sampler(temperature=0.0)


def test_invalid_top_k_raises():
    with pytest.raises(ValueError):
        Sampler(top_k=-1)


def test_invalid_top_p_raises():
    with pytest.raises(ValueError):
        Sampler(top_p=0.0)

    with pytest.raises(ValueError):
        Sampler(top_p=1.5)


def test_invalid_repetition_penalty_raises():
    with pytest.raises(ValueError):
        Sampler(repetition_penalty=0.0)


def test_invalid_logits_shape_raises():
    sampler = Sampler(greedy=True)

    with pytest.raises(ValueError):
        sampler.sample(torch.tensor([1.0, 2.0, 3.0]))


def test_deterministic_mode_is_reproducible():
    sampler = Sampler(greedy=True)

    logits = torch.tensor([[1.0, 2.0, 10.0, 3.0]])

    first = sampler.sample(logits)
    second = sampler.sample(logits)

    assert torch.equal(first, second)


def test_random_sampling_varies():
    torch.manual_seed(0)

    sampler = Sampler(temperature=1.0)

    logits = torch.zeros((1, 10))

    samples = torch.stack([sampler.sample(logits) for _ in range(30)])

    assert samples.unique().numel() > 1


def test_batch_dimension_is_preserved():
    sampler = Sampler(greedy=True)

    logits = torch.randn(4, 20)

    next_token = sampler.sample(logits)

    assert next_token.shape == (4,)
