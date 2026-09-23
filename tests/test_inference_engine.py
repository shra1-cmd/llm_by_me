"""
Phase 6 — prefill + decode inference engine.

Phase 5 (tests/test_kv_cache_correctness.py) already proves the cache
does not change the model's answer. These tests validate the engine
interface on top of it: ModelRunner.prefill / decode and the
InferenceEngine lifecycle, and that the engine produces exactly the
same tokens as the Phase 3 naive ModelRunner.generate.
"""

import pytest
import torch

from configs.v1 import ModelConfig
from src.inference.inference_engine import InferenceEngine
from src.inference.model_runner import ModelRunner
from src.inference.sampler import Sampler
from src.model.model import V1LanguageModel

ATOL = 1e-3
RTOL = 1e-3


class IdTokenizer:
    """Space-separated integer ids; `<eos>` maps to a configurable id."""

    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id

    def encode(self, text):
        return [int(t) for t in text.split()]

    def decode(self, token_ids):
        return " ".join(str(i) for i in token_ids)

    def token_to_id(self, token):
        return self.eos_token_id if token == "<eos>" else None


def make_model(seed: int = 0, max_seq_len: int = 64):
    torch.manual_seed(seed)

    config = ModelConfig(
        vocab_size=64,
        max_seq_len=max_seq_len,
        hidden_dim=32,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        ffn_dim=64,
    )

    model = V1LanguageModel(config)
    model.eval()

    return model, config


def make_engine(model, eos_token_id=None, sampler=None):
    tokenizer = IdTokenizer(eos_token_id=eos_token_id)
    sampler = sampler or Sampler(greedy=True)

    runner = ModelRunner(model, tokenizer, sampler, device="cpu")
    engine = InferenceEngine(runner, tokenizer, sampler, device="cpu")

    return runner, engine


def random_ids(config, length, seed):
    torch.manual_seed(seed)
    return torch.randint(0, config.vocab_size, (1, length))


# --------------------------------------------------
# 1. Prefill populates the cache
# --------------------------------------------------


def test_prefill_populates_cache():
    model, config = make_model()
    runner, _ = make_engine(model)

    input_ids = random_ids(config, 7, seed=1)

    out = runner.prefill(input_ids)

    assert out.kv_cache.get_seq_length() == 7
    assert out.logits.shape == (1, config.vocab_size)

    for layer_idx in range(config.num_layers):
        assert out.kv_cache.get_seq_length(layer_idx) == 7


# --------------------------------------------------
# 2. Single decode
# --------------------------------------------------


def test_single_decode_grows_cache_by_one():
    model, config = make_model()
    runner, _ = make_engine(model)

    input_ids = random_ids(config, 5, seed=2)

    prefill = runner.prefill(input_ids)
    token = prefill.logits.argmax(dim=-1, keepdim=True)

    decode = runner.decode(token, prefill.kv_cache)

    assert decode.kv_cache is prefill.kv_cache
    assert decode.kv_cache.get_seq_length() == 5 + 1
    assert decode.logits.shape == (1, config.vocab_size)


# --------------------------------------------------
# 3. Multiple decode steps
# --------------------------------------------------


def test_multiple_decodes_grow_cache_one_per_step():
    model, config = make_model()
    runner, _ = make_engine(model)

    prompt_len = 4
    input_ids = random_ids(config, prompt_len, seed=3)

    out = runner.prefill(input_ids)
    kv_cache = out.kv_cache

    lengths = [kv_cache.get_seq_length()]

    for _ in range(5):
        token = out.logits.argmax(dim=-1, keepdim=True)
        out = runner.decode(token, kv_cache)
        lengths.append(kv_cache.get_seq_length())

    assert lengths == [prompt_len + i for i in range(6)]


def test_decode_rejects_bad_inputs():
    model, config = make_model()
    runner, _ = make_engine(model)

    prefill = runner.prefill(random_ids(config, 3, seed=4))

    with pytest.raises(ValueError):
        runner.decode(random_ids(config, 2, seed=5), prefill.kv_cache)

    from src.inference.kv_cache import KVCache

    with pytest.raises(ValueError):
        runner.decode(random_ids(config, 1, seed=6), KVCache(config.num_layers))

    with pytest.raises(ValueError):
        runner.prefill(torch.zeros((1, 0), dtype=torch.long))


# --------------------------------------------------
# 4. Logit equivalence through the runner interface
# --------------------------------------------------


def test_prefill_decode_logits_match_full_forward():
    model, config = make_model()
    runner, _ = make_engine(model)

    prompt_len = 6
    input_ids = random_ids(config, 12, seed=7)

    with torch.no_grad():
        full_logits, _ = model(input_ids)

    out = runner.prefill(input_ids[:, :prompt_len])

    assert torch.allclose(out.logits, full_logits[:, prompt_len - 1, :], atol=ATOL, rtol=RTOL)

    for position in range(prompt_len, input_ids.shape[1]):
        out = runner.decode(input_ids[:, position:position + 1], out.kv_cache)

        assert torch.allclose(
            out.logits, full_logits[:, position, :], atol=ATOL, rtol=RTOL
        ), f"logits mismatch at position {position}"


# --------------------------------------------------
# 5. Generation equivalence: Phase 3 naive vs Phase 6 engine
# --------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engine_matches_naive_runner(seed):
    model, config = make_model(seed=seed)
    runner, engine = make_engine(model)

    prompt = " ".join(str(i) for i in random_ids(config, 5, seed=100 + seed)[0].tolist())

    naive = runner.generate(prompt, max_new_tokens=25)
    cached = engine.generate(prompt, max_new_tokens=25)

    assert naive["token_ids"] == cached["token_ids"]
    assert naive["text"] == cached["text"]
    assert naive["stats"]["generated_tokens"] == cached["stats"]["generated_tokens"] == 25


def test_engine_matches_naive_with_repetition_penalty():
    """Sampler history (generated tokens) must be threaded identically."""

    model, config = make_model()
    sampler = Sampler(greedy=True, repetition_penalty=1.3, repeat_ngram_size=2)
    runner, engine = make_engine(model, sampler=sampler)

    prompt = "3 14 15 9 26"

    naive = runner.generate(prompt, max_new_tokens=20)
    cached = engine.generate(prompt, max_new_tokens=20)

    assert naive["token_ids"] == cached["token_ids"]


def test_engine_matches_naive_at_max_seq_len():
    model, config = make_model(max_seq_len=16)
    runner, engine = make_engine(model)

    prompt = "1 2 3 4 5 6 7 8 9 10"

    naive = runner.generate(prompt, max_new_tokens=50)
    cached = engine.generate(prompt, max_new_tokens=50)

    assert naive["token_ids"] == cached["token_ids"]
    assert len(cached["token_ids"]) == 16
    assert cached["stats"]["final_kv_length"] <= 16


# --------------------------------------------------
# 6. EOS stops generation
# --------------------------------------------------


def test_engine_stops_at_eos():
    model, config = make_model()

    _, reference_engine = make_engine(model)
    prompt_ids = random_ids(config, 4, seed=8)[0].tolist()

    reference = reference_engine.generate_from_ids(prompt_ids, max_new_tokens=15)["token_ids"]

    # Pick a token from the unconstrained trace as EOS; stopping must
    # happen at its first occurrence among the generated tokens.
    eos_token_id = reference[len(prompt_ids) + 5]
    first_occurrence = reference.index(eos_token_id, len(prompt_ids))

    runner, engine = make_engine(model, eos_token_id=eos_token_id)

    prompt = " ".join(str(i) for i in prompt_ids)

    cached = engine.generate(prompt, max_new_tokens=15)
    naive = runner.generate(prompt, max_new_tokens=15)

    assert cached["token_ids"][-1] == eos_token_id
    assert cached["token_ids"] == reference[:first_occurrence + 1]
    assert cached["token_ids"] == naive["token_ids"]


def test_eos_on_first_token_skips_decode():
    model, config = make_model()

    _, reference_engine = make_engine(model)
    prompt_ids = [5, 6, 7]
    first = reference_engine.generate_from_ids(prompt_ids, max_new_tokens=1)["token_ids"][-1]

    _, engine = make_engine(model, eos_token_id=first)
    result = engine.generate_from_ids(prompt_ids, max_new_tokens=10)

    assert result["stats"]["generated_tokens"] == 1
    assert result["stats"]["final_kv_length"] == len(prompt_ids)


# --------------------------------------------------
# 7. max_new_tokens
# --------------------------------------------------


@pytest.mark.parametrize("max_new_tokens", [0, 1, 10])
def test_engine_respects_max_new_tokens(max_new_tokens):
    model, config = make_model()
    _, engine = make_engine(model)

    prompt_ids = random_ids(config, 5, seed=9)[0].tolist()

    result = engine.generate_from_ids(prompt_ids, max_new_tokens=max_new_tokens)
    stats = result["stats"]

    assert stats["generated_tokens"] == max_new_tokens
    assert len(result["token_ids"]) == 5 + max_new_tokens

    if max_new_tokens > 0:
        assert stats["prefill_kv_length"] == 5
        # The last sampled token is never fed back through decode.
        assert stats["final_kv_length"] == 5 + max_new_tokens - 1


# --------------------------------------------------
# CUDA path
# --------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_engine_matches_naive_on_cuda():
    model, config = make_model()
    model = model.to("cuda")

    tokenizer = IdTokenizer()
    sampler = Sampler(greedy=True)
    runner = ModelRunner(model, tokenizer, sampler, device="cuda")
    engine = InferenceEngine(runner, tokenizer, sampler, device="cuda")

    prompt = "10 20 30 40 50"

    naive = runner.generate(prompt, max_new_tokens=20)
    cached = engine.generate(prompt, max_new_tokens=20)

    assert naive["token_ids"] == cached["token_ids"]
