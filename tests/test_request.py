"""
Phase 7 — request abstraction.

InferenceRequest carries all per-generation state (tokens, sampling
params, KV cache, position, lifecycle status). These tests check the
request on its own, then its integration with InferenceEngine, which
must still produce exactly the Phase 3 naive / Phase 6 tokens.
"""

import pytest
import torch

from configs.v1 import ModelConfig
from src.inference.inference_engine import InferenceEngine
from src.inference.model_runner import ModelRunner
from src.inference.request import (
    FinishReason,
    InferenceRequest,
    RequestStatus,
    SamplingParams,
)
from src.inference.sampler import Sampler
from src.model.model import V1LanguageModel


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


GREEDY = SamplingParams(greedy=True)


# --------------------------------------------------
# 1. Request creation
# --------------------------------------------------


def test_request_creation_fields():
    params = SamplingParams(temperature=0.7, top_k=40, top_p=0.9)

    request = InferenceRequest(
        prompt="1 2 3",
        input_tokens=[1, 2, 3],
        max_new_tokens=12,
        sampling_params=params,
        request_id="req_test",
    )

    assert request.request_id == "req_test"
    assert request.prompt == "1 2 3"
    assert request.sampling_params is params
    assert request.max_new_tokens == 12
    assert request.status == RequestStatus.WAITING
    assert request.finish_reason is None
    assert request.kv_cache is None
    assert not request.is_finished


def test_request_ids_are_unique_by_default():
    a = InferenceRequest(prompt="a", input_tokens=[1])
    b = InferenceRequest(prompt="b", input_tokens=[2])

    assert a.request_id != b.request_id
    assert a.request_id.startswith("req_")


def test_request_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        InferenceRequest(prompt="", input_tokens=[])

    with pytest.raises(ValueError):
        InferenceRequest(prompt="x", input_tokens=[1], max_new_tokens=-1)

    with pytest.raises(ValueError):
        SamplingParams(temperature=0.0)


def test_sampling_params_roundtrip_with_sampler():
    params = SamplingParams(
        greedy=False,
        temperature=0.8,
        top_k=5,
        top_p=0.95,
        repetition_penalty=1.2,
        repeat_ngram_size=3,
    )

    assert SamplingParams.from_sampler(params.to_sampler()) == params


# --------------------------------------------------
# 2. Token state
# --------------------------------------------------


def test_token_state_and_position():
    prompt_ids = [5, 6, 7, 8]

    request = InferenceRequest(prompt="", input_tokens=prompt_ids, max_new_tokens=10)

    # The request owns its own copy of the prompt tokens.
    prompt_ids.append(99)

    assert request.input_tokens == [5, 6, 7, 8]
    assert request.generated_tokens == []
    assert request.prompt_len == 4
    assert request.position == 4

    request.start_prefill()
    request.append_token(11)
    request.append_token(12)

    assert request.generated_tokens == [11, 12]
    assert request.all_tokens == [5, 6, 7, 8, 11, 12]
    assert request.last_token == 12
    assert request.position == 6


# --------------------------------------------------
# 3. State transitions
# --------------------------------------------------


def test_state_transitions_happy_path():
    request = InferenceRequest(prompt="", input_tokens=[1, 2], max_new_tokens=2)

    assert request.status == RequestStatus.WAITING

    request.start_prefill()
    assert request.status == RequestStatus.PREFILLING

    request.append_token(3)
    assert request.status == RequestStatus.DECODING

    request.append_token(4)
    assert request.status == RequestStatus.FINISHED
    assert request.finish_reason == FinishReason.MAX_NEW_TOKENS
    assert request.is_finished


def test_invalid_transitions_raise():
    request = InferenceRequest(prompt="", input_tokens=[1], max_new_tokens=5)

    with pytest.raises(RuntimeError):
        request.start_decode()

    with pytest.raises(RuntimeError):
        request.append_token(1)

    request.start_prefill()

    with pytest.raises(RuntimeError):
        request.start_prefill()

    request.abort()

    assert request.status == RequestStatus.ABORTED
    assert request.finish_reason == FinishReason.ABORTED

    with pytest.raises(RuntimeError):
        request.start_decode()


def test_engine_walks_request_through_lifecycle():
    model, _ = make_model()
    _, engine = make_engine(model)

    request = engine.create_request("1 2 3", max_new_tokens=3, sampling_params=GREEDY)

    seen = [request.status]

    engine.prefill(request)
    seen.append(request.status)

    while request.status == RequestStatus.DECODING:
        engine.decode(request)
        seen.append(request.status)

    assert seen == [
        RequestStatus.WAITING,
        RequestStatus.DECODING,
        RequestStatus.DECODING,
        RequestStatus.FINISHED,
    ]

    with pytest.raises(RuntimeError):
        engine.decode(request)

    with pytest.raises(RuntimeError):
        engine.prefill(request)


# --------------------------------------------------
# 4. KV cache association
# --------------------------------------------------


def test_each_request_owns_its_kv_cache():
    model, _ = make_model()
    _, engine = make_engine(model)

    a = engine.create_request("1 2 3", max_new_tokens=5, sampling_params=GREEDY)
    b = engine.create_request("4 5 6 7 8 9", max_new_tokens=5, sampling_params=GREEDY)

    engine.prefill(a)
    engine.prefill(b)

    assert a.kv_cache is not None and b.kv_cache is not None
    assert a.kv_cache is not b.kv_cache
    assert a.num_cached_tokens == 3
    assert b.num_cached_tokens == 6

    # Decoding A must not touch B's cache.
    engine.decode(a)

    assert a.num_cached_tokens == 4
    assert b.num_cached_tokens == 6


def test_interleaved_requests_match_sequential_runs():
    """
    Because all state lives on the request, manually interleaving two
    requests' decode steps must give the same tokens as running each
    alone. (Not a scheduler — just proving state isolation.)
    """

    model, _ = make_model()
    _, engine = make_engine(model)

    solo_a = engine.generate("1 2 3", max_new_tokens=8)["token_ids"]
    solo_b = engine.generate("9 8 7 6", max_new_tokens=8)["token_ids"]

    a = engine.create_request("1 2 3", max_new_tokens=8)
    b = engine.create_request("9 8 7 6", max_new_tokens=8)

    engine.prefill(a)
    engine.prefill(b)

    while not (a.is_finished and b.is_finished):
        for request in (a, b):
            if not request.is_finished:
                engine.decode(request)

    assert a.all_tokens == solo_a
    assert b.all_tokens == solo_b


def test_decode_detects_cache_position_mismatch():
    model, _ = make_model()
    _, engine = make_engine(model)

    a = engine.create_request("1 2 3", max_new_tokens=5)
    b = engine.create_request("4 5", max_new_tokens=5)

    engine.prefill(a)
    engine.prefill(b)

    # Wire the wrong cache onto A.
    a.kv_cache = b.kv_cache

    with pytest.raises(RuntimeError):
        engine.decode(a)


# --------------------------------------------------
# 5. Generated token tracking
# --------------------------------------------------


def test_generated_tokens_grow_by_one_per_decode():
    model, _ = make_model()
    _, engine = make_engine(model)

    request = engine.create_request("1 2 3 4", max_new_tokens=6)

    engine.prefill(request)

    assert request.num_generated == 1
    assert request.num_cached_tokens == 4

    step = 1

    while request.status == RequestStatus.DECODING:
        engine.decode(request)
        step += 1

        assert request.num_generated == step
        assert request.position == 4 + step
        assert request.num_cached_tokens == request.position - 1

    assert request.num_generated == 6


# --------------------------------------------------
# 6. Completion conditions
# --------------------------------------------------


def test_request_finishes_on_eos():
    request = InferenceRequest(
        prompt="", input_tokens=[1, 2], max_new_tokens=10, eos_token_id=7
    )

    request.start_prefill()
    request.append_token(3)
    request.append_token(7)

    assert request.status == RequestStatus.FINISHED
    assert request.finish_reason == FinishReason.EOS
    assert request.generated_tokens == [3, 7]


def test_eos_in_prompt_does_not_stop():
    request = InferenceRequest(
        prompt="", input_tokens=[7, 7], max_new_tokens=3, eos_token_id=7
    )

    assert request.check_stop() is None


def test_request_finishes_on_max_new_tokens():
    request = InferenceRequest(prompt="", input_tokens=[1], max_new_tokens=3)

    request.start_prefill()

    for token in (4, 5, 6):
        assert not request.is_finished
        request.append_token(token)

    assert request.finish_reason == FinishReason.MAX_NEW_TOKENS
    assert request.num_generated == 3


def test_request_finishes_on_max_seq_len():
    request = InferenceRequest(
        prompt="", input_tokens=[1, 2, 3], max_new_tokens=10, max_seq_len=5
    )

    request.start_prefill()
    request.append_token(4)
    request.append_token(5)

    assert request.finish_reason == FinishReason.MAX_SEQ_LEN
    assert request.position == 5


def test_zero_max_new_tokens_finishes_without_prefill():
    model, _ = make_model()
    _, engine = make_engine(model)

    request = engine.create_request("1 2 3", max_new_tokens=0)

    engine.prefill(request)

    assert request.status == RequestStatus.FINISHED
    assert request.finish_reason == FinishReason.MAX_NEW_TOKENS
    assert request.kv_cache is None
    assert request.generated_tokens == []


def test_engine_eos_completion():
    model, config = make_model()
    _, reference_engine = make_engine(model)

    prompt = "3 1 4 1"
    reference = reference_engine.generate(prompt, max_new_tokens=15)["token_ids"]

    eos_token_id = reference[4 + 5]
    first_occurrence = reference.index(eos_token_id, 4)

    _, engine = make_engine(model, eos_token_id=eos_token_id)

    result = engine.generate(prompt, max_new_tokens=15)
    request = result["request"]

    assert request.finish_reason == FinishReason.EOS
    assert result["stats"]["finish_reason"] == "eos"
    assert request.all_tokens == reference[:first_occurrence + 1]


# --------------------------------------------------
# 7. Engine integration: identical to Phase 3 naive
# --------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_request_engine_matches_naive_runner(seed):
    model, config = make_model(seed=seed)
    runner, engine = make_engine(model)

    torch.manual_seed(100 + seed)
    prompt = " ".join(str(i) for i in torch.randint(0, config.vocab_size, (5,)).tolist())

    naive = runner.generate(prompt, max_new_tokens=25)
    result = engine.generate(prompt, max_new_tokens=25)

    assert result["token_ids"] == naive["token_ids"]
    assert result["text"] == naive["text"]
    assert result["request"].prompt == prompt
    assert result["request"].finish_reason == FinishReason.MAX_NEW_TOKENS


def test_per_request_sampling_params_override_engine_default():
    """
    A request's own SamplingParams must be what's used, not the
    engine's default sampler.
    """

    model, _ = make_model()
    penalized = Sampler(greedy=True, repetition_penalty=1.5, repeat_ngram_size=2)

    naive_runner, _ = make_engine(model, sampler=penalized)
    _, engine = make_engine(model, sampler=Sampler(greedy=True))

    prompt = "3 14 15 9 26"

    naive = naive_runner.generate(prompt, max_new_tokens=20)
    result = engine.generate(
        prompt,
        max_new_tokens=20,
        sampling_params=SamplingParams.from_sampler(penalized),
    )

    assert result["token_ids"] == naive["token_ids"]
    assert result["request"].sampling_params.repetition_penalty == 1.5
