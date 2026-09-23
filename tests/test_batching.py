"""
Phase 9 — static batching.

Several independent requests share one model forward per step. Only
execution is shared: each request keeps its own tokens, KV cache,
sampling params and stop conditions. Every test here compares a
batched result against the same request run alone (Phase 8 path).
"""

import pytest
import torch

from configs.v1 import ModelConfig
from src.inference.batch import DECODE, PREFILL, build_decode_batch, build_prefill_batch
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache import BatchedKVCache
from src.inference.model_runner import ModelRunner
from src.inference.request import FinishReason, RequestStatus, SamplingParams
from src.inference.sampler import Sampler
from src.model.model import V1LanguageModel

ATOL = 1e-4
RTOL = 1e-4

GREEDY = SamplingParams(greedy=True)


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


def make_model(seed: int = 0, device: str = "cpu"):
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

    model = V1LanguageModel(config).to(device)
    model.eval()

    return model


def make_engine(model, eos_token_id=None, device="cpu"):
    tokenizer = IdTokenizer(eos_token_id=eos_token_id)
    sampler = Sampler(greedy=True)

    runner = ModelRunner(model, tokenizer, sampler, device=device)
    engine = InferenceEngine(runner, tokenizer, sampler, device=device)

    return runner, engine


def prompt_of(length, offset=0):
    return " ".join(str((offset + 7 * i) % 64) for i in range(length))


# A = 5 tokens, B = 8 tokens, C = 12 tokens
PROMPTS = [prompt_of(5, 1), prompt_of(8, 2), prompt_of(12, 3)]


def solo_results(engine, prompts, max_new_tokens, sampling_params=None):
    return [
        engine.generate(p, max_new_tokens=n, sampling_params=sampling_params)
        for p, n in zip(prompts, max_new_tokens)
    ]


def batched_results(engine, prompts, max_new_tokens, sampling_params=None):
    requests = [
        engine.submit(
            engine.create_request(p, max_new_tokens=n, sampling_params=sampling_params)
        )
        for p, n in zip(prompts, max_new_tokens)
    ]

    results = engine.run_until_complete(max_batch_size=len(prompts))

    assert [r["request"] for r in results] == requests

    return results


# --------------------------------------------------
# Batch construction: padding + masking
# --------------------------------------------------


def test_prefill_batch_pads_and_masks():
    _, engine = make_engine(make_model())

    requests = [engine.create_request(p) for p in PROMPTS]
    batch = build_prefill_batch(requests, pad_token_id=0)

    assert batch.phase == PREFILL
    assert batch.input_ids.shape == (3, 12)
    assert batch.seq_lens == [5, 8, 12]
    assert batch.request_ids == [r.request_id for r in requests]
    assert batch.row_of(requests[1].request_id) == 1

    for row, request in enumerate(requests):
        n = request.prompt_len
        assert batch.input_ids[row, :n].tolist() == request.input_tokens
        assert batch.attention_mask[row].tolist() == [True] * n + [False] * (12 - n)
        assert (batch.input_ids[row, n:] == 0).all()

    mask = batch.model_attention_mask()
    assert mask.shape == (3, 1, 12, 12)

    # Row 0 (5 real tokens): query 3 sees keys 0..3, never padding.
    assert mask[0, 0, 3].tolist() == [True] * 4 + [False] * 8
    assert mask[0, 0, 11].tolist() == [True] * 5 + [False] * 7


def test_decode_batch_positions_and_mask():
    _, engine = make_engine(make_model())

    requests = [engine.create_request(p, max_new_tokens=5) for p in PROMPTS]
    engine.prefill_batch(requests)

    batch = build_decode_batch(requests)

    assert batch.phase == DECODE
    assert batch.input_ids.shape == (3, 1)
    assert batch.input_ids[:, 0].tolist() == [r.last_token for r in requests]
    assert batch.position_ids[:, 0].tolist() == [5, 8, 12]
    assert batch.seq_lens == [5, 8, 12]

    # 12 padded past slots + the new token.
    assert batch.attention_mask.shape == (3, 13)
    assert batch.attention_mask[0].tolist() == [True] * 5 + [False] * 7 + [True]
    assert batch.attention_mask[2].all()


def test_padding_token_value_does_not_matter():
    model = make_model()
    runner, engine = make_engine(model)

    requests = [engine.create_request(p) for p in PROMPTS]

    out_a = runner.prefill_batch(build_prefill_batch(requests, pad_token_id=0))
    out_b = runner.prefill_batch(build_prefill_batch(requests, pad_token_id=63))

    assert torch.allclose(out_a.logits, out_b.logits, atol=ATOL, rtol=RTOL)


# --------------------------------------------------
# 1. Single-request batch == Phase 8
# --------------------------------------------------


def test_single_request_batch_matches_phase8():
    model = make_model()
    runner, engine = make_engine(model)

    prompt = PROMPTS[1]

    naive = runner.generate(prompt, max_new_tokens=20)
    solo = engine.generate(prompt, max_new_tokens=20)
    [batched] = batched_results(engine, [prompt], [20])

    assert batched["token_ids"] == solo["token_ids"] == naive["token_ids"]


# --------------------------------------------------
# 2 & 3. Multiple requests, different prompt lengths
# --------------------------------------------------


def test_two_requests_match_individual_runs():
    model = make_model()
    _, engine = make_engine(model)

    prompts = PROMPTS[:2]

    solo = solo_results(engine, prompts, [15, 15])
    batched = batched_results(engine, prompts, [15, 15])

    for s, b in zip(solo, batched):
        assert b["token_ids"] == s["token_ids"]
        assert b["stats"]["batch_size"] == 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_different_prompt_lengths_match_individual_runs(seed):
    model = make_model(seed=seed)
    runner, engine = make_engine(model)

    batched = batched_results(engine, PROMPTS, [20, 20, 20])

    for prompt, result in zip(PROMPTS, batched):
        naive = runner.generate(prompt, max_new_tokens=20)
        assert result["token_ids"] == naive["token_ids"]


def test_batch_larger_than_max_batch_size_runs_in_fifo_groups():
    model = make_model()
    _, engine = make_engine(model)

    prompts = [prompt_of(n, n) for n in (3, 6, 4, 9, 2)]
    solo = solo_results(engine, prompts, [8] * 5)

    requests = [engine.submit(engine.create_request(p, max_new_tokens=8)) for p in prompts]

    results = engine.run_until_complete(max_batch_size=2)

    assert [r["request"] for r in results] == requests
    assert [r["stats"]["batch_size"] for r in results] == [2, 2, 2, 2, 1]

    for s, b in zip(solo, results):
        assert b["token_ids"] == s["token_ids"]


# --------------------------------------------------
# 4. KV-cache independence
# --------------------------------------------------


def test_kv_caches_are_independent():
    model = make_model()
    _, engine = make_engine(model)

    requests = [engine.create_request(p, max_new_tokens=6) for p in PROMPTS]
    engine.prefill_batch(requests)

    caches = [r.kv_cache for r in requests]

    assert len({id(c) for c in caches}) == 3
    assert [c.get_seq_length() for c in caches] == [5, 8, 12]

    # No two requests share KV memory. Since Phase 10 all K/V lives in
    # one KVCacheManager pool, so independence means disjoint blocks.
    block_sets = [set(r.block_ids) for r in requests]
    assert all(block_sets)
    assert not (block_sets[0] & block_sets[1])
    assert not (block_sets[0] & block_sets[2])
    assert not (block_sets[1] & block_sets[2])

    engine.decode_batch(requests)

    assert [c.get_seq_length() for c in caches] == [6, 9, 13]

    # Decoding only B leaves A and C untouched.
    before_a = requests[0].kv_cache.get(0)[0].clone()

    engine.decode_batch([requests[1]])

    assert [c.get_seq_length() for c in caches] == [6, 10, 13]
    assert torch.equal(requests[0].kv_cache.get(0)[0], before_a)


def test_batched_kv_cache_is_request_ordered_and_padded():
    model = make_model()
    _, engine = make_engine(model)

    requests = [engine.create_request(p, max_new_tokens=6) for p in PROMPTS]
    engine.prefill_batch(requests)

    batched = BatchedKVCache([r.kv_cache for r in requests])

    assert batched.seq_lens == [5, 8, 12]
    assert batched.get_seq_length() == 12

    k = batched.key_cache[0]
    assert k.shape[0] == 3

    for row, request in enumerate(requests):
        n = request.prompt_len
        own_k = request.kv_cache.get(0)[0]
        assert torch.equal(k[row:row + 1, :, :n], own_k)
        assert (k[row, :, n:] == 0).all()


def test_shared_cache_in_one_batch_is_rejected():
    model = make_model()
    _, engine = make_engine(model)

    a = engine.create_request(PROMPTS[0], max_new_tokens=5)
    b = engine.create_request(PROMPTS[0], max_new_tokens=5)

    engine.prefill_batch([a, b])
    b.kv_cache = a.kv_cache

    with pytest.raises(RuntimeError):
        engine.decode_batch([a, b])


# --------------------------------------------------
# 5. Batched prefill logits == individual
# --------------------------------------------------


def test_batched_prefill_matches_individual():
    model = make_model()
    runner, engine = make_engine(model)

    requests = [engine.create_request(p) for p in PROMPTS]
    batched = runner.prefill_batch(build_prefill_batch(requests))

    for row, request in enumerate(requests):
        solo = runner.prefill(torch.tensor([request.input_tokens]))

        assert torch.allclose(batched.logits[row:row + 1], solo.logits, atol=ATOL, rtol=RTOL)

        for layer_idx in range(model.config.num_layers):
            bk, bv = batched.kv_caches[row].get(layer_idx)
            sk, sv = solo.kv_cache.get(layer_idx)

            assert bk.shape == sk.shape
            assert torch.allclose(bk, sk, atol=ATOL, rtol=RTOL)
            assert torch.allclose(bv, sv, atol=ATOL, rtol=RTOL)


# --------------------------------------------------
# 6. Batched decode logits == individual
# --------------------------------------------------


def test_batched_decode_matches_individual():
    model = make_model()
    runner, engine = make_engine(model)

    requests = [engine.create_request(p, max_new_tokens=10) for p in PROMPTS]
    engine.prefill_batch(requests)

    # Solo references: same prompt + same first token through the
    # single-sequence runner API.
    solo_caches = []
    for request in requests:
        solo = runner.prefill(torch.tensor([request.input_tokens]))
        solo_caches.append(solo.kv_cache)

    for step in range(4):
        batch = build_decode_batch(requests)
        tokens = [r.last_token for r in requests]

        batched = runner.decode_batch(batch)

        for row, request in enumerate(requests):
            solo = runner.decode(torch.tensor([[tokens[row]]]), solo_caches[row])

            assert torch.allclose(
                batched.logits[row:row + 1], solo.logits, atol=ATOL, rtol=RTOL
            ), f"decode step {step}, row {row}"

            request.append_token(int(batched.logits[row].argmax()))

        assert [r.num_cached_tokens for r in requests] == [
            c.get_seq_length() for c in solo_caches
        ]


# --------------------------------------------------
# 7. Different sampling params in one batch
# --------------------------------------------------


def test_each_request_keeps_its_own_sampling_params():
    model = make_model()
    _, engine = make_engine(model)

    greedy = SamplingParams(greedy=True)
    stochastic = SamplingParams(temperature=0.8, top_k=10)
    penalized = SamplingParams(greedy=True, repetition_penalty=1.5, repeat_ngram_size=2)

    params = [greedy, stochastic, penalized]

    solo = [
        engine.generate(p, max_new_tokens=15, sampling_params=sp)
        for p, sp in zip(PROMPTS, params)
    ]

    requests = [
        engine.submit(engine.create_request(p, max_new_tokens=15, sampling_params=sp))
        for p, sp in zip(PROMPTS, params)
    ]

    torch.manual_seed(0)
    engine.run_until_complete(max_batch_size=3)

    assert [r.sampling_params for r in requests] == params

    # Deterministic rows must be unaffected by sharing a batch with a
    # stochastic one.
    assert requests[0].all_tokens == solo[0]["token_ids"]
    assert requests[2].all_tokens == solo[2]["token_ids"]

    assert requests[1].num_generated == 15
    assert requests[1].status == RequestStatus.FINISHED


def test_stochastic_row_is_reproducible_with_seed():
    model = make_model()
    _, engine = make_engine(model)

    stochastic = SamplingParams(temperature=1.0, top_k=5)

    def run():
        torch.manual_seed(1234)
        reqs = [
            engine.submit(engine.create_request(p, max_new_tokens=10, sampling_params=stochastic))
            for p in PROMPTS
        ]
        engine.run_until_complete(max_batch_size=3)
        return [r.all_tokens for r in reqs]

    assert run() == run()


# --------------------------------------------------
# 8. Requests finish independently
# --------------------------------------------------


def test_requests_with_different_limits_finish_independently():
    model = make_model()
    _, engine = make_engine(model)

    limits = [3, 12, 7]

    solo = solo_results(engine, PROMPTS, limits)
    batched = batched_results(engine, PROMPTS, limits)

    for s, b, limit in zip(solo, batched, limits):
        assert b["token_ids"] == s["token_ids"]
        assert b["request"].num_generated == limit
        assert b["request"].finish_reason == FinishReason.MAX_NEW_TOKENS

    # The batch ran as long as its longest member.
    assert batched[0]["stats"]["decode_steps"] == max(limits) - 1


def test_eos_in_one_request_does_not_stop_others():
    model = make_model()
    _, reference = make_engine(model)

    ref_a = reference.generate(PROMPTS[0], max_new_tokens=20)["token_ids"]
    eos_token_id = ref_a[PROMPTS[0].count(" ") + 1 + 2]

    _, engine = make_engine(model, eos_token_id=eos_token_id)

    solo = solo_results(engine, PROMPTS, [20, 20, 20])
    batched = batched_results(engine, PROMPTS, [20, 20, 20])

    for s, b in zip(solo, batched):
        assert b["token_ids"] == s["token_ids"]
        assert b["request"].finish_reason == s["request"].finish_reason

    assert batched[0]["request"].finish_reason == FinishReason.EOS
    assert batched[0]["request"].num_generated <= 3


def test_zero_max_new_tokens_row_finishes_without_prefill():
    model = make_model()
    _, engine = make_engine(model)

    solo = engine.generate(PROMPTS[1], max_new_tokens=6)

    batched = batched_results(engine, PROMPTS[:2], [0, 6])

    assert batched[0]["request"].kv_cache is None
    assert batched[0]["request"].generated_tokens == []
    assert batched[1]["token_ids"] == solo["token_ids"]


def test_failed_batch_does_not_leave_requests_running():
    model = make_model()
    _, engine = make_engine(model)

    def boom(requests):
        raise RuntimeError("boom")

    engine.execute_batch = boom

    requests = [engine.submit(engine.create_request(p)) for p in PROMPTS]

    with pytest.raises(RuntimeError):
        engine.step_batch(3)

    assert all(r.status == RequestStatus.ABORTED for r in requests)
    assert not engine.has_work()


# --------------------------------------------------
# CUDA path
# --------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batching_matches_individual_on_cuda():
    model = make_model(device="cuda")
    runner, engine = make_engine(model, device="cuda")

    requests = [engine.create_request(p) for p in PROMPTS]
    batched = runner.prefill_batch(build_prefill_batch(requests, device="cuda"))

    for row, request in enumerate(requests):
        solo = runner.prefill(torch.tensor([request.input_tokens], device="cuda"))
        assert torch.allclose(batched.logits[row:row + 1], solo.logits, atol=1e-3, rtol=1e-3)

    solo = solo_results(engine, PROMPTS, [15, 15, 15])
    batched = batched_results(engine, PROMPTS, [15, 15, 15])

    for s, b in zip(solo, batched):
        assert b["token_ids"] == s["token_ids"]
