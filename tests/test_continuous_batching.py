"""
Phase 11 — continuous batching tests.

ContinuousBatchingEngine.step() is one serving-loop iteration:
ADMIT -> PREFILL (newcomers) -> DECODE (already active) -> RETIRE.
These tests drive it step by step with a tiny real model and check:

    1. initial admission respects max_batch_size (FIFO)
    2. a finished request's slot is refilled on the next step
    3. its KV blocks are released immediately
    4. a newcomer reuses those blocks
    5. requests with different generation lengths finish independently
    6. prefill of a newcomer coexists with decode of active requests
    7. the recorded batch membership of every step is exactly right
    8. every request's tokens == running it alone (and == naive Phase 3)

plus admission under a tight KV pool, OOM without deadlock, error
cleanup, generate() compatibility and the CUDA path.
"""

import pytest
import torch

from configs.v1 import ModelConfig
from src.inference.continuous_batching import ContinuousBatchingEngine
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache_manager import KVCacheManager
from src.inference.model_runner import ModelRunner
from src.inference.request import FinishReason, RequestStatus
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


def make_model(seed=0, device="cpu"):
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


def make_engines(model, max_batch_size=2, num_blocks=None, block_size=4, device="cpu"):
    """A continuous engine plus a plain engine for solo reference runs."""

    tokenizer = IdTokenizer()
    sampler = Sampler(greedy=True)
    runner = ModelRunner(model, tokenizer, sampler, device=device)

    manager = KVCacheManager.for_model(
        model, num_blocks=num_blocks, block_size=block_size, device=device
    )
    continuous = ContinuousBatchingEngine(
        runner, tokenizer, sampler, device=device,
        kv_cache_manager=manager, max_batch_size=max_batch_size,
    )
    solo = InferenceEngine(runner, tokenizer, sampler, device=device)

    return runner, continuous, solo, manager


def prompt_of(length, offset=0):
    return " ".join(str((offset + 7 * i) % 64) for i in range(length))


def submit_all(engine, specs):
    """specs: list of (prompt, max_new_tokens). Returns requests."""

    return [
        engine.submit(engine.create_request(p, max_new_tokens=n, request_id=rid))
        for rid, (p, n) in specs.items()
    ]


def ids(requests):
    return [r.request_id for r in requests]


# --------------------------------------------------
# 1. Initial admission
# --------------------------------------------------


def test_initial_admission_respects_max_batch_size():
    _, engine, _, _ = make_engines(make_model(), max_batch_size=2)

    submit_all(engine, {
        "A": (prompt_of(4, 1), 5),
        "B": (prompt_of(5, 2), 5),
        "C": (prompt_of(6, 3), 5),
        "D": (prompt_of(3, 4), 5),
    })

    engine.step()

    assert ids(engine.active) == ["A", "B"]
    assert ids(engine.waiting) == ["C", "D"]
    assert engine.history[0].admitted == ["A", "B"]
    assert engine.history[0].decoded == []


# --------------------------------------------------
# 2. Completion replacement
# --------------------------------------------------


def test_finished_request_is_replaced_next_step():
    _, engine, _, _ = make_engines(make_model(), max_batch_size=2)

    submit_all(engine, {
        "A": (prompt_of(4, 1), 2),     # prefill token + 1 decode
        "B": (prompt_of(5, 2), 10),
        "C": (prompt_of(6, 3), 10),
        "D": (prompt_of(3, 4), 10),
    })

    engine.step()                      # prefill A, B
    finished = engine.step()           # decode A, B -> A done

    assert [r["request"].request_id for r in finished] == ["A"]
    assert ids(engine.active) == ["B"]

    engine.step()                      # C takes A's slot

    assert engine.history[-1].admitted == ["C"]
    assert engine.history[-1].decoded == ["B"]
    assert ids(engine.active) == ["B", "C"]
    assert ids(engine.waiting) == ["D"]


# --------------------------------------------------
# 3 & 4. KV release and reuse
# --------------------------------------------------


def test_kv_blocks_released_on_completion():
    _, engine, _, manager = make_engines(make_model(), max_batch_size=2, block_size=4)

    a, b = submit_all(engine, {
        "A": (prompt_of(6), 2),
        "B": (prompt_of(5), 10),
    })

    engine.step()

    a_blocks = a.block_ids
    assert a_blocks and manager.has("A")
    free_before = manager.num_free_blocks

    engine.step()                      # A finishes

    assert a.status == RequestStatus.FINISHED
    assert not manager.has("A")
    assert a.block_ids == []
    assert set(a_blocks) <= set(manager.free_block_ids)
    assert manager.num_free_blocks >= free_before + len(a_blocks) - 1  # B may grow by one


def test_newcomer_reuses_released_blocks():
    # Pool is exactly big enough for two requests at a time, so C can
    # only run on blocks A gave back.
    _, engine, solo, manager = make_engines(
        make_model(), max_batch_size=2, num_blocks=5, block_size=4
    )

    specs = {
        "A": (prompt_of(6, 1), 2),     # peaks at 7 tokens  -> 2 blocks
        "B": (prompt_of(5, 2), 6),     # peaks at 10 tokens -> 3 blocks
        "C": (prompt_of(6, 3), 2),
    }
    a, b, c = submit_all(engine, specs)

    engine.step()
    a_blocks = set(a.block_ids)
    engine.step()                      # A finishes, its blocks go free

    assert a_blocks <= set(manager.free_block_ids)

    engine.step()                      # C admitted

    assert set(c.block_ids) & a_blocks
    assert not (set(c.block_ids) & set(b.block_ids))

    engine.run_until_complete()

    for rid, request in zip(specs, (a, b, c)):
        prompt, n = specs[rid]
        assert request.all_tokens == solo.generate(prompt, max_new_tokens=n)["token_ids"]

    assert manager.num_free_blocks == manager.num_blocks


# --------------------------------------------------
# 5. Different generation lengths
# --------------------------------------------------


def test_different_generation_lengths_finish_independently():
    _, engine, solo, _ = make_engines(make_model(), max_batch_size=3)

    specs = {
        "A": (prompt_of(4, 1), 2),
        "B": (prompt_of(5, 2), 10),
        "C": (prompt_of(6, 3), 4),
    }
    requests = submit_all(engine, specs)

    finish_step = {}

    while engine.has_work():
        for result in engine.step():
            finish_step[result["request"].request_id] = engine.history[-1].step

    # Admitted together at step 1; finishes at step = max_new_tokens.
    assert finish_step == {"A": 2, "C": 4, "B": 10}

    for request in requests:
        prompt, n = specs[request.request_id]
        assert request.num_generated == n
        assert request.all_tokens == solo.generate(prompt, max_new_tokens=n)["token_ids"]


# --------------------------------------------------
# 6. Prefill and decode coexist
# --------------------------------------------------


def test_prefill_and_decode_in_same_step():
    _, engine, solo, _ = make_engines(make_model(), max_batch_size=3)

    specs = {"A": (prompt_of(4, 1), 8), "B": (prompt_of(9, 2), 8)}
    a, b = submit_all(engine, specs)

    engine.step()
    engine.step()

    assert a.num_generated == b.num_generated == 2

    # C arrives while A and B are mid-generation.
    c = engine.submit(engine.create_request(prompt_of(12, 3), max_new_tokens=8, request_id="C"))

    engine.step()

    record = engine.history[-1]
    assert record.admitted == ["C"]
    assert record.decoded == ["A", "B"]

    assert c.status == RequestStatus.DECODING
    assert c.num_generated == 1
    assert c.num_cached_tokens == c.prompt_len
    assert a.num_generated == b.num_generated == 3
    assert a.num_cached_tokens == a.position - 1
    assert b.num_cached_tokens == b.position - 1

    engine.run_until_complete()

    specs["C"] = (prompt_of(12, 3), 8)
    for request in (a, b, c):
        prompt, n = specs[request.request_id]
        assert request.all_tokens == solo.generate(prompt, max_new_tokens=n)["token_ids"]


# --------------------------------------------------
# 7. Batch membership at every step
# --------------------------------------------------


def test_batch_membership_matches_spec_example():
    """
    The Phase 11 spec example: max_batch_size=3,
    A=3, B=7, C=4, D=5, E=2 generated tokens.
    """

    _, engine, _, _ = make_engines(make_model(), max_batch_size=3)

    submit_all(engine, {
        "A": (prompt_of(4, 1), 3),
        "B": (prompt_of(5, 2), 7),
        "C": (prompt_of(6, 3), 4),
        "D": (prompt_of(3, 4), 5),
        "E": (prompt_of(7, 5), 2),
    })

    engine.run_until_complete()

    membership = [(h.admitted, h.decoded, h.finished, h.active_after) for h in engine.history]

    assert membership == [
        (["A", "B", "C"], [],              [],         ["A", "B", "C"]),
        ([],              ["A", "B", "C"], [],         ["A", "B", "C"]),
        ([],              ["A", "B", "C"], ["A"],      ["B", "C"]),
        (["D"],           ["B", "C"],      ["C"],      ["B", "D"]),
        (["E"],           ["B", "D"],      [],         ["B", "D", "E"]),
        ([],              ["B", "D", "E"], ["E"],      ["B", "D"]),
        ([],              ["B", "D"],      ["B"],      ["D"]),
        ([],              ["D"],           ["D"],      []),
    ]

    # Never more than max_batch_size requests in one step's batch.
    for h in engine.history:
        assert len(set(h.admitted) | set(h.decoded)) <= 3


# --------------------------------------------------
# 8. Output correctness
# --------------------------------------------------


@pytest.mark.parametrize("max_batch_size", [1, 2, 4])
def test_continuous_output_matches_independent_runs(max_batch_size):
    runner, engine, solo, manager = make_engines(make_model(seed=1), max_batch_size=max_batch_size)

    specs = {
        f"R{i}": (prompt_of(length, i), n)
        for i, (length, n) in enumerate([(4, 12), (9, 3), (2, 7), (12, 15), (6, 1), (5, 9), (3, 4)])
    }
    requests = submit_all(engine, specs)

    results = engine.run_until_complete()

    assert sorted(r["request"].request_id for r in results) == sorted(specs)

    for request in requests:
        prompt, n = specs[request.request_id]
        expected = solo.generate(prompt, max_new_tokens=n)["token_ids"]

        assert request.all_tokens == expected
        assert request.all_tokens == runner.generate(prompt, max_new_tokens=n)["token_ids"]

    assert manager.num_free_blocks == manager.num_blocks
    assert not engine.has_work()


def test_eos_finishes_request_early_and_frees_slot():
    model = make_model()
    _, reference, _, _ = make_engines(model)

    prompt_a = prompt_of(4, 1)
    trace = reference.generate(prompt_a, max_new_tokens=10)["token_ids"]
    eos = trace[4 + 2]

    tokenizer = IdTokenizer(eos_token_id=eos)
    sampler = Sampler(greedy=True)
    runner = ModelRunner(model, tokenizer, sampler, device="cpu")
    engine = ContinuousBatchingEngine(runner, tokenizer, sampler, device="cpu", max_batch_size=2)
    solo = InferenceEngine(runner, tokenizer, sampler, device="cpu")

    specs = {"A": (prompt_a, 10), "B": (prompt_of(6, 2), 10), "C": (prompt_of(5, 3), 10)}
    requests = submit_all(engine, specs)

    engine.run_until_complete()

    assert requests[0].finish_reason == FinishReason.EOS
    assert requests[0].num_generated <= 3

    for request in requests:
        prompt, n = specs[request.request_id]
        assert request.all_tokens == solo.generate(prompt, max_new_tokens=n)["token_ids"]


# --------------------------------------------------
# Memory-aware admission
# --------------------------------------------------


def test_admission_waits_for_kv_blocks_not_just_slots():
    # 4 blocks of 4 tokens. An 8-token prompt fills 2 blocks, and its
    # first decode writes token 9 -> a 3rd block. The one-step
    # lookahead therefore admits only one request at a time even
    # though there are 3 slots (admitting two would force one of them
    # out of memory on the next step).
    _, engine, solo, manager = make_engines(
        make_model(), max_batch_size=3, num_blocks=4, block_size=4
    )

    specs = {"A": (prompt_of(8, 1), 3), "B": (prompt_of(8, 2), 3), "C": (prompt_of(8, 3), 3)}
    requests = submit_all(engine, specs)

    engine.run_until_complete()

    for h in engine.history:
        assert len(h.admitted) + len(h.decoded) <= 3
        assert h.free_blocks_after >= 0

    assert engine.history[0].admitted == ["A"]
    assert all(r.finish_reason == FinishReason.MAX_NEW_TOKENS for r in requests)

    for request in requests:
        prompt, n = specs[request.request_id]
        assert request.all_tokens == solo.generate(prompt, max_new_tokens=n)["token_ids"]


def test_request_too_big_for_pool_is_aborted_without_deadlock():
    _, engine, solo, manager = make_engines(
        make_model(), max_batch_size=2, num_blocks=2, block_size=4
    )

    too_big, fits = submit_all(engine, {
        "big": (prompt_of(12), 3),
        "small": (prompt_of(3), 3),
    })

    engine.run_until_complete()

    assert too_big.finish_reason == FinishReason.OUT_OF_KV_BLOCKS
    assert fits.all_tokens == solo.generate(prompt_of(3), max_new_tokens=3)["token_ids"]
    assert manager.num_free_blocks == 2


# --------------------------------------------------
# Robustness / compatibility
# --------------------------------------------------


def test_error_during_step_releases_everything():
    _, engine, _, manager = make_engines(make_model(), max_batch_size=2)

    requests = submit_all(engine, {"A": (prompt_of(4), 5), "B": (prompt_of(5), 5)})
    engine.step()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    engine.runner.decode_batch = boom

    with pytest.raises(RuntimeError):
        engine.step()

    assert all(r.status == RequestStatus.ABORTED for r in requests)
    assert manager.num_free_blocks == manager.num_blocks
    assert not engine.has_work()


def test_generate_and_stats():
    runner, engine, _, _ = make_engines(make_model(), max_batch_size=2)

    prompt = prompt_of(7, 5)
    result = engine.generate(prompt, max_new_tokens=6)

    assert result["token_ids"] == runner.generate(prompt, max_new_tokens=6)["token_ids"]

    stats = result["stats"]
    assert stats["generated_tokens"] == 6
    assert stats["admitted_step"] == 1
    assert stats["finished_step"] == 6
    assert stats["latency_seconds"] >= stats["ttft_seconds"] >= 0


def test_static_step_batch_is_not_available():
    _, engine, _, _ = make_engines(make_model())

    with pytest.raises(NotImplementedError):
        engine.step_batch(2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_continuous_batching_on_cuda():
    runner, engine, solo, manager = make_engines(
        make_model(device="cuda"), max_batch_size=3, device="cuda"
    )

    specs = {f"R{i}": (prompt_of(3 + 2 * i, i), 4 + 3 * i) for i in range(6)}
    requests = submit_all(engine, specs)

    engine.run_until_complete()

    for request in requests:
        prompt, n = specs[request.request_id]
        assert request.all_tokens == solo.generate(prompt, max_new_tokens=n)["token_ids"]

    assert manager.num_free_blocks == manager.num_blocks
