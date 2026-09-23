"""
Phase 8 — FIFO scheduler.

The scheduler only decides which request runs next; it never touches
the model. The first half of this file tests it with bare
InferenceRequest objects (no model at all). The second half checks
InferenceEngine drives execution through it and still produces the
Phase 3 naive / Phase 6 tokens.
"""

import pytest
import torch

import src.inference.scheduler as scheduler_module
from configs.v1 import ModelConfig
from src.inference.inference_engine import InferenceEngine
from src.inference.model_runner import ModelRunner
from src.inference.request import FinishReason, InferenceRequest, RequestStatus
from src.inference.sampler import Sampler
from src.inference.scheduler import Scheduler
from src.model.model import V1LanguageModel


def make_request(request_id, tokens=(1, 2, 3), max_new_tokens=5):
    return InferenceRequest(
        prompt="",
        input_tokens=list(tokens),
        max_new_tokens=max_new_tokens,
        request_id=request_id,
    )


def finish(request):
    """Walk a request through its lifecycle without a model."""

    request.start_prefill()

    while not request.is_finished:
        request.append_token(0)


# --------------------------------------------------
# 1. Empty scheduler
# --------------------------------------------------


def test_empty_scheduler():
    scheduler = Scheduler()

    assert scheduler.next() is None
    assert not scheduler.has_work()
    assert scheduler.num_waiting == 0
    assert scheduler.num_running == 0


# --------------------------------------------------
# 2. Single request
# --------------------------------------------------


def test_single_request():
    scheduler = Scheduler()
    a = make_request("A")

    scheduler.add(a)

    assert scheduler.has_work()
    assert scheduler.num_waiting == 1
    assert a in scheduler

    assert scheduler.next() is a
    assert scheduler.num_waiting == 0
    assert scheduler.num_running == 1
    assert scheduler.next() is None


# --------------------------------------------------
# 3. FIFO order
# --------------------------------------------------


def test_fifo_order():
    scheduler = Scheduler()
    a, b, c = make_request("A"), make_request("B"), make_request("C")

    for request in (a, b, c):
        scheduler.add(request)

    assert scheduler.next() is a
    assert scheduler.next() is b
    assert scheduler.next() is c
    assert scheduler.next() is None


def test_fifo_order_with_interleaved_adds():
    scheduler = Scheduler()
    a, b, c = make_request("A"), make_request("B"), make_request("C")

    scheduler.add(a)
    scheduler.add(b)
    assert scheduler.next() is a

    scheduler.add(c)
    assert scheduler.next() is b
    assert scheduler.next() is c


# --------------------------------------------------
# 4. Finished requests are never selected
# --------------------------------------------------


def test_finished_request_is_not_selected():
    scheduler = Scheduler()
    a, b = make_request("A"), make_request("B")

    scheduler.add(a)
    scheduler.add(b)

    # A finishes (e.g. is aborted by the client) while still queued.
    a.abort()

    assert scheduler.num_waiting == 1
    assert scheduler.next() is b
    assert scheduler.next() is None
    assert a not in scheduler


def test_cannot_add_non_waiting_request():
    scheduler = Scheduler()

    done = make_request("done")
    finish(done)

    with pytest.raises(ValueError):
        scheduler.add(done)

    decoding = make_request("decoding")
    decoding.start_prefill()
    decoding.append_token(0)

    with pytest.raises(ValueError):
        scheduler.add(decoding)


# --------------------------------------------------
# 5. Every waiting request eventually runs
# --------------------------------------------------


def test_all_waiting_requests_get_scheduled():
    scheduler = Scheduler()
    requests = [make_request(f"req_{i}") for i in range(10)]

    for request in requests:
        scheduler.add(request)

    order = []

    while scheduler.has_work():
        request = scheduler.next()
        finish(request)
        scheduler.complete(request)
        order.append(request.request_id)

    assert order == [r.request_id for r in requests]
    assert all(r.is_finished for r in requests)


# --------------------------------------------------
# 6. Duplicates
# --------------------------------------------------


def test_duplicate_add_raises():
    scheduler = Scheduler()
    a = make_request("A")

    scheduler.add(a)

    with pytest.raises(ValueError):
        scheduler.add(a)

    assert scheduler.num_waiting == 1


def test_duplicate_add_while_running_raises():
    scheduler = Scheduler()
    a = make_request("A")

    scheduler.add(a)
    assert scheduler.next() is a

    with pytest.raises(ValueError):
        scheduler.add(a)


def test_distinct_request_with_same_id_is_rejected():
    scheduler = Scheduler()

    scheduler.add(make_request("A"))

    with pytest.raises(ValueError):
        scheduler.add(make_request("A"))


# --------------------------------------------------
# 7. Lifecycle leaves no stale state
# --------------------------------------------------


def test_lifecycle_leaves_no_stale_requests():
    scheduler = Scheduler()
    a = make_request("A")

    scheduler.add(a)
    assert a.status == RequestStatus.WAITING
    assert list(scheduler.waiting) == [a]

    selected = scheduler.next()
    assert selected is a
    assert list(scheduler.waiting) == []
    assert "A" in scheduler.running

    finish(a)
    scheduler.complete(a)

    assert scheduler.running == {}
    assert list(scheduler.waiting) == []
    assert not scheduler.has_work()
    assert a not in scheduler


def test_complete_requires_finished_running_request():
    scheduler = Scheduler()
    a = make_request("A")

    with pytest.raises(ValueError):
        scheduler.complete(a)

    scheduler.add(a)
    scheduler.next()

    with pytest.raises(ValueError):
        scheduler.complete(a)


def test_abort_waiting_and_running():
    scheduler = Scheduler()
    a, b = make_request("A"), make_request("B")

    scheduler.add(a)
    scheduler.add(b)

    scheduler.abort(b)
    assert b.status == RequestStatus.ABORTED
    assert b not in scheduler

    assert scheduler.next() is a
    scheduler.abort(a)

    assert a.finish_reason == FinishReason.ABORTED
    assert not scheduler.has_work()


def test_scheduler_is_independent_of_model_execution():
    assert not hasattr(scheduler_module, "torch")
    assert not hasattr(scheduler_module, "ModelRunner")
    assert not hasattr(scheduler_module, "KVCache")


# --------------------------------------------------
# Engine integration
# --------------------------------------------------


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


def make_model(seed: int = 0):
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

    model = V1LanguageModel(config)
    model.eval()

    return model


def make_engine(model):
    tokenizer = IdTokenizer()
    sampler = Sampler(greedy=True)

    runner = ModelRunner(model, tokenizer, sampler, device="cpu")
    engine = InferenceEngine(runner, tokenizer, sampler, device="cpu")

    return runner, engine


PROMPTS = ["1 2 3", "9 8 7 6 5", "42", "10 20 30 40"]


def test_engine_runs_submitted_requests_in_fifo_order():
    model = make_model()
    runner, engine = make_engine(model)

    requests = [engine.create_request(p, max_new_tokens=10) for p in PROMPTS]

    for request in requests:
        engine.submit(request)

    assert engine.scheduler.num_waiting == len(PROMPTS)

    results = engine.run_until_complete()

    assert [r["request"].request_id for r in results] == [r.request_id for r in requests]
    assert all(r.status == RequestStatus.FINISHED for r in requests)
    assert not engine.has_work()

    for prompt, result in zip(PROMPTS, results):
        naive = runner.generate(prompt, max_new_tokens=10)
        assert result["token_ids"] == naive["token_ids"]


def test_engine_step_executes_one_request_at_a_time():
    model = make_model()
    _, engine = make_engine(model)

    a = engine.submit(engine.create_request("1 2 3", max_new_tokens=4))
    b = engine.submit(engine.create_request("4 5 6", max_new_tokens=4))

    first = engine.step()

    assert first["request"] is a
    assert a.is_finished
    assert b.status == RequestStatus.WAITING
    assert engine.scheduler.num_running == 0

    second = engine.step()

    assert second["request"] is b
    assert engine.step() is None


def test_engine_skips_request_aborted_while_queued():
    model = make_model()
    _, engine = make_engine(model)

    a = engine.submit(engine.create_request("1 2 3", max_new_tokens=4))
    b = engine.submit(engine.create_request("4 5 6", max_new_tokens=4))

    engine.scheduler.abort(a)

    results = engine.run_until_complete()

    assert [r["request"] for r in results] == [b]
    assert a.status == RequestStatus.ABORTED
    assert a.kv_cache is None


def test_generate_still_matches_naive_through_scheduler():
    model = make_model(seed=3)
    runner, engine = make_engine(model)

    prompt = "5 17 33 2 60"

    naive = runner.generate(prompt, max_new_tokens=25)
    result = engine.generate(prompt, max_new_tokens=25)

    assert result["token_ids"] == naive["token_ids"]
    assert result["text"] == naive["text"]
    assert not engine.has_work()


def test_generate_behind_queued_requests_keeps_their_results():
    model = make_model()
    _, engine = make_engine(model)

    queued = engine.submit(engine.create_request("7 7 7", max_new_tokens=3))

    result = engine.generate("1 2 3", max_new_tokens=3)

    # FIFO: the earlier request ran first, and its result is kept.
    assert queued.is_finished
    assert result["request"].prompt == "1 2 3"
    assert engine.pop_result(queued.request_id)["request"] is queued


def test_failed_execution_does_not_leave_request_running():
    model = make_model()
    _, engine = make_engine(model)

    def boom(request):
        raise RuntimeError("boom")

    engine.execute = boom

    a = engine.submit(engine.create_request("1 2 3", max_new_tokens=3))

    with pytest.raises(RuntimeError):
        engine.step()

    assert a.status == RequestStatus.ABORTED
    assert not engine.has_work()
