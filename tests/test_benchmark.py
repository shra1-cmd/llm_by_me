"""
Phase 12 — benchmark framework validation.

Before trusting any performance number, check the harness itself:

    - the on_token hooks fire exactly once per generated token, in
      the naive runner and in every engine path
    - paged_kv=False (the honest Phase 6 contiguous-KV baseline)
      produces the same tokens as the paged default
    - every benchmark mode emits identical tokens for the same
      workload (the correctness gate for comparing their speed)
    - batched modes really run requests concurrently (identical tokens
      alone can't catch a batch silently shrinking to size 1)
    - per-request metrics are internally consistent:
          0 < TTFT <= latency <= total,  #ITL gaps = generated - 1,
          token counts add up
    - warm-up runs are discarded, measured iterations are kept, and
      mean / median / min / max are computed correctly

Uses a tiny random model on CPU, so it checks the machinery, not the
real model's speed.
"""

import pytest
import torch

from configs.v1 import ModelConfig
from src.inference.benchmark import (
    MODES,
    BenchmarkContext,
    BenchmarkRequest,
    benchmark_mode,
    outputs_match,
    run_mode,
    summarize,
)
from src.inference.continuous_batching import ContinuousBatchingEngine
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache import KVCache
from src.inference.model_runner import ModelRunner
from src.inference.sampler import Sampler
from src.model.model import V1LanguageModel


class IdTokenizer:
    def encode(self, text):
        return [int(t) for t in text.split()]

    def decode(self, token_ids):
        return " ".join(str(i) for i in token_ids)

    def token_to_id(self, token):
        return None


def make_model(seed=0):
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


def prompt_of(length, offset=0):
    return " ".join(str((offset + 7 * i) % 64) for i in range(length))


WORKLOAD = [
    BenchmarkRequest("R0", prompt_of(4, 1), 6),
    BenchmarkRequest("R1", prompt_of(9, 2), 3),
    BenchmarkRequest("R2", prompt_of(2, 3), 8),
    BenchmarkRequest("R3", prompt_of(6, 4), 5),
    BenchmarkRequest("R4", prompt_of(5, 5), 2),
]


@pytest.fixture(scope="module")
def ctx():
    return BenchmarkContext(
        model=make_model(),
        tokenizer=IdTokenizer(),
        device="cpu",
        batch_size=2,
        block_size=4,
    )


# --------------------------------------------------
# Hooks
# --------------------------------------------------


def test_naive_on_token_fires_once_per_token():
    model = make_model()
    runner = ModelRunner(model, IdTokenizer(), Sampler(greedy=True), device="cpu")

    seen = []
    result = runner.generate(prompt_of(4), max_new_tokens=7, on_token=seen.append)

    assert seen == result["token_ids"][4:]


@pytest.mark.parametrize("engine_kind", ["sequential", "static", "continuous"])
def test_request_on_token_fires_once_per_token(engine_kind):
    model = make_model()
    tokenizer = IdTokenizer()
    sampler = Sampler(greedy=True)
    runner = ModelRunner(model, tokenizer, sampler, device="cpu")

    if engine_kind == "continuous":
        engine = ContinuousBatchingEngine(runner, tokenizer, sampler, device="cpu", max_batch_size=2)
    else:
        engine = InferenceEngine(runner, tokenizer, sampler, device="cpu")

    seen = {}
    requests = []

    for spec in WORKLOAD:
        request = engine.create_request(spec.prompt, max_new_tokens=spec.max_new_tokens)
        request.on_token = lambda req, tok: seen.setdefault(req.request_id, []).append(tok)
        requests.append(engine.submit(request))

    if engine_kind == "static":
        engine.run_until_complete(max_batch_size=2)
    else:
        engine.run_until_complete()

    for request in requests:
        assert seen[request.request_id] == request.generated_tokens


# --------------------------------------------------
# paged_kv=False baseline
# --------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 3])
def test_contiguous_kv_engine_matches_paged(batch_size):
    model = make_model()
    tokenizer = IdTokenizer()
    sampler = Sampler(greedy=True)
    runner = ModelRunner(model, tokenizer, sampler, device="cpu")

    def run(paged):
        engine = InferenceEngine(runner, tokenizer, sampler, device="cpu", paged_kv=paged)
        requests = [
            engine.submit(engine.create_request(s.prompt, max_new_tokens=s.max_new_tokens))
            for s in WORKLOAD
        ]
        engine.run_until_complete(max_batch_size=batch_size)
        return requests, engine

    contiguous, contiguous_engine = run(paged=False)
    paged, _ = run(paged=True)

    assert [r.all_tokens for r in contiguous] == [r.all_tokens for r in paged]
    assert all(isinstance(r.kv_cache, KVCache) for r in contiguous)
    # The contiguous path never touches the block pool.
    assert contiguous_engine.kv_cache_manager.request_ids == []

    for spec, request in zip(WORKLOAD, contiguous):
        assert request.all_tokens == runner.generate(spec.prompt, max_new_tokens=spec.max_new_tokens)["token_ids"]


def test_contiguous_kv_batches_are_not_limited_by_block_pool():
    """
    Regression: with paged_kv=False the engine must not apply KV-pool
    admission, or a small placeholder pool silently shrinks every
    static batch to one request (same tokens, wrong performance).
    """

    model = make_model()
    tokenizer = IdTokenizer()
    sampler = Sampler(greedy=True)
    runner = ModelRunner(model, tokenizer, sampler, device="cpu")

    from src.inference.kv_cache_manager import KVCacheManager

    engine = InferenceEngine(
        runner, tokenizer, sampler, device="cpu", paged_kv=False,
        kv_cache_manager=KVCacheManager.for_model(model, num_blocks=1),
    )

    for spec in WORKLOAD:
        engine.submit(engine.create_request(spec.prompt, max_new_tokens=spec.max_new_tokens))

    results = engine.step_batch(3)

    assert len(results) == 3
    assert all(r["stats"]["batch_size"] == 3 for r in results)


@pytest.mark.parametrize("mode", ["batch", "batch_paged", "continuous"])
def test_batched_modes_really_batch(ctx, mode):
    """Tokens can match even if batching silently degrades to batch
    size 1, so check concurrency directly: with batch_size=2 the first
    two requests must be generating at the same time."""

    result = run_mode(ctx, WORKLOAD, mode)
    r0, r1 = result.traces["R0"], result.traces["R1"]

    # R1's first token arrives before R0 has finished.
    assert r1.token_times[0] < r0.token_times[-1]


# --------------------------------------------------
# Every mode: same tokens, consistent metrics
# --------------------------------------------------


def test_all_modes_emit_identical_tokens(ctx):
    results = {mode: run_mode(ctx, WORKLOAD, mode) for mode in MODES}

    reference = results["naive"].outputs()

    for mode, result in results.items():
        assert result.outputs() == reference, mode


@pytest.mark.parametrize("mode", list(MODES))
def test_metrics_are_consistent(ctx, mode):
    result = run_mode(ctx, WORKLOAD, mode)
    metrics = result.metrics()

    for spec in WORKLOAD:
        trace = result.traces[spec.request_id]

        assert trace.generated_tokens == spec.max_new_tokens
        assert len(trace.token_ids) == trace.prompt_tokens + spec.max_new_tokens
        assert len(trace.itls()) == spec.max_new_tokens - 1
        assert all(gap >= 0 for gap in trace.itls())

        ttft = trace.ttft(result.t0)
        latency = trace.latency(result.t0)

        assert 0 < ttft <= latency <= result.total_seconds

    assert metrics["generated_tokens"] == sum(s.max_new_tokens for s in WORKLOAD)
    assert metrics["prompt_tokens"] == sum(len(s.prompt.split()) for s in WORKLOAD)
    assert metrics["total_tokens"] == metrics["prompt_tokens"] + metrics["generated_tokens"]
    assert metrics["tokens_per_s"] > 0
    assert metrics["peak_mb"] is None          # CPU run

    paged = mode in ("kv_cache_paged", "batch_paged", "continuous")
    assert (metrics["kv_pool_mb"] > 0) == paged


def test_sequential_ttft_includes_queueing(ctx):
    """In one-at-a-time modes, a later request can't start before an
    earlier one has finished."""

    result = run_mode(ctx, WORKLOAD, "kv_cache")
    traces = [result.traces[s.request_id] for s in WORKLOAD]

    for earlier, later in zip(traces, traces[1:]):
        assert later.token_times[0] > earlier.token_times[-1]


# --------------------------------------------------
# Iterations / aggregation
# --------------------------------------------------


def test_benchmark_mode_keeps_only_measured_iterations(ctx):
    report = benchmark_mode(ctx, WORKLOAD, "continuous", warmup=2, iterations=3)

    assert len(report.runs) == 3
    assert report.deterministic

    stats = report.stats()
    for name in ("total_s", "tokens_per_s", "ttft_mean_s", "itl_mean_ms"):
        s = stats[name]
        assert s["min"] <= s["median"] <= s["max"]
        assert s["min"] <= s["mean"] <= s["max"]

    assert stats["generated_tokens"] == sum(s.max_new_tokens for s in WORKLOAD)


def test_outputs_match_detects_mismatch(ctx):
    a = benchmark_mode(ctx, WORKLOAD, "naive", warmup=0, iterations=1)
    b = benchmark_mode(ctx, WORKLOAD, "batch", warmup=0, iterations=1)

    assert outputs_match([a, b])

    b.runs[0].traces["R0"].token_ids = b.runs[0].traces["R0"].token_ids[:-1] + [63]
    assert not outputs_match([a, b])


def test_summarize():
    assert summarize([3.0, 1.0, 2.0, 10.0]) == {
        "mean": 4.0,
        "median": 2.5,
        "min": 1.0,
        "max": 10.0,
    }


def test_unknown_mode_rejected(ctx):
    with pytest.raises(ValueError):
        run_mode(ctx, WORKLOAD, "warp_speed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_modes_match_and_report_memory():
    model = make_model().to("cuda")
    cuda_ctx = BenchmarkContext(model=model, tokenizer=IdTokenizer(), device="cuda",
                                batch_size=2, block_size=4)

    results = {mode: run_mode(cuda_ctx, WORKLOAD, mode) for mode in MODES}
    reference = results["naive"].outputs()

    for mode, result in results.items():
        assert result.outputs() == reference, mode
        assert result.metrics()["peak_mb"] > 0
