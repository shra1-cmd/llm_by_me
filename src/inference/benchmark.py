"""
Phase 12: common inference benchmark framework.

Every implementation built so far is measured with the *same* code,
so differences in the numbers come from the implementation, not from
how it was timed:

    workload (same prompts, same max_new_tokens, greedy)
        │
        ▼
    run_mode(mode)          one of MODES below
        │   all requests "arrive" at t0 (offline serving)
        │   every generated token is timestamped through the same
        │   hook (InferenceRequest.on_token / ModelRunner.generate's
        │   on_token), right after the token reaches the host
        ▼
    RunResult               per-request token timestamps + peak memory
        │
        ▼
    benchmark_mode()        warm-up runs (discarded) + N measured runs
        │
        ▼
    ModeReport              mean / median / min / max over iterations

Modes (the architecture each phase introduced):

    naive            Phase 3   ModelRunner.generate, full recompute every
                               token, one request at a time
    kv_cache         Phase 6   prefill + decode with a private contiguous
                               KVCache, one request at a time
    kv_cache_paged   Phase 10  same, KV in KVCacheManager blocks
    batch            Phase 9   static batches, contiguous KVCache
    batch_paged      Phase 10  static batches, paged KV
    continuous       Phase 11  continuous batching, paged KV

Metrics (per measured run):

    TTFT     time from t0 (request arrival) to its first generated token.
             In sequential modes this includes waiting for earlier
             requests — that is what a user would experience.
    ITL      gaps between consecutive generated tokens of one request
    latency  t0 -> last token of the request
    total    t0 -> last token of the whole workload
    tok/s    all generated tokens / total
    peak MB  torch.cuda.max_memory_allocated() during the run
             (model weights + KV pool/caches + activations)

Fairness:
    - one model / tokenizer / device / dtype shared by all modes
    - greedy sampling, so every mode must emit identical tokens
      (checked, and a mismatch fails the benchmark)
    - torch.cuda.synchronize() before t0 and after the run; token
      times are taken after .item(), which already synchronizes
    - each mode builds a fresh engine; paged modes get a KV pool sized
      for their concurrency (not a huge default pool)
"""

import gc
import statistics
import time
from dataclasses import dataclass, field

import torch

from src.inference.continuous_batching import ContinuousBatchingEngine
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache_manager import KVCacheManager
from src.inference.model_runner import ModelRunner
from src.inference.request import SamplingParams

MODES = {
    "naive": "Phase 3 naive",
    "kv_cache": "Phase 6 KV cache",
    "kv_cache_paged": "Phase 10 paged KV",
    "batch": "Phase 9 static batch",
    "batch_paged": "Phase 10 static batch, paged",
    "continuous": "Phase 11 continuous",
}

SEQUENTIAL_MODES = {"naive", "kv_cache", "kv_cache_paged"}


# ======================================================================
# Workload + per-run records
# ======================================================================


@dataclass(frozen=True)
class BenchmarkRequest:
    request_id: str
    prompt: str
    max_new_tokens: int


@dataclass
class RequestTrace:
    request_id: str
    prompt_tokens: int
    token_ids: list[int] = field(default_factory=list)       # prompt + generated
    token_times: list[float] = field(default_factory=list)   # one per generated token

    @property
    def generated_tokens(self) -> int:
        return len(self.token_times)

    def ttft(self, t0: float) -> float | None:
        return self.token_times[0] - t0 if self.token_times else None

    def latency(self, t0: float) -> float | None:
        return self.token_times[-1] - t0 if self.token_times else None

    def itls(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]


@dataclass
class RunResult:
    mode: str
    t0: float
    t_end: float
    traces: dict[str, RequestTrace]
    peak_memory_bytes: int | None
    kv_pool_bytes: int

    @property
    def total_seconds(self) -> float:
        return self.t_end - self.t0

    @property
    def generated_tokens(self) -> int:
        return sum(t.generated_tokens for t in self.traces.values())

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.traces.values())

    def outputs(self) -> dict[str, list[int]]:
        return {rid: t.token_ids for rid, t in self.traces.items()}

    def metrics(self) -> dict:
        ttfts = [t.ttft(self.t0) for t in self.traces.values() if t.token_times]
        latencies = [t.latency(self.t0) for t in self.traces.values() if t.token_times]
        itls = [gap for t in self.traces.values() for gap in t.itls()]

        return {
            "total_s": self.total_seconds,
            "tokens_per_s": self.generated_tokens / self.total_seconds,
            "ttft_mean_s": statistics.mean(ttfts),
            "latency_mean_s": statistics.mean(latencies),
            "latency_max_s": max(latencies),
            "itl_mean_ms": statistics.mean(itls) * 1000 if itls else 0.0,
            "itl_p50_ms": statistics.median(itls) * 1000 if itls else 0.0,
            "peak_mb": (
                self.peak_memory_bytes / 1024 ** 2
                if self.peak_memory_bytes is not None
                else None
            ),
            "kv_pool_mb": self.kv_pool_bytes / 1024 ** 2,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_tokens": self.prompt_tokens + self.generated_tokens,
        }


# ======================================================================
# Measurement helpers
# ======================================================================


def is_cuda(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def sync(device: str):
    if is_cuda(device):
        torch.cuda.synchronize()


def free_memory(device: str):
    gc.collect()
    if is_cuda(device):
        torch.cuda.empty_cache()


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


class _Clock:
    """Starts the run: syncs, resets peak memory, records t0."""

    def __init__(self, device: str):
        self.device = device

    def start(self) -> float:
        sync(self.device)
        if is_cuda(self.device):
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    def stop(self) -> tuple[float, int | None]:
        sync(self.device)
        t_end = time.perf_counter()
        peak = torch.cuda.max_memory_allocated() if is_cuda(self.device) else None
        return t_end, peak


# ======================================================================
# Mode runners
# ======================================================================


@dataclass
class BenchmarkContext:
    model: object
    tokenizer: object
    device: str
    batch_size: int = 4
    block_size: int = 16

    def __post_init__(self):
        self.sampling_params = SamplingParams(greedy=True)
        self.sampler = self.sampling_params.to_sampler()
        self.runner = ModelRunner(
            model=self.model,
            tokenizer=self.tokenizer,
            sampler=self.sampler,
            device=self.device,
        )

    def pool_for(self, workload: list[BenchmarkRequest], concurrency: int) -> KVCacheManager:
        """KV pool big enough for `concurrency` of the longest requests."""

        longest = max(
            len(self.tokenizer.encode(r.prompt)) + r.max_new_tokens for r in workload
        )
        blocks_per_request = -(-longest // self.block_size)

        return KVCacheManager.for_model(
            self.model,
            num_blocks=blocks_per_request * concurrency,
            block_size=self.block_size,
            device=self.device,
        )

    def tiny_pool(self) -> KVCacheManager:
        """Placeholder pool for non-paged engines (they never touch it)."""

        return KVCacheManager.for_model(self.model, num_blocks=1, device=self.device)


def _attach_traces(engine, workload):
    traces = {}
    requests = []

    for spec in workload:
        request = engine.create_request(
            spec.prompt,
            max_new_tokens=spec.max_new_tokens,
            request_id=spec.request_id,
        )
        trace = RequestTrace(request_id=spec.request_id, prompt_tokens=request.prompt_len)

        request.on_token = lambda _req, _tok, trace=trace: trace.token_times.append(
            time.perf_counter()
        )

        traces[spec.request_id] = trace
        requests.append(request)

    return requests, traces


def _run_naive(ctx: BenchmarkContext, workload, clock: _Clock):
    traces = {
        spec.request_id: RequestTrace(
            request_id=spec.request_id,
            prompt_tokens=len(ctx.tokenizer.encode(spec.prompt)),
        )
        for spec in workload
    }

    t0 = clock.start()

    for spec in workload:
        trace = traces[spec.request_id]
        result = ctx.runner.generate(
            spec.prompt,
            max_new_tokens=spec.max_new_tokens,
            on_token=lambda _tok, trace=trace: trace.token_times.append(time.perf_counter()),
        )
        trace.token_ids = result["token_ids"]

    t_end, peak = clock.stop()

    return RunResult("naive", t0, t_end, traces, peak, kv_pool_bytes=0)


def _run_engine(ctx: BenchmarkContext, workload, clock: _Clock, mode: str):
    paged = mode in ("kv_cache_paged", "batch_paged", "continuous")
    concurrency = 1 if mode in SEQUENTIAL_MODES else ctx.batch_size
    manager = ctx.pool_for(workload, concurrency) if paged else ctx.tiny_pool()

    common = dict(
        runner=ctx.runner,
        tokenizer=ctx.tokenizer,
        sampler=ctx.sampler,
        device=ctx.device,
        kv_cache_manager=manager,
    )

    if mode == "continuous":
        engine = ContinuousBatchingEngine(**common, max_batch_size=ctx.batch_size)
    else:
        engine = InferenceEngine(**common, paged_kv=paged)

    requests, traces = _attach_traces(engine, workload)

    t0 = clock.start()

    for request in requests:
        engine.submit(request)

    if mode == "continuous":
        while engine.has_work():
            engine.step()
    elif mode in SEQUENTIAL_MODES:
        while engine.scheduler.has_waiting():
            engine.step()
    else:
        while engine.scheduler.has_waiting():
            engine.step_batch(ctx.batch_size)

    t_end, peak = clock.stop()

    for request in requests:
        traces[request.request_id].token_ids = request.all_tokens

    pool_bytes = manager.pool.num_bytes if paged else 0

    return RunResult(mode, t0, t_end, traces, peak, kv_pool_bytes=pool_bytes)


def run_mode(ctx: BenchmarkContext, workload: list[BenchmarkRequest], mode: str) -> RunResult:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {list(MODES)}")

    clock = _Clock(ctx.device)

    try:
        if mode == "naive":
            return _run_naive(ctx, workload, clock)

        return _run_engine(ctx, workload, clock, mode)
    finally:
        free_memory(ctx.device)


# ======================================================================
# Warm-up + iterations + aggregation
# ======================================================================


REPORTED_METRICS = [
    "total_s",
    "tokens_per_s",
    "ttft_mean_s",
    "latency_mean_s",
    "itl_mean_ms",
    "itl_p50_ms",
    "peak_mb",
]


@dataclass
class ModeReport:
    mode: str
    runs: list[RunResult]

    @property
    def outputs(self) -> dict[str, list[int]]:
        return self.runs[0].outputs()

    @property
    def deterministic(self) -> bool:
        """Every measured iteration produced the same tokens."""

        first = self.outputs
        return all(run.outputs() == first for run in self.runs[1:])

    def stats(self) -> dict[str, dict]:
        per_run = [run.metrics() for run in self.runs]
        out = {}

        for name in REPORTED_METRICS:
            values = [m[name] for m in per_run if m[name] is not None]
            if values:
                out[name] = summarize(values)

        # Token counts are identical across runs; report them as-is.
        for name in ("prompt_tokens", "generated_tokens", "total_tokens", "kv_pool_mb"):
            out[name] = per_run[0][name]

        return out


def benchmark_mode(
    ctx: BenchmarkContext,
    workload: list[BenchmarkRequest],
    mode: str,
    warmup: int = 1,
    iterations: int = 3,
) -> ModeReport:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    for _ in range(warmup):
        run_mode(ctx, workload, mode)

    return ModeReport(mode, [run_mode(ctx, workload, mode) for _ in range(iterations)])


def outputs_match(reports: list[ModeReport]) -> bool:
    """All modes deterministic and emitting identical tokens per request."""

    if not reports:
        return True

    reference = reports[0].outputs

    return all(r.deterministic and r.outputs == reference for r in reports)
