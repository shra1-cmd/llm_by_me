"""
Phase 13: PyTorch execution profiling framework.

Phase 12 measured *how fast* each engine is. This module answers
*where the time goes*, by running a workload under torch.profiler with
our named regions (src/model/profiling.py) switched on:

    workload fn  (e.g. one prefill, or 20 decode steps)
        │  optional setup() before every run, never timed/profiled
        │  (e.g. build a prefilled KV cache for a decode-only profile)
        │
        │  1. warm-up run            (cuBLAS / allocator init excluded)
        │  2. timed runs, regions OFF, no profiler -> true wall time
        │  3. one profiled run, regions ON         -> trace + tables
        ▼
    ProfileResult
        regions    our labels: embedding, layer_i, rmsnorm, attention/*,
                   mlp/*, lm_head, sampling, kv_cache/*, batch/build
                   cpu_ms = time the CPU spent inside the region
                            (Python + PyTorch dispatch + kernel launch)
                   gpu_ms = GPU execution time of the kernels launched
                            from inside the region
        kernels    every CUDA kernel, bucketed into categories:
                   gemm / gemv, attention, elementwise, reduction,
                   copy / cat, index / gather, other
        ops        aten:: operators with self CPU time (dispatch cost)
                   and the GPU time of the kernels they launched
        gpu_busy   sum of all kernel time; busy / wall = how much of
                   the wall-clock time the GPU was actually working
        files      trace.json (open in https://ui.perfetto.dev or
                   chrome://tracing), ops.txt, shapes.txt

How to read CPU vs GPU time:

    Python / PyTorch dispatch (CPU)  ──launch──►  kernel runs (GPU)

The CPU enqueues kernels asynchronously. If CPU time per step is much
larger than GPU busy time, the GPU is waiting for the CPU to launch
work (launch/dispatch-bound — typical for small models and decode).
If GPU busy approaches wall time, the kernels themselves are the
bottleneck (compute/memory-bound — typical for large prefill).

Note on region GPU time: torch.profiler also records a GPU-side span
for each record_function, but that span runs from the region's first
to its last kernel *including idle gaps*, so it is not busy time. We
use the CPU-side annotation's device_time_total instead, which sums
only the kernels launched inside the region.

Kineto pitfall: if a process ever starts a CPU-only profiler session,
later sessions in that process silently record *no* CUDA kernels.
profiler_activities() therefore always includes CUDA when a GPU is
present, even for CPU workloads — use it for any profiler you start.

Measurement caveat: enabling regions + the profiler adds CPU overhead
(~35% per decode step on the V1 model), so absolute CPU numbers from
the profiled run are inflated; wall_ms comes from the unprofiled runs.
Kernel durations are essentially unaffected.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from src.model import profiling

# --------------------------------------------------
# Kernel categories
# --------------------------------------------------

# Order matters: first match wins.
#   - attention before gemm: the memory-efficient SDPA kernel is named
#     "fmha_cutlassF_..." and must not count as a plain GEMM
#   - gemm before gemv: cuBLAS's small-N GEMM "gemmSN_TN_kernel<...,
#     cublasGemvTensor...>" mentions Gemv only in a template argument
KERNEL_CATEGORIES = [
    ("attention", re.compile(r"flash|fmha|attention|efficient|softmax", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|xmma|splitK|sgemm", re.I)),
    ("gemv", re.compile(r"gemv", re.I)),
    ("copy/cat", re.compile(r"CatArray|copy|memcpy|memset", re.I)),
    ("index/gather", re.compile(r"index|gather|scatter", re.I)),
    ("reduction", re.compile(r"reduce", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled", re.I)),
]


def categorize_kernel(name: str) -> str:
    for category, pattern in KERNEL_CATEGORIES:
        if pattern.search(name):
            return category
    return "other"


# --------------------------------------------------
# Result records
# --------------------------------------------------


@dataclass
class RegionStat:
    name: str
    calls: int
    cpu_ms: float
    gpu_ms: float


@dataclass
class KernelStat:
    name: str
    category: str
    calls: int
    gpu_ms: float


@dataclass
class OpStat:
    name: str
    calls: int
    self_cpu_ms: float
    gpu_ms: float
    shapes: str = ""


@dataclass
class ProfileResult:
    name: str
    steps: int
    wall_ms: float                  # per run, unprofiled, regions off
    profiled_wall_ms: float         # per run, under the profiler
    regions: dict[str, RegionStat]
    kernels: list[KernelStat]
    ops: list[OpStat]
    ops_by_shape: list[OpStat] = field(default_factory=list)
    out_dir: Path | None = None
    files: dict[str, str] = field(default_factory=dict)

    # ---- totals for the whole profiled run
    @property
    def gpu_busy_ms(self) -> float:
        return sum(k.gpu_ms for k in self.kernels)

    @property
    def kernel_launches(self) -> int:
        return sum(k.calls for k in self.kernels)

    # ---- per step
    @property
    def wall_ms_per_step(self) -> float:
        return self.wall_ms / self.steps

    @property
    def gpu_busy_ms_per_step(self) -> float:
        return self.gpu_busy_ms / self.steps

    @property
    def launches_per_step(self) -> float:
        return self.kernel_launches / self.steps

    @property
    def gpu_utilization(self) -> float:
        """GPU busy time / unprofiled wall time (0..1)."""

        return self.gpu_busy_ms / self.wall_ms if self.wall_ms > 0 else 0.0

    def categories(self) -> dict[str, float]:
        """GPU ms per kernel category, largest first."""

        totals: dict[str, float] = {}
        for k in self.kernels:
            totals[k.category] = totals.get(k.category, 0.0) + k.gpu_ms
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    def region(self, name: str) -> RegionStat:
        return self.regions.get(name, RegionStat(name, 0, 0.0, 0.0))

    def top_kernels(self, n: int = 10) -> list[KernelStat]:
        return sorted(self.kernels, key=lambda k: -k.gpu_ms)[:n]

    def top_ops(self, n: int = 10, by: str = "gpu_ms") -> list[OpStat]:
        return sorted(self.ops, key=lambda o: -getattr(o, by))[:n]


# --------------------------------------------------
# Profiling
# --------------------------------------------------


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def _sync(device: str):
    if _is_cuda(device):
        torch.cuda.synchronize()


def profiler_activities() -> list:
    """CPU + CUDA whenever a GPU exists (see the Kineto pitfall above)."""

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


def _us_to_ms(value) -> float:
    return float(value) / 1000.0


def _collect(prof):
    regions: dict[str, RegionStat] = {}
    kernels: list[KernelStat] = []
    ops: list[OpStat] = []

    for evt in prof.key_averages():
        is_annotation = getattr(evt, "is_user_annotation", False)
        device_type = getattr(evt, "device_type", DeviceType.CPU)

        if is_annotation:
            # CPU-side annotation only (see module docstring).
            if device_type == DeviceType.CPU:
                regions[evt.key] = RegionStat(
                    name=evt.key,
                    calls=evt.count,
                    cpu_ms=_us_to_ms(evt.cpu_time_total),
                    gpu_ms=_us_to_ms(evt.device_time_total),
                )
            continue

        if device_type == DeviceType.CUDA:
            kernels.append(
                KernelStat(
                    name=evt.key,
                    category=categorize_kernel(evt.key),
                    calls=evt.count,
                    gpu_ms=_us_to_ms(evt.self_device_time_total),
                )
            )
        elif evt.key.startswith("aten::"):
            ops.append(
                OpStat(
                    name=evt.key,
                    calls=evt.count,
                    self_cpu_ms=_us_to_ms(evt.self_cpu_time_total),
                    gpu_ms=_us_to_ms(evt.device_time_total),
                )
            )

    return regions, kernels, ops


def _shape_table(prof, names=("aten::mm", "aten::addmm", "aten::bmm", "aten::linear",
                              "aten::scaled_dot_product_attention", "aten::cat",
                              "aten::index", "aten::index_put_")) -> list[OpStat]:
    rows = []

    for evt in prof.key_averages(group_by_input_shape=True):
        if evt.key in names and getattr(evt, "device_type", DeviceType.CPU) == DeviceType.CPU:
            rows.append(
                OpStat(
                    name=evt.key,
                    calls=evt.count,
                    self_cpu_ms=_us_to_ms(evt.self_cpu_time_total),
                    gpu_ms=_us_to_ms(evt.device_time_total),
                    shapes=str(evt.input_shapes),
                )
            )

    return sorted(rows, key=lambda o: -o.gpu_ms)


def profile_workload(
    name: str,
    fn,
    device: str,
    steps: int = 1,
    out_dir: str | Path | None = None,
    setup=None,
    warmup: int = 1,
    timing_repeats: int = 5,
    record_shapes: bool = True,
    profile_memory: bool = True,
    regions: bool = True,
) -> ProfileResult:
    """
    Profile `fn` (which should perform `steps` model steps).

    setup: optional callable run before *every* call of fn and never
    timed or profiled; its return value is passed to fn. Use it to
    build fresh state, e.g. a prefilled KV cache for a decode-only
    profile. Without setup, fn is called with no arguments.

    regions=False profiles with our named regions left off (Phase 15:
    a torch.compile'd model would otherwise recompile for the
    record_function calls; kernels are recorded either way).

    Returns a ProfileResult and, if out_dir is given, writes
    trace.json / ops.txt / shapes.txt there.
    """

    def call():
        if setup is None:
            return lambda: fn()
        state = setup()
        return lambda: fn(state)

    for _ in range(warmup):
        call()()
    _sync(device)

    # True wall time: no profiler, regions off, setup excluded.
    with profiling.enabled(False):
        total = 0.0
        for _ in range(timing_repeats):
            run = call()
            _sync(device)
            start = time.perf_counter()
            run()
            _sync(device)
            total += time.perf_counter() - start
        wall_ms = total * 1000 / timing_repeats

    activities = profiler_activities()

    run = call()

    with profiling.enabled(regions):
        with profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
        ) as prof:
            _sync(device)
            start = time.perf_counter()
            run()
            _sync(device)
            profiled_wall_ms = (time.perf_counter() - start) * 1000

    regions, kernels, ops = _collect(prof)

    result = ProfileResult(
        name=name,
        steps=steps,
        wall_ms=wall_ms,
        profiled_wall_ms=profiled_wall_ms,
        regions=regions,
        kernels=kernels,
        ops=ops,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.out_dir = out

        trace = out / "trace.json"
        prof.export_chrome_trace(str(trace))
        result.files["trace"] = str(trace)

        sort_key = "self_cuda_time_total" if _is_cuda(device) else "self_cpu_time_total"

        ops_path = out / "ops.txt"
        ops_path.write_text(prof.key_averages().table(sort_by=sort_key, row_limit=60))
        result.files["ops"] = str(ops_path)

        if record_shapes:
            shapes_path = out / "shapes.txt"
            shapes_path.write_text(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by=sort_key, row_limit=60
                )
            )
            result.files["shapes"] = str(shapes_path)

    if record_shapes:
        result.ops_by_shape = _shape_table(prof)

    return result


# --------------------------------------------------
# Memory
# --------------------------------------------------


def capture_memory_snapshot(fn, path: str | Path, device: str) -> str | None:
    """
    Record every CUDA allocation made by fn() and dump a snapshot
    (open it at https://pytorch.org/memory_viz). Returns the path, or
    None when not on CUDA / unsupported.
    """

    if not _is_cuda(device):
        return None

    try:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
    except Exception:
        return None

    try:
        fn()
        _sync(device)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(path))
        return str(path)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def measure_peak_memory(fn, device: str) -> dict:
    """
    Allocated memory before fn(), the peak during it, and what it left
    allocated afterwards (MB). peak - before = transient working set.
    """

    if not _is_cuda(device):
        return {"before_mb": None, "peak_mb": None, "after_mb": None, "transient_mb": None}

    _sync(device)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    keep = fn()
    _sync(device)

    peak = torch.cuda.max_memory_allocated()
    after = torch.cuda.memory_allocated()
    del keep

    mb = 1024 ** 2

    return {
        "before_mb": before / mb,
        "peak_mb": peak / mb,
        "after_mb": after / mb,
        "transient_mb": (peak - before) / mb,
    }
