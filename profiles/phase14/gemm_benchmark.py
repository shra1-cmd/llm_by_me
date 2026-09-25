"""
Phase 14 — CUDA / GEMM / cuBLAS investigation.

Phase 13 showed *where* GPU time goes (prefill: ampere_sgemm_* +
cublasLt::splitKreduce_kernel; decode: gemv kernels, ~37% GPU busy).
This script explains *why*, by tracing and measuring the matrix
multiplications of the V1 model in isolation. Nothing is optimized:
every experiment runs the model's own dtype (fp32) and default
PyTorch settings, except the TF32 experiment, which flips precision
temporarily to show what hardware *could* run the GEMM and restores it.

Experiments (each one feeds a section of findings.md):

    trace      nn.Linear -> F.linear -> aten::linear -> aten::t /
               aten::matmul -> aten::mm -> cuBLAS kernel, recorded twice:
               TorchDispatchMode (every aten op, shapes + strides) and
               torch.profiler (op tree with the kernels each op launched)
    tensors    shape / stride / dtype / contiguity of the real tensors
               entering every Linear of layer 0 (+ LM head) during a
               128-token prefill and a 1-token decode (forward hooks)
    layers     the model's matmul table: M, K, N, FLOPs = 2*M*K*N,
               minimum bytes, arithmetic intensity, for prefill (M=128)
               and decode (M=1)
    roofline   measured peak FP32 / TF32 GEMM throughput, DRAM bandwidth
               and the CPU cost of launching one tiny kernel
    sweep      every model Linear shape at M = 1, 2, 4, ..., 512:
               wall time per call, GPU kernel time, kernels chosen,
               TFLOPS, GB/s, number of output tiles (CTAs) vs SMs
    tf32       same shapes with TF32 allowed: which kernels cuBLAS
               switches to (Tensor Cores) and how much faster
    batching   B separate [1,K]x[K,N] calls vs one [B,K]x[K,N] call
               (what continuous batching buys decode)
    forward    sum of isolated kernel times over one full forward,
               cross-checked against the Phase 13 profile
    kv_cat     cost of KVCache.update's torch.cat vs sequence length,
               allocations per decode step, total bytes copied

Output (next to this file):

    results.json   every number above
    findings.md    answers to the ten Phase 14 questions, generated
                   from results.json (rerun to refresh)

Usage:

    python profiles/phase14/gemm_benchmark.py
    python profiles/phase14/gemm_benchmark.py --quick     # fewer iterations
    python profiles/phase14/gemm_benchmark.py --checkpoint checkpoints/v1/step_9950.pt
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.profiler import profile
from torch.utils._python_dispatch import TorchDispatchMode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from configs.v1 import ModelConfig  # noqa: E402
from src.inference.kv_cache import KVCache  # noqa: E402
from src.inference.profiler import categorize_kernel, profiler_activities  # noqa: E402

M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
KV_LENGTHS = [16, 32, 64, 128, 256, 384, 511]
BYTES_FP32 = 4

# ======================================================================
# Model shapes
# ======================================================================


def linear_shapes(config: ModelConfig) -> list[dict]:
    """
    Every matmul of one forward pass, as [M,K] x [K,N].

    count = how many times this shape runs per forward (per layer x
    num_layers). k_proj / v_proj share a shape, as do gate / up.
    """

    head_dim = config.hidden_dim // config.num_q_heads
    kv_dim = config.num_kv_heads * head_dim
    L = config.num_layers

    return [
        {"name": "q_proj", "K": config.hidden_dim, "N": config.hidden_dim, "count": L,
         "module": "attention.q_proj"},
        {"name": "k_proj/v_proj", "K": config.hidden_dim, "N": kv_dim, "count": 2 * L,
         "module": "attention.k_proj, attention.v_proj"},
        {"name": "out_proj", "K": config.hidden_dim, "N": config.hidden_dim, "count": L,
         "module": "attention.out_proj"},
        {"name": "gate_proj/up_proj", "K": config.hidden_dim, "N": config.ffn_dim,
         "count": 2 * L, "module": "ffn.gate_proj, ffn.up_proj"},
        {"name": "down_proj", "K": config.ffn_dim, "N": config.hidden_dim, "count": L,
         "module": "ffn.down_proj"},
        {"name": "lm_head", "K": config.hidden_dim, "N": config.vocab_size, "count": 1,
         "module": "x @ token_embedding.weight.T (tied)"},
    ]


def gemm_cost(M: int, K: int, N: int, elem_bytes: int = BYTES_FP32) -> dict:
    """
    FLOPs and minimum DRAM traffic of [M,K] x [K,N].

    flops = 2*M*K*N (one multiply + one add per term)
    bytes = read A + read B + write C, each exactly once (a lower
            bound: real kernels may re-read tiles or split-K partials)
    arithmetic intensity = flops / bytes
    """

    flops = 2 * M * K * N
    bytes_moved = elem_bytes * (M * K + K * N + M * N)
    return {
        "M": M, "K": K, "N": N,
        "flops": flops,
        "bytes": bytes_moved,
        "weight_bytes": elem_bytes * K * N,
        "arithmetic_intensity": flops / bytes_moved,
    }


def layer_table(config: ModelConfig, prefill_tokens: int = 128) -> list[dict]:
    rows = []
    for s in linear_shapes(config):
        rows.append({
            **s,
            "prefill": gemm_cost(prefill_tokens, s["K"], s["N"]),
            "decode": gemm_cost(1, s["K"], s["N"]),
        })
    return rows


TILE_RE = re.compile(r"_(\d+)x(\d+)_")


def output_tiles(kernel: str, M: int, N: int) -> int | None:
    """
    Thread blocks (CTAs) a tiled cuBLAS GEMM launches for C = [M, N].

    cuBLAS is column-major, so PyTorch computes the row-major C[M,N]
    as the column-major C^T[N,M]. For "ampere_sgemm_128x64_tn" a CTA
    owns a 128 (along N) x 64 (along M) output tile. Split-K multiplies
    this by the number of K slices (not visible in the name).
    """

    match = TILE_RE.search(kernel)
    if not match or "gemm" not in kernel:
        return None
    tile_n, tile_m = int(match.group(1)), int(match.group(2))
    return math.ceil(N / tile_n) * math.ceil(M / tile_m)


# ======================================================================
# Timing helpers
# ======================================================================


def time_fn(fn, iters: int, warmup: int = 10) -> dict:
    """
    wall_us   CPU wall time per call for `iters` back-to-back calls,
              synchronized once at the end = max(CPU issue rate, GPU)
    stream_us CUDA-event time per call on the stream (includes any idle
              gaps while the GPU waits for the CPU to launch)
    """

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return {
        "wall_us": (t1 - t0) * 1e6 / iters,
        "stream_us": start.elapsed_time(end) * 1e3 / iters,
    }


def profile_kernels(fn, iters: int = 10) -> list[dict]:
    """Kernels launched by one call of fn: name, calls/call, GPU us/call."""

    fn()
    torch.cuda.synchronize()
    with profile(activities=profiler_activities()) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    kernels = []
    for evt in prof.key_averages():
        if getattr(evt, "device_type", DeviceType.CPU) != DeviceType.CUDA:
            continue
        kernels.append({
            "name": evt.key,
            "category": categorize_kernel(evt.key),
            "calls": evt.count / iters,
            "us": evt.self_device_time_total / iters,
        })
    return sorted(kernels, key=lambda k: -k["us"])


def short_kernel(name: str, width: int = 60) -> str:
    """
    Drop template arguments: 'void gemv2T_kernel_val<int, ...>' -> 'gemv2T_kernel_val'.
    CUTLASS wraps the real kernel in a template ('cutlass::Kernel2<cutlass_80_
    tensorop_s1688gemm_...>'), so for those the template argument is the name.
    """

    name = re.sub(r"^(void |std::enable_if<[^>]*>::type )", "", name)
    base = name.split("<", 1)[0].strip()
    if base.startswith("cutlass::Kernel") and "<" in name:
        base = re.split(r"[<>(]", name)[1]
    return (base or name)[:width]


class fp32_precision:
    """Temporarily set torch.set_float32_matmul_precision ('highest' = true FP32)."""

    def __init__(self, value: str):
        self.value = value

    def __enter__(self):
        self.saved = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(self.value)

    def __exit__(self, *exc):
        torch.set_float32_matmul_precision(self.saved)


# ======================================================================
# 14.3 — trace nn.Linear down to the kernel
# ======================================================================


class AtenLogger(TorchDispatchMode):
    """Record every aten op that reaches the backend, with tensor metadata."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.calls.append({
            "op": str(func.overloadpacket.__name__) if hasattr(func, "overloadpacket") else str(func),
            "overload": str(func),
            "inputs": [describe(a) for a in args if isinstance(a, torch.Tensor)],
            "output": describe(out) if isinstance(out, torch.Tensor) else None,
        })
        return out


def describe(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "contiguous": t.is_contiguous(),
        "storage_offset": t.storage_offset(),
    }


def trace_linear(K: int, N: int, M: int, device: str) -> dict:
    layer = nn.Linear(K, N, bias=False).to(device)
    x = torch.randn(1, M, K, device=device)

    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    with torch.no_grad():
        layer(x)
        sync()

        logger = AtenLogger()
        with logger:
            layer(x)

        with profile(activities=profiler_activities(), record_shapes=True) as prof:
            layer(x)
            sync()

    def walk(evt, depth=0):
        node = {
            "name": evt.name,
            "input_shapes": [list(s) for s in (evt.input_shapes or []) if s],
            "kernels": [k.name for k in getattr(evt, "kernels", [])],
            "children": [walk(c, depth + 1) for c in evt.cpu_children
                         if c.name.startswith("aten::")],
        }
        return node

    roots = [walk(e) for e in prof.events()
             if e.cpu_parent is None and e.name.startswith("aten::")]

    return {
        "module": f"nn.Linear({K}, {N}, bias=False)",
        "input": describe(x),
        "weight": describe(layer.weight),
        "weight_t": describe(layer.weight.t()),
        "dispatch_ops": logger.calls,
        "profiler_tree": roots,
    }


# ======================================================================
# 14.9 — tensors inside the real model
# ======================================================================


def load_model(args, config: ModelConfig):
    if args.checkpoint or (ROOT / args.checkpoint_dir).exists():
        try:
            from src.inference.checkpoint_loader import (
                find_latest_checkpoint,
                load_inference_checkpoint,
            )

            path = args.checkpoint or find_latest_checkpoint(ROOT / args.checkpoint_dir)
            model, _ = load_inference_checkpoint(path, device=args.device)
            return model, str(Path(path).relative_to(ROOT) if Path(path).is_absolute() else path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  (no checkpoint: {exc}; using random weights)")

    from src.model.model import V1LanguageModel

    return V1LanguageModel(config).to(args.device).eval(), None


def model_tensors(model, device: str, prompt_len: int = 128) -> list[dict]:
    """Hook every Linear in layer 0; also capture the LM-head operands."""

    rows = []
    phase = {"name": None}
    block = model.blocks[0]
    linears = {
        "q_proj": block.attention.q_proj,
        "k_proj": block.attention.k_proj,
        "v_proj": block.attention.v_proj,
        "out_proj": block.attention.out_proj,
        "gate_proj": block.ffn.gate_proj,
        "up_proj": block.ffn.up_proj,
        "down_proj": block.ffn.down_proj,
    }

    def hook(name):
        def fn(module, inputs):
            x = inputs[0]
            rows.append({
                "phase": phase["name"],
                "layer": f"blocks.0.{name}",
                "input": describe(x),
                "weight": describe(module.weight),
                "mm_operands": f"{list(x.reshape(-1, x.shape[-1]).shape)} x "
                               f"{list(module.weight.t().shape)}",
            })
        return fn

    handles = [m.register_forward_pre_hook(hook(n)) for n, m in linears.items()]

    def final_norm_hook(module, inputs, output):
        emb = model.token_embedding.weight
        rows.append({
            "phase": phase["name"],
            "layer": "lm_head",
            "input": describe(output),
            "weight": describe(emb),
            "mm_operands": f"{list(output.reshape(-1, output.shape[-1]).shape)} x "
                           f"{list(emb.t().shape)}",
        })

    handles.append(model.final_norm.register_forward_hook(final_norm_hook))

    vocab = model.config.vocab_size
    cache = KVCache(model.config.num_layers)
    with torch.no_grad():
        phase["name"] = "prefill"
        model(torch.randint(0, vocab, (1, prompt_len), device=device), kv_cache=cache)
        phase["name"] = "decode"
        model(torch.randint(0, vocab, (1, 1), device=device), kv_cache=cache)

    for h in handles:
        h.remove()
    return rows


# ======================================================================
# 14.6 / 14.10 — roofline of this GPU
# ======================================================================


def roofline(iters: int) -> dict:
    """
    Best of a few large square F.linear GEMMs (same `tn` layout as the
    model). A laptop GPU's clocks move with power and temperature, so
    main() also raises the FP32 peak to the best point of the M sweep.
    """

    out = {}
    for label, precision in (("fp32", "highest"), ("tf32", "high")):
        best = (0.0, None)
        for n in (2048, 4096):
            x = torch.randn(n, n, device="cuda")
            w = torch.randn(n, n, device="cuda")
            with fp32_precision(precision):
                t = time_fn(lambda: F.linear(x, w), max(iters // 10, 10), warmup=5)
                kernels = profile_kernels(lambda: F.linear(x, w), iters=3)
            tflops = 2 * n ** 3 / (t["stream_us"] * 1e-6) / 1e12
            if tflops > best[0]:
                best = (tflops, kernels[0]["name"] if kernels else None)
            del x, w
        out[f"peak_{label}_tflops"], out[f"peak_{label}_kernel"] = best

    x = torch.empty(64 * 1024 * 1024, device="cuda")  # 256 MB
    y = torch.empty_like(x)
    t = time_fn(lambda: y.copy_(x), 20, warmup=3)
    out["dram_bandwidth_gbs"] = 2 * x.numel() * BYTES_FP32 / (t["stream_us"] * 1e-6) / 1e9
    out["ridge_point_fp32"] = out["peak_fp32_tflops"] * 1e12 / (out["dram_bandwidth_gbs"] * 1e9)

    tiny = torch.zeros(1, device="cuda")
    t = time_fn(lambda: tiny.add_(1), iters * 5)
    out["tiny_kernel_wall_us"] = t["wall_us"]

    del x, y
    torch.cuda.empty_cache()
    return out


# ======================================================================
# 14.11 — M sweep over the model's Linear shapes
# ======================================================================


def measure_linear(M: int, K: int, N: int, iters: int) -> dict:
    x = torch.randn(M, K, device="cuda")
    w = torch.randn(N, K, device="cuda")  # nn.Linear layout [out, in]
    fn = lambda: F.linear(x, w)  # noqa: E731

    timing = time_fn(fn, iters)
    kernels = profile_kernels(fn)
    kernel_us = sum(k["us"] for k in kernels)
    cost = gemm_cost(M, K, N)
    main = kernels[0]["name"] if kernels else ""

    return {
        **cost,
        **timing,
        "kernel_us": kernel_us,
        "kernel_count": sum(k["calls"] for k in kernels),
        "kernels": [{**k, "short": short_kernel(k["name"])} for k in kernels],
        "main_kernel": short_kernel(main),
        "category": categorize_kernel(main) if main else None,
        "split_k": any("splitK" in k["name"] for k in kernels),
        "split_k_us": sum(k["us"] for k in kernels if "splitK" in k["name"]),
        "memset": any("Memset" in k["name"] for k in kernels),
        "ctas": output_tiles(main, M, N),
        "tflops": cost["flops"] / (kernel_us * 1e-6) / 1e12 if kernel_us else 0.0,
        "gbs": cost["bytes"] / (kernel_us * 1e-6) / 1e9 if kernel_us else 0.0,
        "gpu_busy_fraction": kernel_us / timing["wall_us"] if timing["wall_us"] else 0.0,
    }


def m_sweep(config: ModelConfig, iters: int, precision: str = "highest",
            m_values=M_VALUES) -> dict:
    out = {}
    with fp32_precision(precision):
        for s in linear_shapes(config):
            out[s["name"]] = [measure_linear(M, s["K"], s["N"], iters) for M in m_values]
            torch.cuda.empty_cache()
    return out


# ======================================================================
# 14.12 — batching: B x [1,K] vs one [B,K]
# ======================================================================


def batching(config: ModelConfig, iters: int) -> list[dict]:
    K, N = config.hidden_dim, config.ffn_dim  # gate/up projection
    w = torch.randn(N, K, device="cuda")
    rows = []
    for B in BATCH_SIZES:
        singles = [torch.randn(1, K, device="cuda") for _ in range(B)]
        batch = torch.cat(singles)

        def separate():
            for x in singles:
                F.linear(x, w)

        t_sep = time_fn(separate, max(iters // B, 10))
        t_bat = time_fn(lambda: F.linear(batch, w), iters)
        k_sep = profile_kernels(separate)
        k_bat = profile_kernels(lambda: F.linear(batch, w))
        rows.append({
            "B": B, "K": K, "N": N,
            "separate_wall_us": t_sep["wall_us"],
            "separate_kernel_us": sum(k["us"] for k in k_sep),
            "separate_kernel_count": sum(k["calls"] for k in k_sep),
            "batched_wall_us": t_bat["wall_us"],
            "batched_kernel_us": sum(k["us"] for k in k_bat),
            "batched_kernel_count": sum(k["calls"] for k in k_bat),
            "batched_kernel": short_kernel(k_bat[0]["name"]) if k_bat else None,
            "weight_reads_separate_bytes": B * K * N * BYTES_FP32,
            "weight_reads_batched_bytes": K * N * BYTES_FP32,
            "speedup_wall": t_sep["wall_us"] / t_bat["wall_us"],
            "wall_us_per_token_batched": t_bat["wall_us"] / B,
        })
    return rows


# ======================================================================
# Full-forward estimate from the isolated GEMMs
# ======================================================================


def phase13_reference() -> dict:
    """Pull the prefill_128 / decode_20 rows out of profiles/FINDINGS.md, if present."""

    path = ROOT / "profiles" / "FINDINGS.md"
    ref = {}
    if not path.exists():
        return ref
    row = re.compile(r"^\| (prefill_128|decode_20) \| \d+ \| ([\d.]+) \| ([\d.]+) \| "
                     r"([\d.]+)% \| (\d+) \| ([\d.]+)% \|")
    for line in path.read_text().splitlines():
        m = row.match(line)
        if m:
            name, wall, busy, util, launches, mm = m.groups()
            ref[name] = {
                "wall_ms": float(wall), "gpu_busy_ms": float(busy),
                "gpu_util": float(util) / 100, "kernels_per_step": int(launches),
                "matmul_share": float(mm) / 100,
                "matmul_ms": float(busy) * float(mm) / 100,
            }
    return ref


def forward_estimate(config: ModelConfig, sweep: dict, prefill_tokens: int = 128) -> dict:
    out = {}
    for label, M in (("prefill", prefill_tokens), ("decode", 1)):
        flops = kernel_us = weight_bytes = 0.0
        launches = 0.0
        for s in linear_shapes(config):
            point = next(p for p in sweep[s["name"]] if p["M"] == M)
            flops += s["count"] * point["flops"]
            kernel_us += s["count"] * point["kernel_us"]
            launches += s["count"] * point["kernel_count"]
            weight_bytes += s["count"] * point["weight_bytes"]
        out[label] = {
            "M": M,
            "matmul_flops": flops,
            "matmul_kernel_ms": kernel_us / 1e3,
            "matmul_kernel_launches": launches,
            "weight_bytes": weight_bytes,
            "achieved_tflops": flops / (kernel_us * 1e-6) / 1e12,
            "flops_per_token": flops / M,
        }
    return out


# ======================================================================
# 14.13 — the KV-cache torch.cat problem
# ======================================================================


def kv_cat(config: ModelConfig, iters: int, prompt_len: int = 128, new_tokens: int = 20) -> dict:
    head_dim = config.hidden_dim // config.num_q_heads
    H = config.num_kv_heads
    row_bytes = H * head_dim * BYTES_FP32  # one token, one of K or V, one layer

    per_length = []
    for T in KV_LENGTHS:
        old = torch.randn(1, H, T, head_dim, device="cuda")
        new = torch.randn(1, H, 1, head_dim, device="cuda")
        t = time_fn(lambda: torch.cat([old, new], dim=2), iters)
        k = profile_kernels(lambda: torch.cat([old, new], dim=2))
        per_length.append({
            "T": T,
            **t,
            "kernel_us": sum(x["us"] for x in k),
            "kernel": short_kernel(k[0]["name"]) if k else None,
            "bytes_copied": 2 * (T + 1) * row_bytes,  # read old+new, write all
            "useful_bytes": row_bytes,
        })

    # Real KVCache: allocations + kernels for `new_tokens` decode steps.
    L = config.num_layers
    cache = KVCache(L)
    for layer in range(L):
        cache.update(layer, torch.randn(1, H, prompt_len, head_dim, device="cuda"),
                     torch.randn(1, H, prompt_len, head_dim, device="cuda"))
    step_k = torch.randn(1, H, 1, head_dim, device="cuda")
    torch.cuda.synchronize()

    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(new_tokens):
        for layer in range(L):
            cache.update(layer, step_k, step_k)
    end.record()
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"] - before

    def total_copy_bytes(prompt, n):
        # step t (cache holds prompt+t tokens) rewrites prompt+t+1 rows, K and V, all layers
        rows = sum(prompt + t + 1 for t in range(n))
        return 2 * L * rows * row_bytes * 2  # read + write

    max_new = config.max_seq_len - prompt_len
    return {
        "row_bytes": row_bytes,
        "per_length": per_length,
        "decode_sim": {
            "prompt_len": prompt_len,
            "steps": new_tokens,
            "cat_calls": 2 * L * new_tokens,
            "allocations": allocations,
            "allocations_per_step": allocations / new_tokens,
            "stream_ms_per_step": start.elapsed_time(end) / new_tokens,
        },
        "full_generation": {
            "prompt_len": prompt_len,
            "new_tokens": max_new,
            "bytes_copied": total_copy_bytes(prompt_len, max_new),
            "bytes_new_kv": 2 * L * max_new * row_bytes,
            "cache_bytes_at_end": 2 * L * config.max_seq_len * row_bytes,
        },
    }


# ======================================================================
# Report
# ======================================================================


def fmt_flops(f: float) -> str:
    for unit, scale in (("GFLOP", 1e9), ("MFLOP", 1e6), ("KFLOP", 1e3)):
        if f >= scale:
            return f"{f / scale:.2f} {unit}"
    return f"{f:.0f} FLOP"


def fmt_bytes(b: float) -> str:
    for unit, scale in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if b >= scale:
            return f"{b / scale:.2f} {unit}"
    return f"{b:.0f} B"


def linear_shapes_from_table(table: list[dict]) -> list[dict]:
    return [{"name": row["name"], "count": row["count"]} for row in table]


def at(points: list[dict], M: int) -> dict:
    return next(p for p in points if p["M"] == M)


def write_findings(path: Path, r: dict):
    meta, roof, sweep = r["meta"], r["roofline"], r["m_sweep"]
    tf32, fwd, ref = r["tf32_sweep"], r["forward_estimate"], r["phase13"]
    kv, bat, table = r["kv_cat"], r["batching"], r["layer_table"]
    gu = sweep["gate_proj/up_proj"]
    gu128, gu1 = at(gu, 128), at(gu, 1)
    lm1 = at(sweep["lm_head"], 1)
    sms = meta["sm_count"]
    lines = []
    add = lines.append

    add("# Phase 14 — CUDA / GEMM / cuBLAS findings\n")
    add(f"Generated by `profiles/phase14/gemm_benchmark.py` on **{meta['device']}** "
        f"(compute capability {meta['compute_capability']}, {sms} SMs, "
        f"PyTorch {meta['torch']}, CUDA {meta['cuda']}). "
        f"fp32 matmul precision: `{meta['fp32_matmul_precision']}`. "
        "All numbers come from `results.json`; rerun the script to refresh both files.\n")
    add("Measured roofline of this GPU:\n")
    add("| peak FP32 GEMM (CUDA cores) | peak TF32 GEMM (Tensor Cores) | DRAM bandwidth | "
        "FP32 ridge point | CPU cost of one tiny kernel |")
    add("|---|---|---|---|---|")
    add(f"| {roof['peak_fp32_tflops']:.2f} TFLOPS | {roof['peak_tf32_tflops']:.2f} TFLOPS | "
        f"{roof['dram_bandwidth_gbs']:.0f} GB/s | {roof['ridge_point_fp32']:.1f} FLOP/byte | "
        f"{roof['tiny_kernel_wall_us']:.1f} us |\n")
    add(f"FP32 peak source: {roof['peak_fp32_source']} (`{short_kernel(roof['peak_fp32_kernel'] or '')}`). "
        "The \"tiny kernel\" cost is the CPU wall time per `x.add_(1)` on a 1-element "
        "tensor: Python + PyTorch dispatcher + `cudaLaunchKernel`, with ~no GPU work.\n")
    add("A kernel whose arithmetic intensity (FLOP per byte of DRAM traffic) is below the "
        "ridge point cannot be compute-bound: DRAM delivers bytes slower than the SMs "
        "consume them.\n")

    # ------------------------------------------------------------------
    add("## The model's matrix multiplications\n")
    add("Every matmul of one forward pass (8 layers). `count` = calls per forward. "
        "Weights are `nn.Linear` layout `[N, K]`; the multiplication is `x[M,K] @ W.T[K,N]`.\n")
    add("| op | count | prefill `[128,K]x[K,N]` | FLOPs | AI | decode `[1,K]x[K,N]` | FLOPs | AI |")
    add("|---|---|---|---|---|---|---|---|")
    for row in table:
        p, d = row["prefill"], row["decode"]
        add(f"| {row['name']} | {row['count']} | `[128,{p['K']}]x[{p['K']},{p['N']}]` | "
            f"{fmt_flops(p['flops'])} | {p['arithmetic_intensity']:.1f} | "
            f"`[1,{d['K']}]x[{d['K']},{d['N']}]` | {fmt_flops(d['flops'])} | "
            f"{d['arithmetic_intensity']:.2f} |")
    add("")
    add(f"One forward: prefill does **{fmt_flops(fwd['prefill']['matmul_flops'])}** of matmul "
        f"work, decode **{fmt_flops(fwd['decode']['matmul_flops'])}** — "
        f"{fwd['prefill']['matmul_flops'] / fwd['decode']['matmul_flops']:.0f}x less — yet "
        f"both read the same {fmt_bytes(fwd['decode']['weight_bytes'])} of weights. "
        f"The LM head is the largest single matmul "
        f"({100 * table[-1]['prefill']['flops'] / fwd['prefill']['matmul_flops']:.0f}% of "
        "prefill matmul FLOPs), and its weight is the embedding table itself.\n")
    add("Note: the model has **separate** q/k/v projections (512→512, 512→128, 512→128 "
        "because of GQA with 2 KV heads), not one fused `[512, 1536]` QKV matmul, and "
        "separate gate/up projections — so each forward launches 7 Linear matmuls per "
        "layer + the LM head = 57 matmuls (Phase 13 counted exactly 57 `aten::mm`).\n")

    # ------------------------------------------------------------------
    add("## 1. What happens to `nn.Linear`?\n")
    tr = r["trace"]
    add(f"Traced `{tr['module']}` on an input of shape {tr['input']['shape']} "
        "(the prefill shape):\n")
    add("```text")
    add("nn.Linear.forward(x)")
    add("  -> F.linear(x, weight)                  # y = x @ weight.T (+ bias)")
    add("  -> aten::linear                          # CompositeImplicit op: decomposes")

    def tree(node, depth):
        kernels = f"   => GPU: {', '.join(short_kernel(k) for k in node['kernels'])}" \
            if node["kernels"] else ""
        add(f"{'  ' * depth}  -> {node['name']} {node['input_shapes']}{kernels}")
        for c in node["children"]:
            tree(c, depth + 1)

    for root in tr["profiler_tree"]:
        for c in root["children"]:
            tree(c, 1)
    add("```\n")
    add("What reaches the CUDA backend (`TorchDispatchMode`, below autograd):\n")
    add("| aten op | inputs (shape / stride) | output (shape / stride) |")
    add("|---|---|---|")
    for c in tr["dispatch_ops"]:
        ins = "; ".join(f"{i['shape']} / {i['stride']}" for i in c["inputs"])
        o = c["output"]
        add(f"| `{c['overload']}` | {ins} | "
            f"{o['shape'] if o else ''} / {o['stride'] if o else ''} |")
    add("")
    w, wt = tr["weight"], tr["weight_t"]
    add(f"- `aten::linear` is not a kernel. With no bias and a 3-D input it becomes "
        f"`t` (a free transpose *view*: weight {w['shape']} strides {w['stride']} -> "
        f"{wt['shape']} strides {wt['stride']}, no copy, not contiguous), `matmul` folds "
        f"`[1, 128, K]` into `[128, K]` (`view`, free), then **one `aten::mm`**, then "
        "`_unsafe_view` back to 3-D. (With a bias it would be `aten::addmm` instead.)")
    add("- `aten::mm` on CUDA calls `at::cuda::blas::gemm`, i.e. `cublasGemmEx` / "
        "`cublasLtMatmul`. cuBLAS is column-major, so PyTorch asks for C^T = W · x^T: the "
        "weight's transposed view is passed as-is with op `T`, the activations with op `N` "
        "— that is the `_tn` suffix in the kernel name. No data is moved to make this work.")
    add("- The LM head is written as `x @ token_embedding.weight.T`, so it enters at "
        "`aten::matmul` directly but ends in the same `aten::mm`.\n")

    # ------------------------------------------------------------------
    add("## 2. Why is prefill GEMM-heavy?\n")
    add(f"With 128 tokens every Linear is a real matrix-matrix product. For gate/up "
        f"(`[128,512]x[512,1408]`): {fmt_flops(gu128['flops'])}, minimum traffic "
        f"{fmt_bytes(gu128['bytes'])}, arithmetic intensity "
        f"**{gu128['arithmetic_intensity']:.0f} FLOP/byte** — above the ridge point "
        f"({roof['ridge_point_fp32']:.0f}), so the math, not DRAM, is the limit. Each weight "
        "value loaded into shared memory is reused for many tokens (every row of the tile).\n")
    add(f"Measured in isolation it runs `{gu128['main_kernel']}` in "
        f"{gu128['kernel_us']:.1f} us of GPU time = **{gu128['tflops']:.2f} TFLOPS** "
        f"({100 * gu128['tflops'] / roof['peak_fp32_tflops']:.0f}% of this GPU's measured "
        f"FP32 peak). Over a whole forward the isolated matmul kernels add up to "
        f"{fwd['prefill']['matmul_kernel_ms']:.2f} ms"
        + (f" vs {ref['prefill_128']['matmul_ms']:.2f} ms of matmul kernels measured in the "
           f"full model in Phase 13 ({100 * ref['prefill_128']['matmul_share']:.1f}% of "
           "prefill GPU time)" if "prefill_128" in ref else "")
        + ". GEMMs dominate prefill simply because they are where the FLOPs are: "
        "RMSNorm/RoPE/SiLU are O(T·d), matmuls O(T·d²).\n")

    # ------------------------------------------------------------------
    add("## 3. Why is decode GEMV-heavy?\n")
    add(f"With one new token, M = 1: `[1,512]x[512,1408]` is a matrix-**vector** product. "
        f"cuBLAS recognises this and does not launch a tiled GEMM at all — it launches "
        f"`{gu1['main_kernel']}`. Its arithmetic intensity is "
        f"**{gu1['arithmetic_intensity']:.2f} FLOP/byte**: every weight is read from DRAM "
        "once and used for exactly one multiply-add. Nothing can be reused, so the kernel "
        "is a pure weight-streaming loop.\n")
    add("For 2 <= M <= ~16 cuBLAS uses `gemmSN_TN_kernel` — a *small-N* GEMM (cuBLAS "
        "sees C^T[N,M], so its \"N\" is our M): still essentially one pass over the weights, "
        "with each weight reused for the few rows. Only from M ~ 16-32 does it switch to "
        "tiled `ampere_sgemm_*` kernels that stage tiles through shared memory.\n")
    add("Kernel chosen per M (fp32, gate/up shape `K=512, N=1408`):\n")
    add("| M | kernel(s) | kernels | GPU us | TFLOPS | GB/s | tiles (CTAs) vs "
        f"{sms} SMs | split-K |")
    add("|---|---|---|---|---|---|---|---|")
    for p in gu:
        names = ", ".join(dict.fromkeys(k["short"] for k in p["kernels"]))
        add(f"| {p['M']} | `{names}` | {p['kernel_count']:.0f} | {p['kernel_us']:.1f} | "
            f"{p['tflops']:.3f} | {p['gbs']:.0f} | {p['ctas'] if p['ctas'] else '—'} | "
            f"{'yes' if p['split_k'] else ''} |")
    add("")
    add(f"At M = 1 the gate/up gemv moves {fmt_bytes(gu1['bytes'])} in "
        f"{gu1['kernel_us']:.1f} us = {gu1['gbs']:.0f} GB/s "
        f"({100 * gu1['gbs'] / roof['dram_bandwidth_gbs']:.0f}% of measured DRAM bandwidth), "
        f"but only {gu1['tflops'] * 1e3:.1f} GFLOPS. The LM-head gemv "
        f"(`[1,512]x[512,{lm1['N']}]`, {fmt_bytes(lm1['weight_bytes'])} of weights) runs at "
        f"{lm1['gbs']:.0f} GB/s ({lm1['kernel_us']:.0f} us). Small gemvs (q/k/v/out) are "
        "too short to reach full bandwidth at all: they are over before the memory "
        "pipeline fills, and the weights may even sit in the 1 MB L2.\n")

    # ------------------------------------------------------------------
    add("## 4. Why is decode GPU utilization lower?\n")
    if ref:
        add(f"Phase 13 measured GPU utilization "
            f"{100 * ref['prefill_128']['gpu_util']:.1f}% for prefill vs "
            f"{100 * ref['decode_20']['gpu_util']:.1f}% for decode, with a similar kernel "
            f"count per forward ({ref['prefill_128']['kernels_per_step']} vs "
            f"{ref['decode_20']['kernels_per_step']}).\n")
    tiny = roof["tiny_kernel_wall_us"]
    add("In the isolated loops below the only work is one Linear, so wall time per call = "
        "max(CPU cost to issue it, GPU time to run it):\n")
    add("| shape | M | wall us/call | GPU kernel us | GPU busy |")
    add("|---|---|---|---|---|")
    for name in ("q_proj", "k_proj/v_proj", "out_proj", "gate_proj/up_proj", "down_proj",
                 "lm_head"):
        for M in (1, 128):
            p = at(sweep[name], M)
            add(f"| {name} | {M} | {p['wall_us']:.1f} | {p['kernel_us']:.1f} | "
                f"{100 * min(p['gpu_busy_fraction'], 1):.0f}% |")
    add("")
    short = [s_ for s_ in linear_shapes_from_table(table) if at(sweep[s_["name"]], 1)["kernel_us"] < tiny]
    n_short = sum(s_["count"] for s_ in short)
    n_mm = sum(row["count"] for row in table)
    add(f"1. **Launch/dispatch bound.** Issuing one kernel costs the CPU ~{tiny:.1f} us. "
        f"In decode, {n_short} of the {n_mm} matmuls per forward "
        f"({', '.join(s_['name'] for s_ in short) or 'none'}) run for *less* GPU time than "
        "that, and so do almost all of the other kernels (RMSNorm, RoPE, SiLU, residual "
        "adds, cats: 1-5 us each). The GPU finishes each one and then waits for the next "
        "launch. In prefill the same kernels are fed 128 rows, the GEMMs run for tens to "
        "hundreds of us, and launches are hidden behind GPU work.")
    if ref:
        dec, pre = ref["decode_20"], ref["prefill_128"]
        issue = at(sweep["k_proj/v_proj"], 1)["wall_us"]
        rows_ = []
        for label, x in (("prefill", pre), ("decode", dec)):
            n = x["kernels_per_step"]
            rows_.append(f"| {label} | {n} | {1e3 * x['wall_ms'] / n:.1f} | "
                         f"{1e3 * x['gpu_busy_ms'] / n:.1f} | {100 * x['gpu_util']:.1f}% |")
        add("\n   Check against Phase 13 (whole forward, per kernel launch):\n")
        add("   | step | launches | wall us / launch | GPU us / kernel | GPU util |")
        add("   |---|---|---|---|---|")
        for row_ in rows_:
            add("   " + row_)
        add(f"\n   In decode the average kernel runs ~{1e3 * dec['gpu_busy_ms'] / dec['kernels_per_step']:.0f} us "
            f"but each launch takes ~{1e3 * dec['wall_ms'] / dec['kernels_per_step']:.0f} us of wall time — "
            f"the CPU cost of one PyTorch op (here: {tiny:.1f} us for a bare `add_`, "
            f"{issue:.1f} us for an `F.linear` whose kernel is only "
            f"{at(sweep['k_proj/v_proj'], 1)['kernel_us']:.1f} us; model ops also pay "
            "`nn.Module` / Python overhead). The GPU work per launch is smaller than the "
            "CPU cost per launch, so the GPU idles. In prefill the kernels are ~3x longer "
            "and cover the launch cost, so the CPU runs ahead and the GPU stays fed.")
    add(f"\n2. **Too little parallel work per kernel.** At M = 128 a gate/up GEMM does "
        f"{gu128['flops'] / gu1['flops']:.0f}x more math than at M = 1 for the same weight "
        f"bytes, and runs at {gu128['tflops']:.2f} TFLOPS. At M = 1 there is only enough "
        f"work to saturate DRAM, not the SMs: {gu1['tflops'] * 1e3:.0f} GFLOPS, "
        f"{100 * gu1['tflops'] / roof['peak_fp32_tflops']:.1f}% of peak. Even a \"busy\" "
        "decode GPU is mostly waiting on memory.")
    add("\nSo low decode utilization is mostly the GPU **waiting for the CPU**, and when it "
        "does run, it runs memory-bound kernels. Faster GEMM math would fix neither; fewer "
        "launches (fusion, CUDA graphs) and more rows per launch (batching) would.\n")

    # ------------------------------------------------------------------
    add("## 5. Why does batching improve decode?\n")
    add(f"`[B,512]x[512,1408]` (gate/up) as B separate M=1 calls vs one M=B call:\n")
    add("| B | separate: wall us | kernels | batched: wall us | kernels | batched kernel | "
        "speedup | batched wall us / token |")
    add("|---|---|---|---|---|---|---|---|")
    for b in bat:
        add(f"| {b['B']} | {b['separate_wall_us']:.1f} | {b['separate_kernel_count']:.0f} | "
            f"{b['batched_wall_us']:.1f} | {b['batched_kernel_count']:.0f} | "
            f"`{b['batched_kernel']}` | {b['speedup_wall']:.1f}x | "
            f"{b['wall_us_per_token_batched']:.2f} |")
    add("")
    add("The M sweep shows the same thing from the GPU side: the kernel time is almost flat "
        "while M is small, because the cost is reading the weight matrix, and that is paid "
        "once per call no matter how many rows share it:\n")
    add("| shape | " + " | ".join(f"M={M}" for M in M_VALUES) + " |")
    add("|---|" + "---|" * len(M_VALUES))
    for name, points in sweep.items():
        add(f"| {name} GPU us | " + " | ".join(f"{p['kernel_us']:.1f}" for p in points) + " |")
    for name, points in sweep.items():
        add(f"| {name} TFLOPS | " + " | ".join(f"{p['tflops']:.2f}" for p in points) + " |")
    add("")
    bumps = [(name, a["M"], a["kernel_us"], b["M"], b["kernel_us"])
             for name, pts in sweep.items() for a, b in zip(pts, pts[1:])
             if a["kernel_us"] > 1.15 * b["kernel_us"] and a["kernel_us"] > 5]
    add("Caveat: the isolated loops call the same weights back-to-back, so shapes whose "
        "weights fit in the 1 MB L2 (q/k/v/out: <= 1 MB) can look faster than DRAM allows; "
        "in the model the other layers' weights evict them between calls.\n")
    if bumps:
        add("Non-monotonic points — a *larger* M running faster — show that the kernel "
            "choice comes from a heuristic, not from timing every candidate: "
            + "; ".join(f"{n} M={m1} {t1:.1f} us > M={m2} {t2:.1f} us" for n, m1, t1, m2, t2 in bumps)
            + ".\n")
    add("Continuous batching (Phase 11) turns B requests' `[1,K]` rows into one `[B,K]` "
        "operand: one launch instead of B, one pass over the weights instead of B, and "
        "arithmetic intensity grows ~linearly with B until the kernel becomes compute-bound. "
        "This is why decode throughput scales with batch size almost for free until M "
        "reaches the tens.\n")

    # ------------------------------------------------------------------
    add("## 6. What is cuBLAS / cuBLASLt doing?\n")
    kernels_seen = sorted({k["short"] for pts in sweep.values() for p in pts
                           for k in p["kernels"] if "Memset" not in k["name"]})
    add("- **cuBLAS** is NVIDIA's BLAS: the classic `cublasSgemm` / `cublasGemmEx` API with "
        "heuristics that pick one of many precompiled kernels.")
    add("- **cuBLASLt** (\"lightweight\") is the newer matmul-only API (`cublasLtMatmul`) "
        "with explicit matrix layouts, epilogues (bias, activation fusion), workspaces and "
        "algorithm search. It exposes the split-K algorithms, whose reduction kernel "
        "carries the `cublasLt::` namespace. PyTorch uses cuBLASLt for many fp32/fp16 "
        "matmuls (and always for `addmm` with bias epilogues) and cuBLAS otherwise; both "
        "share the same kernel library underneath.")
    add("- Per call, the library looks at (M, N, K), dtype, transposes/strides (`tn`), "
        "alignment, the math mode (FP32 vs TF32 allowed), available workspace and the GPU "
        "architecture (sm_86 here), and picks an algorithm: GEMV for M=1, a tile size, "
        "whether to split K. That is why one `aten::mm` produced these different kernels "
        "across the sweep:\n")
    for k in kernels_seen:
        add(f"  - `{k}`")
    add("")

    # ------------------------------------------------------------------
    add("## 7. What does `ampere_sgemm_128x64_tn` mean?\n")
    add("| part | meaning |")
    add("|---|---|")
    add("| `ampere` | kernel written for the Ampere architecture (sm_80 family); the RTX "
        "3050 is sm_86 and runs it |")
    add("| `sgemm` | **S**ingle-precision (FP32) GEMM, executed with FFMA instructions on "
        "the CUDA cores — not Tensor Cores (those kernels are named `*_tf32_*`, "
        "`*_s1688gemm_*`, `*_16816gemm_*`, `cutlass_*_tensorop_*`) |")
    add("| `128x64` | each thread block (CTA) computes a 128 x 64 tile of the output. "
        "Because cuBLAS works on C^T, the 128 runs along N (output features) and the "
        "64 along M (tokens) |")
    add("| `tn` | operand ops: first operand (our weight view) **T**ransposed, second "
        "(activations) **N**ot transposed — the natural result of `x @ W.T` on row-major "
        "tensors |")
    add("")
    add(f"For gate/up at M=128 that is ceil(1408/128) x ceil(128/64) = "
        f"{output_tiles('ampere_sgemm_128x64_tn', 128, 1408)} CTAs for {sms} SMs — barely one "
        "block per SM, far fewer than the GPU can hold at once, so per-SM latency hiding is "
        "poor. That is the situation where cuBLAS reaches for smaller tiles or split-K "
        "(section 8). Variants like "
        "`ampere_sgemm_32x32_sliced1x4_tn` use a smaller tile (more CTAs for small "
        "matrices) and split K across the 4 warps inside a block (`sliced1x4`).\n")

    # ------------------------------------------------------------------
    add("## 8. What is `cublasLt::splitKreduce_kernel` doing?\n")
    sk = [(name, p["M"]) for name, pts in sweep.items() for p in pts if p["split_k"]]
    add("Split-K: when the output has too few tiles to fill the GPU but K is long, the "
        "library splits the K (reduction) dimension into S slices. S x more CTAs each "
        "compute a partial `[tile]` product over K/S, write it to a workspace, and a "
        "second small kernel — `splitKreduce_kernel` — sums the S partials (and applies "
        "alpha/beta) into C.\n")
    add("```text")
    add("C[M,N] = A[M, K0:K1] B[K0:K1, N] + A[M, K1:K2] B[K1:K2, N] + ...")
    add("         └── CTA group 1 ──┘      └── CTA group 2 ──┘")
    add("                    └──── splitKreduce_kernel ────┘")
    add("```\n")
    add("Cost: an extra launch and an extra pass over the partials in DRAM; benefit: "
        "enough CTAs to occupy every SM. In this run split-K was chosen for: "
        + (", ".join(f"{n} M={M}" for n, M in sk) if sk else "none of the isolated shapes")
        + ". Phase 13 saw it on 40 of the 57 prefill matmuls.\n")
    add("How much of the matmul time the reduction itself takes (M=128):\n")
    add("| shape | GEMM kernel | total us | splitKreduce us | share |")
    add("|---|---|---|---|---|")
    for name, pts in sweep.items():
        p = at(pts, 128)
        if p["split_k"]:
            add(f"| {name} | `{p['main_kernel']}` | {p['kernel_us']:.1f} | "
                f"{p['split_k_us']:.1f} | {100 * p['split_k_us'] / p['kernel_us']:.0f}% |")
    add("")
    ms = [f"{n} M={p['M']}" for n, pts in sweep.items() for p in pts
          if p["memset"] and not p["split_k"]]
    if ms:
        add("Some GEMMs instead launch a `Memset (Device)` right before the sgemm and no "
            "reduce kernel (" + ", ".join(ms[:6]) + (", ..." if len(ms) > 6 else "") + "). "
            "This is most likely cuBLASLt clearing its workspace / output for a split-K "
            "variant that accumulates partials in place; the profiler alone cannot confirm "
            "which algorithm was chosen (that would need `CUBLASLT_LOG_LEVEL`).\n")

    # ------------------------------------------------------------------
    add("## 9. What role do Tensor Cores / CUDA cores play?\n")
    add("- **CUDA cores** execute scalar FP32 FFMA. All fp32 kernels above (`*sgemm*`, "
        "`gemv*`) run here. This is what the model uses today because PyTorch's default "
        f"`float32_matmul_precision` is `\"highest\"` (TF32 off).")
    add("- **Tensor Cores** execute small matrix-multiply-accumulate tiles per instruction. "
        "On Ampere they accept FP16/BF16/INT8 and **TF32** (FP32 range, 10-bit mantissa) "
        "inputs, accumulating in FP32.")
    add(f"- Measured peak on this GPU: FP32 {roof['peak_fp32_tflops']:.2f} TFLOPS "
        f"(`{short_kernel(roof['peak_fp32_kernel'] or '')}`) vs TF32 "
        f"{roof['peak_tf32_tflops']:.2f} TFLOPS "
        f"(`{short_kernel(roof['peak_tf32_kernel'] or '')}`).\n")
    add("Same model shapes with TF32 allowed (observation only — the model still runs FP32):\n")
    add("| shape | M | FP32 kernel | us | TF32 kernel | us | speedup |")
    add("|---|---|---|---|---|---|---|")
    for name, points in tf32.items():
        for p in points:
            base = at(sweep[name], p["M"])
            same = base["main_kernel"] == p["main_kernel"]
            tf32_kernel = "(same kernel)" if same else f"`{p['main_kernel']}`"
            speedup = "noise" if same else f"{base['kernel_us'] / p['kernel_us']:.2f}x"
            add(f"| {name} | {p['M']} | `{base['main_kernel']}` | {base['kernel_us']:.1f} | "
                f"{tf32_kernel} | {p['kernel_us']:.1f} | {speedup} |")
    add("")
    add("Tensor Cores only raise the compute ceiling. They help prefill GEMMs (compute-bound) "
        "and do nothing for M=1 gemvs, which are bound by reading FP32 weights — there the "
        "lever is fewer bytes (FP16/BF16/INT8 weights), not faster math.\n")

    # ------------------------------------------------------------------
    add("## 10. What did the profiler teach us about our own inference engine?\n")
    dp = kv["decode_sim"]
    fg = kv["full_generation"]
    add("**Memory layout of the real tensors** (layer 0, from forward hooks):\n")
    add("| phase | layer | input shape | input stride | contiguous | weight stride | mm operands |")
    add("|---|---|---|---|---|---|---|")
    for t in r["model_tensors"]:
        add(f"| {t['phase']} | {t['layer']} | {t['input']['shape']} | {t['input']['stride']} | "
            f"{t['input']['contiguous']} | {t['weight']['stride']} | `{t['mm_operands']}` |")
    add("")
    add("All activations entering Linears are contiguous row-major fp32 and all weights are "
        "contiguous `[N, K]`; the only transpose is the free `W.t()` view that cuBLAS "
        "consumes as op `T`. No hidden `.contiguous()` copies happen before any matmul.\n")
    add("**The KV-cache `torch.cat` problem.** `KVCache.update` does "
        "`torch.cat([old, new], dim=2)` for K and V in every layer at every step: allocate "
        "a `[1, 2, T+1, 64]` tensor, copy T old rows, copy 1 new row.\n")
    add("| cached tokens T | GPU us per cat | wall us per cat | bytes copied | useful bytes |")
    add("|---|---|---|---|---|")
    for p in kv["per_length"]:
        add(f"| {p['T']} | {p['kernel_us']:.2f} | {p['wall_us']:.1f} | "
            f"{fmt_bytes(p['bytes_copied'])} | {fmt_bytes(p['useful_bytes'])} |")
    add("")
    add(f"- Per decode step: {dp['cat_calls'] // dp['steps']} `cat` calls, "
        f"{dp['allocations_per_step']:.0f} new CUDA allocations "
        f"(caching allocator, so no `cudaMalloc`, but a fresh block each time), "
        f"{dp['stream_ms_per_step']:.3f} ms of stream time with a {dp['prompt_len']}-token "
        "prompt.")
    add(f"- Generating {fg['new_tokens']} tokens after a {fg['prompt_len']}-token prompt "
        f"copies **{fmt_bytes(fg['bytes_copied'])}** to append only "
        f"{fmt_bytes(fg['bytes_new_kv'])} of new K/V "
        f"({fg['bytes_copied'] / fg['bytes_new_kv']:.0f}x) — quadratic in sequence length. "
        f"The whole cache at the end is only {fmt_bytes(fg['cache_bytes_at_end'])}.")
    add("- At this model size each cat is so small that its cost is the launch, not the "
        "copy; the quadratic byte growth would dominate for long contexts and bigger "
        "models. Phase 15 question: preallocate and write in place.\n")
    add("**Summary**\n")
    add(f"- Prefill: {fmt_flops(fwd['prefill']['matmul_flops'])} of matmul per forward, "
        f"compute-bound FP32 SGEMMs on CUDA cores at "
        f"{fwd['prefill']['achieved_tflops']:.2f} TFLOPS; the GPU is busy.")
    add(f"- Decode: {fmt_flops(fwd['decode']['matmul_flops'])} per token; the "
        f"{fmt_bytes(fwd['decode']['weight_bytes'])} of weights must be streamed from DRAM "
        f"every step (~{fwd['decode']['weight_bytes'] / (roof['dram_bandwidth_gbs'] * 1e9) * 1e3:.2f} "
        f"ms at full bandwidth, {fwd['decode']['matmul_kernel_ms']:.2f} ms measured in "
        "isolated gemvs); and ~57 matmul launches plus ~370 small elementwise/copy launches "
        "cost more CPU time than the GPU spends computing.")
    add("- Levers for Phase 15, in order of expected payoff for decode: fewer launches "
        "(CUDA graphs / fusion / torch.compile), bigger M (batching), fewer weight bytes "
        "(lower precision), in-place KV cache. Faster GEMM math (TF32) mainly helps prefill.")

    path.write_text("\n".join(lines) + "\n")


# ======================================================================
# Main
# ======================================================================


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--out-dir", type=str, default=str(HERE))
    parser.add_argument("--iters", type=int, default=200, help="timed calls per measurement")
    parser.add_argument("--quick", action="store_true", help="--iters 30")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("Phase 14 measures CUDA kernels: a GPU is required.")
    iters = 30 if args.quick else args.iters
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    config = ModelConfig()
    print("Loading model ...")
    model, checkpoint = load_model(args, config)
    config = model.config
    props = torch.cuda.get_device_properties(0)

    results = {
        "meta": {
            "device": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_memory_mb": props.total_memory // 2 ** 20,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "fp32_matmul_precision": torch.get_float32_matmul_precision(),
            "checkpoint": checkpoint,
            "config": {k: getattr(config, k) for k in (
                "vocab_size", "max_seq_len", "hidden_dim", "num_layers",
                "num_q_heads", "num_kv_heads", "ffn_dim")},
            "iters": iters,
        },
    }

    steps = [
        ("trace", "tracing nn.Linear -> aten -> kernel",
         lambda: trace_linear(config.hidden_dim, config.ffn_dim, 128, args.device)),
        ("model_tensors", "inspecting real model tensors",
         lambda: model_tensors(model, args.device)),
        ("layer_table", "building the model matmul table", lambda: layer_table(config)),
        ("roofline", "measuring peak FP32/TF32, DRAM bandwidth, launch cost",
         lambda: roofline(iters)),
        ("m_sweep", "M sweep (FP32)", lambda: m_sweep(config, iters)),
        ("tf32_sweep", "M sweep with TF32 allowed (M=1, 128)",
         lambda: m_sweep(config, iters, precision="high", m_values=[1, 128])),
        ("batching", "batching B x [1,K] vs [B,K]", lambda: batching(config, iters)),
        ("kv_cat", "KV-cache torch.cat", lambda: kv_cat(config, iters)),
    ]
    for key, label, fn in steps:
        print(f"  {label} ...")
        results[key] = fn()

    del model
    roof = results["roofline"]
    best = max((p for pts in results["m_sweep"].values() for p in pts), key=lambda p: p["tflops"])
    roof["peak_fp32_source"] = "square GEMM"
    if best["tflops"] > roof["peak_fp32_tflops"]:
        roof["peak_fp32_tflops"] = best["tflops"]
        roof["peak_fp32_kernel"] = best["main_kernel"]
        roof["peak_fp32_source"] = f"M sweep, [{best['M']},{best['K']}]x[{best['K']},{best['N']}]"
    roof["ridge_point_fp32"] = roof["peak_fp32_tflops"] * 1e12 / (roof["dram_bandwidth_gbs"] * 1e9)
    results["forward_estimate"] = forward_estimate(config, results["m_sweep"])
    results["phase13"] = phase13_reference()
    results["meta"]["fp32_matmul_precision_after"] = torch.get_float32_matmul_precision()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_findings(out_dir / "findings.md", results)

    print(f"\n{'shape':<20} {'M':>5} {'kernel':<40} {'GPU us':>8} {'wall us':>8} {'TFLOPS':>7}")
    for name, points in results["m_sweep"].items():
        for p in points:
            print(f"{name:<20} {p['M']:>5} {p['main_kernel'][:40]:<40} "
                  f"{p['kernel_us']:>8.1f} {p['wall_us']:>8.1f} {p['tflops']:>7.3f}")
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'findings.md'}")


if __name__ == "__main__":
    main()
