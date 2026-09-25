"""
Phase 15 — kernel-level optimization, one change at a time.

Phase 13 found where decode time goes (~424 tiny kernels/step, 37% GPU
busy, torch.cat KV growth); Phase 14 explained why (M=1 GEMVs are
memory-bound, and every launch costs more CPU time than its kernel
runs). This script applies the candidate fixes as an optimization
ladder, never more than one change per rung:

    baseline                Phase 13 code path: dynamic KVCache, all
                            fast paths off, eager PyTorch
      -> for each candidate (on top of everything kept so far):
             1. make ONE change      (StaticKVCache, a fast_paths flag,
                                      or torch.compile)
             2. correctness gate     src/inference/correctness.py vs the
                                      baseline: logits, greedy tokens,
                                      EOS, KV cache, batched requests
             3. benchmark            prefill / decode wall time (median)
             4. profile              kernels, launches, categories, regions
             5. keep or reject       kept only if the gate passes AND a
                                      measured speedup beats the noise
                                      threshold without a regression

Side experiments (not rungs of the ladder):

    batching         B = 1..16 decode rows at once: latency per request,
                     aggregate tokens/s, GEMV -> GEMM transition
    attention        what SDPA runs for decode / prefill, mask vs no
                     mask, repeat_interleave vs enable_gqa, math backend,
                     vs KV length
    launch_overhead  1..1000 kernels doing the same total work
    kernel inventory which operations launch the most kernels per
                     decode step (baseline vs final eager config)

Output (next to this file):

    <rung>/prefill/, <rung>/decode/   trace.json (Perfetto), ops.txt, shapes.txt
    batching/, attention/             their profiles
    results.json                      every number
    FINDINGS.md                       generated from results.json

Usage:

    python profiles/phase15/phase15_benchmark.py
    python profiles/phase15/phase15_benchmark.py --quick         # fewer repeats
    python profiles/phase15/phase15_benchmark.py --skip-compile  # no torch.compile rung
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.profiler import profile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from scripts.profile_inference import LEAF_REGIONS, TEXT  # noqa: E402
from src.inference.checkpoint_loader import (  # noqa: E402
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.correctness import Tolerance, Variant, run_gate  # noqa: E402
from src.inference.model_runner import ModelRunner  # noqa: E402
from src.inference.profiler import (  # noqa: E402
    categorize_kernel,
    profile_workload,
    profiler_activities,
)
from src.inference.request import SamplingParams  # noqa: E402
from src.model import fast_paths, profiling  # noqa: E402
from src.model.profiling import region  # noqa: E402
from src.tokenizer.tokenizer import BPETokenizer  # noqa: E402

BATCH_SIZES = [1, 2, 4, 8, 16]
KV_CONTEXTS = [128, 256, 448]
ATTENTION_KV_LENGTHS = [16, 64, 128, 256, 511]
LAUNCH_COUNTS = [1, 10, 100, 1000]
LAUNCH_EVENT = re.compile(r"^(cudaLaunchKernel|cuLaunchKernel|cudaLaunchKernelExC|"
                          r"cudaMemsetAsync|cudaMemcpyAsync|cudaGraphLaunch)")

# ======================================================================
# Configurations
# ======================================================================


@dataclass(frozen=True)
class Config:
    """How to run the model: KV-cache kind, fast-path flags, compile or not."""

    kv_cache: str = "dynamic"
    flags: tuple = ()
    compile: bool = False

    def with_change(self, change: dict) -> "Config":
        if "kv_cache" in change:
            return replace(self, kv_cache=change["kv_cache"])
        if "flag" in change:
            return replace(self, flags=tuple(sorted(set(self.flags) | {change["flag"]})))
        if "compile" in change:
            return replace(self, compile=True)
        raise ValueError(change)

    def describe(self) -> str:
        parts = [f"kv={self.kv_cache}"] + list(self.flags) + (["compile"] if self.compile else [])
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {"kv_cache": self.kv_cache, "flags": list(self.flags), "compile": self.compile}


# One change each. Order: the biggest known bottleneck first (15.3), then
# launch reductions (15.6/15.7), unnecessary work (15.12), compiler (15.8).
CANDIDATES = [
    ("kv_preallocated", {"kv_cache": "static"},
     "StaticKVCache: preallocate [B,H,512,D], write in place (no torch.cat)"),
    ("rope_cache", {"flag": "rope_cache"},
     "precomputed interleaved cos/sin (drops 4 repeat_interleave / layer)"),
    ("fused_rmsnorm", {"flag": "fused_rmsnorm"},
     "F.rms_norm: 1 kernel instead of 6 per norm"),
    ("decode_no_mask", {"flag": "decode_no_mask"},
     "no arange/compare mask for a single unpadded decode query"),
    ("sdpa_gqa", {"flag": "sdpa_gqa"},
     "SDPA enable_gqa instead of repeat_interleave K/V copies"),
    ("fused_qkv", {"flag": "fused_qkv"},
     "one q+k+v matmul instead of three"),
    ("fused_gate_up", {"flag": "fused_gate_up"},
     "one gate+up matmul instead of two"),
    ("last_token_logits", {"flag": "last_token_logits"},
     "final norm + LM head on the last position only"),
    ("torch_compile", {"compile": True},
     "torch.compile(model, dynamic=True) on top of the kept eager set"),
]


# ======================================================================
# Lab: model, tokens, runners
# ======================================================================


def sync():
    torch.cuda.synchronize()


def median(values):
    return statistics.median(values) if values else 0.0


class Lab:
    def __init__(self, args):
        self.args = args
        self.device = "cuda"
        self.checkpoint = args.checkpoint or str(find_latest_checkpoint(ROOT / args.checkpoint_dir))
        self.model, _ = load_inference_checkpoint(self.checkpoint, device=self.device)
        fast_paths.prepare(self.model)   # fused weight buffers (views; no extra memory)
        self.tokenizer = BPETokenizer(str(ROOT / args.tokenizer_path))
        self.sampler = SamplingParams(greedy=True).to_sampler()
        self._compiled = None

        base = self.tokenizer.encode(TEXT)
        self.pool = (base * (1 + 2048 // len(base)))[:2048]

    # ---- token helpers
    def ids(self, n: int, batch: int = 1) -> torch.Tensor:
        rows = [self.pool[7 * b:7 * b + n] for b in range(batch)]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def gate_prompts(self) -> list[list[int]]:
        return [self.pool[0:9], self.pool[40:77], self.pool[100:228]]

    # ---- runners
    def compiled_model(self):
        if self._compiled is None:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self._compiled = torch.compile(self.model, dynamic=True)
        return self._compiled

    def runner(self, cfg: Config) -> ModelRunner:
        model = self.compiled_model() if cfg.compile else self.model
        runner = ModelRunner(model, self.tokenizer, self.sampler, device=self.device,
                             kv_cache=cfg.kv_cache)
        runner.max_seq_len = self.model.config.max_seq_len
        return runner

    def variant(self, name: str, cfg: Config) -> Variant:
        return Variant(name, self.runner(cfg), {f: True for f in cfg.flags})


# ======================================================================
# Workloads (identical to Phase 13's prefill_128 / decode_20)
# ======================================================================


def prefill_workload(runner, ids):
    @torch.inference_mode()
    def fn():
        out = runner.prefill(ids)
        with region("sampling"):
            out.logits.argmax(dim=-1).tolist()
    return fn


def decode_workload(runner, ids, steps):
    @torch.inference_mode()
    def setup():
        out = runner.prefill(ids)
        return out.kv_cache, out.logits.argmax(dim=-1, keepdim=True)

    @torch.inference_mode()
    def fn(state):
        cache, token = state
        for _ in range(steps):
            out = runner.decode(token, cache)
            with region("sampling"):
                token = out.logits.argmax(dim=-1, keepdim=True)
                token.tolist()

    return setup, fn


def time_ms(fn, setup=None, repeats=10, warmup=2) -> list[float]:
    for _ in range(warmup):
        fn(setup()) if setup else fn()
    times = []
    for _ in range(repeats):
        state = setup() if setup else None
        sync()
        start = time.perf_counter()
        fn(state) if setup else fn()
        sync()
        times.append((time.perf_counter() - start) * 1e3)
    return times


def summarize_profile(result) -> dict:
    cats = result.categories()
    return {
        "gpu_busy_ms_per_step": result.gpu_busy_ms_per_step,
        "launches_per_step": result.launches_per_step,
        "categories_ms_per_step": {k: v / result.steps for k, v in cats.items()},
        "regions": {
            name: {"calls_per_step": st.calls / result.steps,
                   "gpu_ms_per_step": st.gpu_ms / result.steps,
                   "cpu_ms_per_step": st.cpu_ms / result.steps}
            for name, st in result.regions.items()
        },
        "top_kernels": [
            {"name": k.name[:160], "category": k.category, "calls_per_step": k.calls / result.steps,
             "gpu_ms_per_step": k.gpu_ms / result.steps}
            for k in result.top_kernels(12)
        ],
        "files": {k: str(Path(v).relative_to(ROOT)) for k, v in result.files.items()},
    }


@torch.inference_mode()
def generation_run(runner, ids, new_tokens) -> dict:
    """End-to-end: prefill + greedy decode, token read back every step (like the engine)."""

    sync()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    out = runner.prefill(ids)
    token = out.logits.argmax(dim=-1, keepdim=True)
    token.tolist()
    ttft = time.perf_counter() - start
    for _ in range(new_tokens - 1):
        out = runner.decode(token, out.kv_cache)
        token = out.logits.argmax(dim=-1, keepdim=True)
        token.tolist()
    sync()
    total = time.perf_counter() - start
    return {
        "ttft_ms": ttft * 1e3,
        "total_ms": total * 1e3,
        "decode_tokens_per_s": (new_tokens - 1) / (total - ttft) if total > ttft else 0.0,
        "peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
        "transient_mb": (torch.cuda.max_memory_allocated() - before) / 2 ** 20,
    }


def measure(lab: Lab, cfg: Config, out_dir: Path, repeats: int) -> dict:
    """Cold start, steady-state timings, profiles and memory for one configuration."""

    args = lab.args
    runner = lab.runner(cfg)
    ids = lab.ids(args.prompt_len)
    flags = {f: True for f in cfg.flags}
    result = {"config": cfg.to_dict()}

    with fast_paths.enabled(**flags):
        # cold: the very first calls (torch.compile compiles here)
        with torch.inference_mode():
            sync()
            t0 = time.perf_counter()
            out = runner.prefill(ids)
            out.logits.argmax(-1).tolist()
            t1 = time.perf_counter()
            token = out.logits.argmax(dim=-1, keepdim=True)
            cold_decode = []
            for _ in range(4):
                s = time.perf_counter()
                out = runner.decode(token, out.kv_cache)
                token = out.logits.argmax(dim=-1, keepdim=True)
                token.tolist()
                cold_decode.append((time.perf_counter() - s) * 1e3)
        result["cold"] = {"first_prefill_ms": (t1 - t0) * 1e3, "first_decode_steps_ms": cold_decode}

        pre = time_ms(prefill_workload(runner, ids), repeats=repeats * 2, warmup=3)
        setup, dec = decode_workload(runner, ids, args.decode_steps)
        dec_times = time_ms(dec, setup, repeats=repeats, warmup=2)

        result["prefill"] = {
            "wall_ms": median(pre),
            "wall_ms_all": pre,
            "tokens_per_s": args.prompt_len / (median(pre) / 1e3),
        }
        result["decode"] = {
            "wall_ms_per_token": median(dec_times) / args.decode_steps,
            "wall_ms_all": [t / args.decode_steps for t in dec_times],
            "tokens_per_s": args.decode_steps / (median(dec_times) / 1e3),
        }

        regions = not cfg.compile
        p = profile_workload("prefill", prefill_workload(runner, ids), lab.device, steps=1,
                             out_dir=out_dir / "prefill", timing_repeats=1, regions=regions)
        d = profile_workload("decode", dec, lab.device, steps=args.decode_steps, setup=setup,
                             out_dir=out_dir / "decode", timing_repeats=1, regions=regions)
        result["prefill"].update(summarize_profile(p))
        result["decode"].update(summarize_profile(d))
        for key in ("prefill", "decode"):
            r = result[key]
            wall = r["wall_ms"] if key == "prefill" else r["wall_ms_per_token"]
            r["gpu_utilization"] = r["gpu_busy_ms_per_step"] / wall if wall else 0.0

        result["generation"] = generation_run(runner, ids, args.gen_len)

    if cfg.compile:
        from torch._dynamo.utils import counters
        result["compile"] = {"unique_graphs": counters["stats"]["unique_graphs"],
                             "graph_breaks": sum(counters["graph_break"].values())}
    return result


# ======================================================================
# Paired A/B timing and keep / reject
# ======================================================================


def paired(lab: Lab, cfg_a: Config, cfg_b: Config, rounds: int, prompt_len: int | None = None) -> dict:
    """
    Interleaved A/B timing (A B B A A B ...) in the same time window.

    Laptop CPU/GPU clocks drift by tens of percent over a run, and
    decode is CPU-bound, so timings of two configs measured minutes
    apart are not comparable. Each round times one prefill (median of
    5 calls) and one 20-step decode for both configs back to back;
    the speedup is the median of the per-round B/A ratios.
    """

    args = lab.args
    ids = lab.ids(prompt_len or args.prompt_len)
    sides = {}
    for key, cfg in (("a", cfg_a), ("b", cfg_b)):
        runner = lab.runner(cfg)
        flags = {f: True for f in cfg.flags}
        setup, dec = decode_workload(runner, ids, args.decode_steps)
        sides[key] = (flags, prefill_workload(runner, ids), setup, dec)
        with fast_paths.enabled(**flags):
            time_ms(sides[key][1], repeats=2, warmup=2)
            time_ms(dec, setup, repeats=1, warmup=1)

    times = {"a": {"prefill": [], "decode": []}, "b": {"prefill": [], "decode": []}}
    for i in range(rounds):
        for key in (("a", "b") if i % 2 == 0 else ("b", "a")):
            flags, pre, setup, dec = sides[key]
            with fast_paths.enabled(**flags):
                times[key]["prefill"].append(median(time_ms(pre, repeats=5, warmup=0)))
                times[key]["decode"].append(
                    time_ms(dec, setup, repeats=1, warmup=0)[0] / args.decode_steps)

    out = {}
    for metric in ("prefill", "decode"):
        ratios = [b / a for a, b in zip(times["a"][metric], times["b"][metric])]
        out[metric] = {
            "a_ms": median(times["a"][metric]),
            "b_ms": median(times["b"][metric]),
            "ratio": median(ratios),          # < 1 means B is faster
            "speedup": 1 / median(ratios),
        }
    return out


def decide(pair: dict, gate, threshold: float) -> tuple[bool, str]:
    if not gate.passed:
        failed = sorted({c.name for c in gate.failures()})
        return False, f"rejected: correctness gate failed ({', '.join(failed)})"

    d_gain = 1 - pair["decode"]["ratio"]
    p_gain = 1 - pair["prefill"]["ratio"]
    text = f"decode {100 * d_gain:+.1f}%, prefill {100 * p_gain:+.1f}%"

    if d_gain < -threshold or p_gain < -threshold:
        return False, f"rejected: regression ({text})"
    if d_gain >= threshold or p_gain >= threshold:
        return True, f"kept: {text}"
    return False, f"rejected: within noise (±{100 * threshold:.0f}%; {text})"


# ======================================================================
# Kernel inventory (15.6 / 15.7)
# ======================================================================


def kernel_inventory(lab: Lab, cfg: Config) -> dict:
    """Kernel launches per decode step, attributed to (region, aten op)."""

    runner = lab.runner(cfg)
    ids = lab.ids(lab.args.prompt_len)
    setup, dec = decode_workload(runner, ids, 1)

    with fast_paths.enabled(**{f: True for f in cfg.flags}):
        dec(setup())
        state = setup()
        sync()
        with profiling.enabled(True), profile(activities=profiler_activities()) as prof:
            dec(state)
            sync()

    leaf = set(LEAF_REGIONS) | {"kv_cache/cat", "kv_cache/write"}
    per_region = defaultdict(int)
    per_op = defaultdict(int)

    for evt in prof.events():
        if not LAUNCH_EVENT.match(evt.name):
            continue
        op, node, reg = None, evt.cpu_parent, "(unlabelled)"
        while node is not None:
            if op is None and node.name.startswith("aten::"):
                op = node.name
            if node.name in leaf:
                reg = node.name
                # kv_cache/* is nested in attention/kv_cache; report the detail
                break
            node = node.cpu_parent
        per_region[reg] += 1
        per_op[(reg, op or "(runtime)")] += 1

    total = sum(per_region.values())
    return {
        "config": cfg.to_dict(),
        "launches": total,
        "by_region": dict(sorted(per_region.items(), key=lambda kv: -kv[1])),
        "by_op": [{"region": r, "op": o, "launches": n}
                  for (r, o), n in sorted(per_op.items(), key=lambda kv: -kv[1])[:30]],
    }


# ======================================================================
# 15.10 — batching
# ======================================================================


def batching(lab: Lab, cfg: Config, name: str, repeats: int) -> list[dict]:
    """
    Decode B rows at once. Timed in two passes (B ascending, then
    descending) and the lower median per B is kept, so a clock dip
    during one pass can't masquerade as a batching effect.
    """

    rows = []
    runner = lab.runner(cfg)
    steps = lab.args.decode_steps
    with fast_paths.enabled(**{f: True for f in cfg.flags}):
        workloads = {B: decode_workload(runner, lab.ids(lab.args.prompt_len, batch=B), steps)
                     for B in BATCH_SIZES}
        best = {}
        for order in (BATCH_SIZES, BATCH_SIZES[::-1]):
            for B in order:
                setup, dec = workloads[B]
                t = median(time_ms(dec, setup, repeats=repeats, warmup=1))
                best[B] = min(best.get(B, t), t)
        for B in BATCH_SIZES:
            setup, dec = workloads[B]
            times = [best[B]]
            prof = profile_workload(f"b{B}", dec, lab.device, steps=steps, setup=setup,
                                    out_dir=HERE / "batching" / f"{name}_b{B}", timing_repeats=1)
            step_ms = median(times) / steps
            cats = {k: v / steps for k, v in prof.categories().items()}
            matmul = [k for k in prof.top_kernels(40) if k.category in ("gemm", "gemv")]
            rows.append({
                "B": B,
                "step_ms": step_ms,
                "latency_ms_per_token_per_request": step_ms,
                "aggregate_tokens_per_s": B * 1000 / step_ms,
                "gpu_busy_ms_per_step": prof.gpu_busy_ms_per_step,
                "gpu_utilization": prof.gpu_busy_ms_per_step / step_ms,
                "launches_per_step": prof.launches_per_step,
                "gemm_ms_per_step": cats.get("gemm", 0.0),
                "gemv_ms_per_step": cats.get("gemv", 0.0),
                "matmul_kernels": sorted({short_kernel(k.name) for k in matmul[:6]}),
            })
            torch.cuda.empty_cache()
    return rows


# ======================================================================
# 15.11 — attention
# ======================================================================


FUNCTOR = re.compile(r"(\w+Functor\w*|\w+Func\b|\w+Op\b)")


def short_kernel(name: str) -> str:
    """'void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>...' -> 'vectorized_elementwise_kernel[FillFunctor]'."""

    clean = re.sub(r"^(void |std::enable_if<[^>]*>::type )", "", name)
    base = clean.split("<", 1)[0].split("(", 1)[0].split("::")[-1] or clean[:40]
    if base in ("kernel",) and "::" in clean.split("<", 1)[0]:
        base = "::".join(clean.split("<", 1)[0].split("::")[-2:])
    functor = FUNCTOR.search(clean[len(base):])
    return f"{base}[{functor.group(1)}]" if functor and "elementwise" in base else base


def _kernels(fn, iters=10) -> list[dict]:
    fn()
    sync()
    with profile(activities=profiler_activities()) as prof:
        for _ in range(iters):
            fn()
        sync()
    out = []
    for evt in prof.key_averages():
        if getattr(evt, "device_type", DeviceType.CPU) == DeviceType.CUDA:
            out.append({"name": short_kernel(evt.key), "calls": evt.count / iters,
                        "us": evt.self_device_time_total / iters,
                        "category": categorize_kernel(evt.key)})
    return sorted(out, key=lambda k: -k["us"])


def _wall_us(fn, iters=200) -> float:
    for _ in range(10):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) * 1e6 / iters


def attention_experiment(lab: Lab) -> dict:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cfg = lab.model.config
    H, Hkv = cfg.num_q_heads, cfg.num_kv_heads
    D = cfg.hidden_dim // H
    groups = H // Hkv

    def decode_variants(T):
        q = torch.randn(1, H, 1, D, device="cuda")
        k = torch.randn(1, Hkv, T, D, device="cuda")
        v = torch.randn(1, Hkv, T, D, device="cuda")

        def baseline():   # what attention.py does in decode today
            qp = torch.arange(T - 1, T, device="cuda").unsqueeze(1)
            kp = torch.arange(T, device="cuda").unsqueeze(0)
            mask = kp <= qp
            kk = k.repeat_interleave(groups, dim=1)
            vv = v.repeat_interleave(groups, dim=1)
            return F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)

        def no_mask():
            kk = k.repeat_interleave(groups, dim=1)
            vv = v.repeat_interleave(groups, dim=1)
            return F.scaled_dot_product_attention(q, kk, vv)

        def gqa():
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        def math_backend():
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        def explicit():   # unfused reference: QK^T, softmax, xV as separate kernels
            kk = k.repeat_interleave(groups, dim=1)
            vv = v.repeat_interleave(groups, dim=1)
            w = (q @ kk.transpose(-2, -1)) * D ** -0.5
            return torch.softmax(w, dim=-1) @ vv

        ref = explicit()
        return {"baseline_mask_repeat": baseline, "no_mask_repeat": no_mask,
                "enable_gqa": gqa, "math_backend": math_backend,
                "explicit_unfused": explicit}, ref

    decode = []
    for T in ATTENTION_KV_LENGTHS:
        fns, ref = decode_variants(T)
        row = {"T_kv": T, "variants": {}}
        for name, fn in fns.items():
            ks = _kernels(fn)
            row["variants"][name] = {
                "wall_us": _wall_us(fn),
                "gpu_us": sum(k["us"] for k in ks),
                "kernel_count": sum(k["calls"] for k in ks),
                "kernels": [k["name"] for k in ks],
                "max_abs_error_vs_explicit": (fn() - ref).abs().max().item(),
            }
        decode.append(row)

    # Prefill (T=128, causal) and which backends accept fp32
    T = lab.args.prompt_len
    q = torch.randn(1, H, T, D, device="cuda")
    k = torch.randn(1, H, T, D, device="cuda")
    import warnings

    backends = {}
    warnings.filterwarnings("ignore", message=".*(Flash|Memory efficient|Mem efficient|CuDNN).*")
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION),
                          ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("math", SDPBackend.MATH)):
        try:
            with sdpa_kernel(backend):
                fn = lambda: F.scaled_dot_product_attention(q, k, k, is_causal=True)  # noqa: E731
                ks = _kernels(fn)
            backends[name] = {"ok": True, "gpu_us": sum(x["us"] for x in ks),
                              "kernel_count": sum(x["calls"] for x in ks),
                              "kernels": [x["name"] for x in ks]}
        except RuntimeError as exc:
            backends[name] = {"ok": False, "error": str(exc).splitlines()[0][:200]}
    default = _kernels(lambda: F.scaled_dot_product_attention(q, k, k, is_causal=True))

    # In-model: the attention regions of one decode step, baseline flags
    return {
        "decode": decode,
        "prefill_backends_fp32": backends,
        "prefill_default_kernels": [x["name"] for x in default],
    }


# ======================================================================
# 15.14 — launch overhead
# ======================================================================


def launch_overhead() -> list[dict]:
    total = 1 << 20
    rows = []
    for n in LAUNCH_COUNTS:
        chunks = [torch.zeros(total // n, device="cuda") for _ in range(n)]
        tiny = [torch.zeros(1, device="cuda") for _ in range(n)]

        def same_work():
            for t in chunks:
                t.add_(1.0)

        def tiny_ops():
            for t in tiny:
                t.add_(1.0)

        k_same = _kernels(same_work, iters=5)
        rows.append({
            "ops": n,
            "elements_per_op": total // n,
            "same_work_wall_us": _wall_us(same_work, iters=max(2000 // n, 5)),
            "same_work_gpu_us": sum(k["us"] for k in k_same),
            "tiny_ops_wall_us": _wall_us(tiny_ops, iters=max(2000 // n, 5)),
            "tiny_ops_gpu_us": sum(k["us"] for k in _kernels(tiny_ops, iters=5)),
        })
    return rows


# ======================================================================
# Report
# ======================================================================


def pct(x):
    return f"{100 * x:.1f}%"


def write_findings(path: Path, r: dict):
    meta, ladder = r["meta"], r["ladder"]
    base = ladder[0]["metrics"]
    kept = [s for s in ladder[1:] if s["kept"]]
    final = r["final"]
    lines = []
    add = lines.append

    add("# Phase 15 — kernel-level optimization findings\n")
    add(f"Generated by `profiles/phase15/phase15_benchmark.py` on **{meta['device']}**, "
        f"checkpoint `{meta['checkpoint']}`, commit `{meta['git_commit']}`"
        f"{' (dirty tree)' if meta['git_dirty'] else ''}, PyTorch {meta['torch']}. "
        "All numbers come from `results.json`; rerun the script to refresh both files.\n")

    # ------------------------------------------------------------------
    add("## 15.1 Baseline (frozen)\n")
    b = base
    add("| | |")
    add("|---|---|")
    for k, v in [
        ("Model", f"V1, {meta['params_m']:.1f}M params, {meta['config']['num_layers']} layers, "
                  f"hidden {meta['config']['hidden_dim']}, vocab {meta['config']['vocab_size']}"),
        ("Checkpoint", meta["checkpoint"]),
        ("Device", meta["device"]),
        ("Dtype", meta["dtype"] + f" (matmul precision `{meta['fp32_matmul_precision']}`)"),
        ("Prompt length", meta["prompt_len"]),
        ("Generation length", f"{meta['decode_steps']} decode steps timed; {meta['gen_len']} "
                              "tokens end-to-end"),
        ("Prefill latency", f"{b['prefill']['wall_ms']:.2f} ms"),
        ("Prefill GPU time", f"{b['prefill']['gpu_busy_ms_per_step']:.2f} ms"),
        ("Prefill GPU utilization", pct(b["prefill"]["gpu_utilization"])),
        ("Prefill tokens/s", f"{b['prefill']['tokens_per_s']:,.0f}"),
        ("Decode latency / token", f"{b['decode']['wall_ms_per_token']:.3f} ms"),
        ("Decode GPU time / token", f"{b['decode']['gpu_busy_ms_per_step']:.3f} ms"),
        ("Decode GPU utilization", pct(b["decode"]["gpu_utilization"])),
        ("Decode tokens/s", f"{b['decode']['tokens_per_s']:.0f}"),
        ("Peak GPU memory (generation)", f"{b['generation']['peak_mb']:.1f} MB"),
        ("Kernel launches / step", f"prefill {b['prefill']['launches_per_step']:.0f}, "
                                   f"decode {b['decode']['launches_per_step']:.0f}"),
    ]:
        add(f"| {k} | {v} |")
    add("")
    add("The baseline is the untouched Phase 13 code path: every Phase 15 change is opt-in "
        "(`ModelRunner(kv_cache=\"dynamic\")`, all `src/model/fast_paths.py` flags off, eager). "
        f"Noise check: the baseline was measured again at the end — decode "
        f"{r['baseline_rerun']['decode']['wall_ms_per_token']:.3f} ms/token, prefill "
        f"{r['baseline_rerun']['prefill']['wall_ms']:.2f} ms "
        f"(first run {b['decode']['wall_ms_per_token']:.3f} / {b['prefill']['wall_ms']:.2f}). "
        f"Keep threshold: ±{100 * meta['threshold']:.0f}%.\n")

    # ------------------------------------------------------------------
    add("## 15.16 The optimization ladder\n")
    add("Each row adds **one** change on top of the rows marked kept above it, is gated "
        "against the baseline, then benchmarked. The keep/reject decision uses **paired** "
        f"timing: previous and candidate config alternate for {meta['paired_rounds']} rounds "
        "in the same time window, and the Δ columns are the median per-round change. The "
        "absolute ms columns come from each rung's own run and drift with the laptop's "
        "clocks, so compare rows by the Δ columns (see the baseline re-run note above).\n")
    add("| version | change | Δ decode vs prev | Δ prefill vs prev | decode ms/token (paired) | "
        "decode tok/s (paired) | prefill ms (paired) | launches/step decode | GPU busy ms/token | peak MB | "
        "gate max err | decision |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in ladder:
        m = s["metrics"]
        gate = s.get("gate")
        pair = s.get("paired")
        err = f"{gate['max_abs_error']:.1e}" if gate else "—"
        if pair:
            dd = f"{100 * (1 - pair['decode']['ratio']):+.1f}%"
            dp = f"{100 * (1 - pair['prefill']['ratio']):+.1f}%"
            dms = f"{pair['decode']['a_ms']:.3f} → {pair['decode']['b_ms']:.3f}"
            tps = f"{1000 / pair['decode']['b_ms']:.0f}"
            pms = f"{pair['prefill']['a_ms']:.2f} → {pair['prefill']['b_ms']:.2f}"
        else:
            dd = dp = "—"
            dms = f"{m['decode']['wall_ms_per_token']:.3f}"
            tps = f"{m['decode']['tokens_per_s']:.0f}"
            pms = f"{m['prefill']['wall_ms']:.2f}"
        bold = "**" if s.get("kept") else ""
        add(f"| {bold}{s['name']}{bold} | {s['description']} | {dd} | {dp} | {dms} | {tps} | {pms} | "
            f"{m['decode']['launches_per_step']:.0f} | {m['decode']['gpu_busy_ms_per_step']:.3f} | "
            f"{m['generation']['peak_mb']:.1f} | {err} | {s['decision']} |")
    add("")
    fm = final["metrics"]
    fp = final["paired_vs_baseline"]
    add(f"**Final configuration** ({final['config_description']}), paired against the "
        f"baseline: prefill {fp['prefill']['a_ms']:.2f} → {fp['prefill']['b_ms']:.2f} ms "
        f"({fp['prefill']['speedup']:.2f}x), decode "
        f"{fp['decode']['a_ms']:.3f} → {fp['decode']['b_ms']:.3f} ms/token "
        f"({fp['decode']['speedup']:.2f}x, {1000 / fp['decode']['a_ms']:.0f} → "
        f"{1000 / fp['decode']['b_ms']:.0f} tokens/s), decode "
        f"launches {b['decode']['launches_per_step']:.0f} → {fm['decode']['launches_per_step']:.0f} "
        f"per step, GPU utilization {pct(b['decode']['gpu_utilization'])} → "
        f"{pct(fm['decode']['gpu_utilization'])}. Correctness gate vs baseline: "
        f"{'PASSED' if final['gate']['passed'] else 'FAILED'} "
        f"(max abs err {final['gate']['max_abs_error']:.2e}, mean "
        f"{final['gate']['mean_abs_error']:.2e}).\n")
    best_eager = r.get("best_eager")
    if best_eager and best_eager["config_description"] != final["config_description"]:
        bp = best_eager["paired_vs_baseline"]
        add(f"Best eager (no compile) configuration ({best_eager['config_description']}), "
            f"paired against the baseline: prefill {bp['prefill']['speedup']:.2f}x, decode "
            f"{bp['decode']['speedup']:.2f}x ({bp['decode']['a_ms']:.3f} → "
            f"{bp['decode']['b_ms']:.3f} ms/token).\n")

    # ------------------------------------------------------------------
    add("## 15.15 Correctness gate\n")
    add("Every rung was compared with the baseline on 3 prompts (9, 37 and 128 tokens), "
        f"{meta['gate_new_tokens']} greedy tokens each. Tolerance: max abs "
        f"{meta['tolerance']['max_abs']:.0e}, mean abs {meta['tolerance']['mean_abs']:.0e}; "
        "tokens must match exactly.\n")
    add("| version | prefill logits | decode logits | greedy | EOS | KV vs recompute | "
        "batched | batched = alone |")
    add("|---|---|---|---|---|---|---|---|")
    for s in ladder[1:]:
        g = s["gate"]
        by = defaultdict(list)
        for c in g["checks"]:
            by[c["name"]].append(c)

        def cell(name):
            cs = by.get(name, [])
            if not cs:
                return "—"
            ok = all(c["passed"] for c in cs)
            errs = [c["max_abs_error"] for c in cs if c["max_abs_error"] is not None]
            return ("✓" if ok else "✗") + (f" {max(errs):.1e}" if errs else "")
        add(f"| {s['name']} | {cell('prefill_logits')} | {cell('decode_logits')} | "
            f"{cell('greedy_tokens')} | {cell('eos')} | {cell('kv_cache')} | "
            f"{cell('batched')} | {cell('batched_independent')} |")
    add("")

    # ------------------------------------------------------------------
    add("## KV cache: did eliminating `torch.cat()` reduce copy/allocation overhead?\n")
    kv = next(s for s in ladder if s["name"] == "kv_preallocated")
    km, bm = kv["metrics"]["decode"], base["decode"]
    cat_b = bm["regions"].get("kv_cache/cat", {})
    wr = km["regions"].get("kv_cache/write", {})
    add("| decode, per step | baseline (`torch.cat`) | preallocated (`copy_` in place) |")
    add("|---|---|---|")
    add(f"| KV region calls | {cat_b.get('calls_per_step', 0):.0f} `kv_cache/cat` | "
        f"{wr.get('calls_per_step', 0):.0f} `kv_cache/write` |")
    add(f"| KV region GPU ms | {cat_b.get('gpu_ms_per_step', 0):.3f} | {wr.get('gpu_ms_per_step', 0):.3f} |")
    add(f"| KV region CPU ms (profiled) | {cat_b.get('cpu_ms_per_step', 0):.3f} | "
        f"{wr.get('cpu_ms_per_step', 0):.3f} |")
    add(f"| copy/cat kernels GPU ms | {bm['categories_ms_per_step'].get('copy/cat', 0):.3f} | "
        f"{km['categories_ms_per_step'].get('copy/cat', 0):.3f} |")
    add(f"| launches | {bm['launches_per_step']:.0f} | {km['launches_per_step']:.0f} |")
    kp = kv["paired"]["decode"]
    add(f"| decode ms/token (paired) | {kp['a_ms']:.3f} | {kp['b_ms']:.3f} |")
    add(f"| peak MB (generation) | {base['generation']['peak_mb']:.1f} | "
        f"{kv['metrics']['generation']['peak_mb']:.1f} |")
    add("")
    sweep = r.get("kv_context_sweep", [])
    if sweep:
        add("Paired dynamic vs preallocated cache (baseline flags) at longer contexts, "
            f"{meta['decode_steps']} decode steps:\n")
        add("| context | torch.cat ms/token | preallocated ms/token | Δ decode | bytes copied per step by cat |")
        add("|---|---|---|---|---|")
        row_bytes = 2 * meta["config"]["num_kv_heads"] * (meta["config"]["hidden_dim"]
                                                          // meta["config"]["num_q_heads"]) * 4
        for row in sweep:
            n = row["context"]
            copied = 2 * meta["config"]["num_layers"] * (n + 1) * row_bytes
            add(f"| {n} | {row['decode']['a_ms']:.3f} | {row['decode']['b_ms']:.3f} | "
                f"{100 * (1 - row['decode']['ratio']):+.1f}% | {copied / 2 ** 20:.2f} MB |")
        add("")
    add(f"Decision: **{kv['decision']}**. The in-place write still launches one copy kernel "
        "per K and per V per layer, the same count as `cat`, but each kernel now copies "
        "1 token instead of T+1, and nothing is allocated. At a 128-token context each "
        "`cat` moves only ~130 KB, so the saving per step is small in absolute terms. It "
        "grows linearly with context: Phase 14 counted 1.88 GB of copies for a full "
        "512-token generation. The preallocated buffer costs 4 MB per sequence up front "
        "(capacity 512), versus `cat`'s transient old+new copies.\n")

    # ------------------------------------------------------------------
    add("## Kernel launches: which operations generate many small kernels?\n")
    inv_b, inv_f = r["inventory"]["baseline"], r["inventory"]["final_eager"]
    add(f"Launches in one decode step, attributed to the profiler region they were issued "
        f"from (baseline: {inv_b['launches']}, final eager config: {inv_f['launches']}):\n")
    add("| region | baseline | final eager |")
    add("|---|---|---|")
    for name in sorted(set(inv_b["by_region"]) | set(inv_f["by_region"]),
                       key=lambda n: -inv_b["by_region"].get(n, 0)):
        add(f"| {name} | {inv_b['by_region'].get(name, 0)} | {inv_f['by_region'].get(name, 0)} |")
    add("")
    add("Top (region, aten op) launch sources in the baseline:\n")
    add("| region | op | launches/step |")
    add("|---|---|---|")
    for row in inv_b["by_op"][:15]:
        add(f"| {row['region']} | `{row['op']}` | {row['launches']} |")
    add("")
    L = meta["config"]["num_layers"]
    rb, rf = inv_b["by_region"], inv_f["by_region"]
    add(f"- **RoPE** ({rb.get('attention/rope', 0)} launches, "
        f"{rb.get('attention/rope', 0) / (2 * L):.0f} per q-or-k call): slice, 2x "
        "`repeat_interleave` (`copy_`), `mul`, `neg` + `stack` (`cat`) for `rotate_half`, `mul`, "
        f"`add`. `rope_cache` removes the `repeat_interleave`s (→ {rf.get('attention/rope', 0)}); "
        "the rest needs a fused kernel (torch.compile does it).")
    add(f"- **RMSNorm** ({rb.get('rmsnorm', 0)} = 6 x {2 * L + 1} norms): pow, mean, add, rsqrt, "
        f"mul, mul. `F.rms_norm` makes it one kernel per norm (→ {rf.get('rmsnorm', 0)}).")
    add(f"- **Mask + masked SDPA** ({rb.get('attention/mask', 0)} + {rb.get('attention/sdpa', 0)}): "
        "the decode mask costs 3 kernels per layer (2x `arange`, compare), and a masked "
        "memory-efficient SDPA call adds `fill_` kernels on top of the attention kernel. "
        f"Unmasked decode: {rf.get('attention/mask', 0)} + {rf.get('attention/sdpa', 0)}.")
    final_eager = r["best_eager"]["config"]
    qkv = next(s for s in ladder if s["name"] == "fused_qkv")
    add(f"- **Matmuls**: each Linear is its own launch, {rb.get('attention/qkv_proj', 0)} for "
        f"q/k/v alone. `fused_qkv` makes that 1 per layer ({L} per step)"
        + ("." if "fused_qkv" in final_eager["flags"] else
           f", but it was not kept in this run ({qkv['decision']}): the matmuls at M=1 are GEMVs "
           "streaming the same weight bytes either way, so fusing saves only 2 launches per layer."))
    if final_eager["kv_cache"] == "static":
        add(f"- **KV cache**: `torch.cat` ({rb.get('kv_cache/cat', 0)}) becomes an in-place `copy_` "
            f"({rf.get('kv_cache/write', 0)}): same launch count, far fewer bytes, no allocation.\n")
    else:
        add(f"- **KV cache**: `torch.cat` ({rb.get('kv_cache/cat', 0)} launches) would become an "
            "in-place `copy_` with the same launch count. The preallocated cache was not kept "
            "at this context length (see the KV section), so the final config still uses `cat`.\n")

    # ------------------------------------------------------------------
    add("## Compilation: does `torch.compile` fuse or improve them?\n")
    comp = next((s for s in ladder if s["name"] == "torch_compile"), None)
    if comp is None:
        add("Skipped (`--skip-compile`).\n")
    else:
        cm = comp["metrics"]
        prev = comp["compared_to"]
        add("| | eager (kept set) | torch.compile |")
        add("|---|---|---|")
        add(f"| first prefill call (compile) | {prev['cold']['first_prefill_ms']:.0f} ms | "
            f"{cm['cold']['first_prefill_ms']:.0f} ms |")
        add(f"| first 4 decode steps | {', '.join(f'{x:.0f}' for x in prev['cold']['first_decode_steps_ms'])} ms | "
            f"{', '.join(f'{x:.0f}' for x in cm['cold']['first_decode_steps_ms'])} ms |")
        cp = comp["paired"]
        add(f"| steady prefill (paired) | {cp['prefill']['a_ms']:.2f} ms | {cp['prefill']['b_ms']:.2f} ms |")
        add(f"| steady decode / token (paired) | {cp['decode']['a_ms']:.3f} ms | "
            f"{cp['decode']['b_ms']:.3f} ms |")
        add(f"| decode launches / step | {prev['decode']['launches_per_step']:.0f} | "
            f"{cm['decode']['launches_per_step']:.0f} |")
        add(f"| prefill launches | {prev['prefill']['launches_per_step']:.0f} | "
            f"{cm['prefill']['launches_per_step']:.0f} |")
        add(f"| decode GPU utilization | {pct(prev['decode']['gpu_utilization'])} | "
            f"{pct(cm['decode']['gpu_utilization'])} |")
        add(f"| peak MB (generation) | {prev['generation']['peak_mb']:.1f} | "
            f"{cm['generation']['peak_mb']:.1f} |")
        add(f"| graphs compiled / graph breaks | — | {cm['compile']['unique_graphs']} / "
            f"{cm['compile']['graph_breaks']} |")
        add("")
        top = [k for k in cm["decode"]["top_kernels"] if "triton" in k["name"]][:4]
        if top:
            add("Inductor replaces chains of elementwise ops with generated Triton kernels, e.g.:\n")
            for k in top:
                add(f"- `{k['name'][:90]}` ({k['calls_per_step']:.0f}/step, "
                    f"{1e3 * k['gpu_ms_per_step']:.1f} us/step)")
            add("")
        add(f"Decision: **{comp['decision']}**. Compilation costs seconds up front "
            f"({cm['cold']['first_prefill_ms'] / 1e3:.1f} s for the first prefill, "
            f"{cm['cold']['first_decode_steps_ms'][0] / 1e3:.1f} s for the first decode step), so "
            "the first iteration says nothing about steady state. With `dynamic=True` and the "
            f"{comp['config']['kv_cache']} KV cache, {cm['compile']['unique_graphs']} graphs cover every prompt length "
            "and every decode position, and later steps reuse them. Matmuls still go to cuBLAS; what shrinks "
            "is the number of elementwise launches and the Python/dispatcher cost per op. "
            "The remaining step is CUDA graphs (`mode=\"reduce-overhead\"`), which needs "
            "static shapes: full-capacity attention with a length mask. That is left for "
            "the next phase.\n")

    # ------------------------------------------------------------------
    add("## Decode GEMV: how does batching change the matrix workload?\n")
    for name, rows in r["batching"].items():
        add(f"**{name}** — B requests decoding together, {meta['prompt_len']}-token contexts:\n")
        add("| B | step ms (= latency/token per request) | aggregate tok/s | GPU util | "
            "launches/step | GEMV ms | GEMM ms | matmul kernels |")
        add("|---|---|---|---|---|---|---|---|")
        for row in rows:
            add(f"| {row['B']} | {row['step_ms']:.3f} | {row['aggregate_tokens_per_s']:.0f} | "
                f"{pct(min(row['gpu_utilization'], 1))} | {row['launches_per_step']:.0f} | "
                f"{row['gemv_ms_per_step']:.3f} | {row['gemm_ms_per_step']:.3f} | "
                f"{', '.join('`' + k + '`' for k in row['matmul_kernels'][:3])} |")
        add("")
    rows = next(iter(r["batching"].values()))
    add(f"Going from B=1 to B={rows[-1]['B']} multiplies aggregate throughput by "
        f"{rows[-1]['aggregate_tokens_per_s'] / rows[0]['aggregate_tokens_per_s']:.1f}x, while "
        f"each request's per-token latency grows only "
        f"{rows[-1]['step_ms'] / rows[0]['step_ms']:.2f}x. The launch count per step is the "
        "same for every B: batching amortises each launch (and each pass over the weights) "
        "over B rows. At B=1 the matmuls are GEMVs, and as B grows cuBLAS moves to "
        "`gemmSN` and then to tiled SGEMMs (Phase 14's M sweep, now inside the full "
        "model). This is the payoff of continuous batching (Phase 11).\n")

    # ------------------------------------------------------------------
    add("## Attention: where does the time go and what does PyTorch use?\n")
    att = r["attention"]
    add("Decode attention for one layer (q `[1,8,1,64]`, K/V cache `[1,2,T,64]` fp32), "
        "GPU us per call (kernels):\n")
    names = list(att["decode"][0]["variants"])
    add("| T_kv | " + " | ".join(names) + " |")
    add("|---|" + "---|" * len(names))
    for row in att["decode"]:
        add(f"| {row['T_kv']} | " + " | ".join(
            f"{v['gpu_us']:.1f} ({v['kernel_count']:.0f})" for v in row["variants"].values()) + " |")
    add("")
    first = att["decode"][0]["variants"]
    add("Kernels per variant (T=16):\n")
    for name, v in first.items():
        add(f"- **{name}**: " + ", ".join(f"`{k}`" for k in v["kernels"]) +
            f" — max abs err vs unfused {v['max_abs_error_vs_explicit']:.1e}")
    add("")
    add("Prefill (T=128, causal), fp32 backend availability:\n")
    for name, v in att["prefill_backends_fp32"].items():
        add(f"- **{name}**: " + (f"{v['gpu_us']:.1f} us, " + ", ".join(f"`{k}`" for k in v["kernels"])
                                 if v["ok"] else f"not available — {v['error']}"))
    add(f"- default choice: {', '.join('`' + k + '`' for k in att['prefill_default_kernels'])}")
    add("")
    add("- PyTorch already runs a **fused** SDPA kernel: one `fmha_cutlassF` "
        "(memory-efficient attention) launch computes QKᵀ, softmax and ×V together. The "
        "unfused reference needs several kernels. FlashAttention does not support fp32, "
        "so the memory-efficient kernel is the best fused option for this model's dtype.")
    add("- The decode mask is unnecessary: a single query at the last position sees every "
        "key. Building it costs 3 extra kernels per layer (2x `arange`, compare), and a "
        "masked call has to read the mask too.")
    v16 = att["decode"][0]["variants"]
    gqa_is_math = v16["enable_gqa"]["kernels"] == v16["math_backend"]["kernels"]
    add("- `repeat_interleave` for GQA materialises 4x copies of K and V every layer and every "
        "step. `enable_gqa=True` avoids them"
        + (", but in fp32 the memory-efficient kernel does not accept it, so SDPA **falls back "
           "to the MATH backend**: identical kernels to the forced-math column, "
           f"{v16['enable_gqa']['kernel_count']:.0f} launches instead of "
           f"{v16['no_mask_repeat']['kernel_count']:.0f}." if gqa_is_math else "."))
    gq = next((s for s in r["ladder"] if s["name"] == "sdpa_gqa"), None)
    if gq:
        prev_idx = r["ladder"].index(gq) - 1
        pv = r["ladder"][prev_idx]["metrics"]["decode"]
        gm = gq["metrics"]["decode"]
        add(f"- That is why the `sdpa_gqa` rung was rejected even though it **lowered** GPU busy "
            f"time per token ({pv['gpu_busy_ms_per_step']:.3f} → {gm['gpu_busy_ms_per_step']:.3f} ms): "
            f"it raised launches per step ({pv['launches_per_step']:.0f} → "
            f"{gm['launches_per_step']:.0f}) and decode is launch-bound, so wall time went "
            f"{100 * (gq['paired']['decode']['ratio'] - 1):+.0f}%. Less GPU work is not a win "
            "if it costs more launches.")
    ex, fu = att["decode"][0]["variants"]["explicit_unfused"], att["decode"][0]["variants"]["no_mask_repeat"]
    if ex["gpu_us"] < fu["gpu_us"]:
        add(f"- With a single query, the fused fp32 `fmha_cutlassF` kernel is not the fastest on the "
            f"GPU: at T=16 the unfused QKᵀ/softmax/×V path takes {ex['gpu_us']:.1f} us of GPU time "
            f"vs {fu['gpu_us']:.1f} us. The fused kernel is tiled for many queries (64-row blocks), "
            "and decode fills one row. It still wins on launches (fewer kernels), which is what "
            "matters at this model size.")
    add("- As T grows, decode attention time grows with the K/V bytes read, but it stays "
        "small next to the matmuls at this model's context of 512 or less.\n")

    # ------------------------------------------------------------------
    add("## Memory: which operations are limited by data movement rather than arithmetic?\n")
    add("Decode GPU ms per step by kernel category along the ladder:\n")
    cats = ["gemv", "gemm", "attention", "elementwise", "copy/cat", "reduction",
            "index/gather", "other"]
    add("| version | " + " | ".join(cats) + " |")
    add("|---|" + "---|" * len(cats))
    for s in ladder:
        c = s["metrics"]["decode"]["categories_ms_per_step"]
        add(f"| {s['name']} | " + " | ".join(f"{c.get(x, 0):.3f}" for x in cats) + " |")
    add("")
    add("Categories come from kernel names. Inductor's Triton kernels are named after the ops "
        "they fuse (e.g. `triton_poi_fused__scaled_dot_product_efficient_attention_..._mul`), so "
        "under torch_compile the fused elementwise work shows up as attention/other, not as "
        "elementwise.\n")
    add("None of the decode kernels are limited by arithmetic. Arithmetic intensity is "
        "about 0.5 FLOP/byte for the GEMVs (Phase 14) and lower still for elementwise ops "
        "(1 FLOP per 8 bytes read and written), so they are bound by DRAM bandwidth, or at "
        "this size by launch latency:")
    add("- **copy/cat** (`torch.cat` KV growth, `repeat_interleave`, `contiguous`, "
        "`rotate_half`'s `stack`) is pure data movement with zero arithmetic.")
    add("- **elementwise** (RoPE, RMSNorm, SiLU·mul, residual adds) reads and writes the "
        "whole activation for 1-2 FLOPs per element.")
    add("- **gemv** streams all 116 MB of weights once per token. It is the one "
        "data-movement cost that no fusion removes, only batching (reuse across rows) or "
        "fewer bytes per weight (lower precision).\n")

    # ------------------------------------------------------------------
    add("## Kernel launch overhead (15.14)\n")
    add("Same total work (2^20 fp32 elements +1) split over N kernels, and N kernels on "
        "1-element tensors:\n")
    add("| N kernels | elements each | wall us (same work) | GPU us (same work) | "
        "wall us (tiny) | GPU us (tiny) | wall us per tiny launch |")
    add("|---|---|---|---|---|---|---|")
    for row in r["launch_overhead"]:
        add(f"| {row['ops']} | {row['elements_per_op']:,} | {row['same_work_wall_us']:.1f} | "
            f"{row['same_work_gpu_us']:.1f} | {row['tiny_ops_wall_us']:.1f} | "
            f"{row['tiny_ops_gpu_us']:.1f} | {row['tiny_ops_wall_us'] / row['ops']:.2f} |")
    add("")
    lo = r["launch_overhead"]
    add(f"The same work costs {lo[-1]['same_work_wall_us'] / lo[0]['same_work_wall_us']:.0f}x "
        f"more wall time when split into {lo[-1]['ops']} launches. GPU time grows too "
        f"({lo[0]['same_work_gpu_us']:.0f} → {lo[-1]['same_work_gpu_us']:.0f} us), because every "
        f"kernel has a floor of ~{lo[-1]['tiny_ops_gpu_us'] / lo[-1]['ops']:.1f} us, but the CPU "
        f"cost is worse: ~{lo[-1]['tiny_ops_wall_us'] / lo[-1]['ops']:.1f} us per launch no matter "
        "how little it computes. More GPU operations does not mean more useful computation. "
        f"A decode step issued {b['decode']['launches_per_step']:.0f} launches in the "
        "baseline.\n")

    # ------------------------------------------------------------------
    add("## Overall: which optimization actually produced a measurable improvement?\n")
    for s in ladder[1:]:
        add(f"- **{s['name']}** — {s['decision']}")
    add("")
    borderline = [s["name"] for s in ladder[1:] if s.get("paired") and
                  max(abs(1 - s["paired"]["decode"]["ratio"]), abs(1 - s["paired"]["prefill"]["ratio"]))
                  < 2 * meta["threshold"]]
    if borderline:
        add(f"Borderline (every |Δ| < {200 * meta['threshold']:.0f}%, twice the threshold): "
            + ", ".join(f"`{n}`" for n in borderline) + ". Their true effect at this context is "
            "about the size of the run-to-run noise, so their keep/reject can flip between runs. "
            "They do no harm (the gate passes), they just don't pay for themselves here.\n")
    add("The measured decision, not the theory, sets the final configuration. A change "
        "marked \"within noise\" may still be worth having (e.g. the KV cache's copy cost "
        "only grows with context), but it did not make this workload measurably faster. "
        "All engine defaults are unchanged. Opt in with `ModelRunner(kv_cache=\"static\")`, "
        "`fast_paths.prepare(model)` + `fast_paths.set_flags(...)` and/or `torch.compile`.")

    path.write_text("\n".join(lines) + "\n")


# ======================================================================
# Main
# ======================================================================


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=12, help="paired A/B rounds per decision")
    parser.add_argument("--threshold", type=float, default=0.03,
                        help="minimum relative speedup to keep a change (and max regression)")
    parser.add_argument("--gate-new-tokens", type=int, default=24)
    parser.add_argument("--quick", action="store_true", help="--repeats 5")
    parser.add_argument("--skip-compile", action="store_true")
    return parser.parse_args()


def git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src"],
                                             cwd=ROOT, text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("Phase 15 measures CUDA kernels: a GPU is required.")
    repeats = 5 if args.quick else args.repeats
    rounds = 6 if args.quick else args.rounds
    torch.manual_seed(0)

    lab = Lab(args)
    commit, dirty = git_info()
    cfg = lab.model.config
    tolerance = Tolerance()
    results = {
        "meta": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "checkpoint": str(Path(lab.checkpoint).relative_to(ROOT))
            if Path(lab.checkpoint).is_absolute() else lab.checkpoint,
            "git_commit": commit,
            "git_dirty": dirty,
            "dtype": str(next(lab.model.parameters()).dtype).replace("torch.", ""),
            "fp32_matmul_precision": torch.get_float32_matmul_precision(),
            "params_m": sum(p.numel() for p in lab.model.parameters()) / 1e6,
            "config": {k: getattr(cfg, k) for k in ("vocab_size", "max_seq_len", "hidden_dim",
                                                    "num_layers", "num_q_heads", "num_kv_heads",
                                                    "ffn_dim")},
            "prompt_len": args.prompt_len,
            "decode_steps": args.decode_steps,
            "gen_len": args.gen_len,
            "repeats": repeats,
            "paired_rounds": rounds,
            "threshold": args.threshold,
            "tolerance": {"max_abs": tolerance.max_abs, "mean_abs": tolerance.mean_abs},
            "gate_new_tokens": args.gate_new_tokens,
        },
    }

    print(f"Phase 15 on {results['meta']['device']} ({lab.checkpoint})")
    baseline_cfg = Config()
    reference = lab.variant("baseline", baseline_cfg)
    ref_cache: dict = {}
    prompts = lab.gate_prompts()

    print("\n[baseline]")
    base_metrics = measure(lab, baseline_cfg, HERE / "baseline", repeats)
    ladder = [{"name": "baseline", "description": "Phase 13 code path", "config": baseline_cfg.to_dict(),
               "metrics": base_metrics, "kept": True, "decision": "reference"}]
    print(f"  prefill {base_metrics['prefill']['wall_ms']:.2f} ms, decode "
          f"{base_metrics['decode']['wall_ms_per_token']:.3f} ms/token")

    current_cfg, current = baseline_cfg, base_metrics
    best_eager_cfg, best_eager = baseline_cfg, base_metrics

    for name, change, description in CANDIDATES:
        if change.get("compile") and args.skip_compile:
            continue
        cfg_new = current_cfg.with_change(change)
        print(f"\n[{name}] {cfg_new.describe()}")
        metrics = measure(lab, cfg_new, HERE / name, repeats)
        gate = run_gate(reference, lab.variant(name, cfg_new), prompts, lab.tokenizer,
                        max_new_tokens=args.gate_new_tokens, tolerance=tolerance,
                        reference_cache=ref_cache)
        pair = paired(lab, current_cfg, cfg_new, rounds)
        kept, decision = decide(pair, gate, args.threshold)
        print(f"  paired vs previous: decode {pair['decode']['a_ms']:.3f} -> "
              f"{pair['decode']['b_ms']:.3f} ms/token, prefill {pair['prefill']['a_ms']:.2f} -> "
              f"{pair['prefill']['b_ms']:.2f} ms, gate {'ok' if gate.passed else 'FAIL'} "
              f"(max err {gate.max_abs_error:.1e}) -> {decision}")
        ladder.append({
            "name": name, "description": description, "config": cfg_new.to_dict(),
            "metrics": metrics, "paired": pair, "gate": gate.to_dict(), "kept": kept,
            "decision": decision,
            "compared_to": {k: current[k] for k in ("prefill", "decode", "generation", "cold")},
        })
        if kept:
            current_cfg, current = cfg_new, metrics
            if not cfg_new.compile:
                best_eager_cfg, best_eager = cfg_new, metrics

    results["ladder"] = ladder
    final_gate = run_gate(reference, lab.variant("final", current_cfg), prompts, lab.tokenizer,
                          max_new_tokens=args.gate_new_tokens, tolerance=tolerance,
                          reference_cache=ref_cache)
    print("\n[paired baseline vs final / best eager]")
    results["final"] = {"config": current_cfg.to_dict(), "config_description": current_cfg.describe(),
                        "metrics": current, "gate": final_gate.to_dict(),
                        "paired_vs_baseline": paired(lab, baseline_cfg, current_cfg, rounds)}
    results["best_eager"] = {"config": best_eager_cfg.to_dict(),
                             "config_description": best_eager_cfg.describe(), "metrics": best_eager,
                             "paired_vs_baseline": paired(lab, baseline_cfg, best_eager_cfg, rounds)}

    print("[KV cache vs context length]")
    static_cfg = baseline_cfg.with_change({"kv_cache": "static"})
    results["kv_context_sweep"] = [
        {"context": n, **paired(lab, baseline_cfg, static_cfg, rounds, prompt_len=n)}
        for n in KV_CONTEXTS
    ]

    print("\n[kernel inventory]")
    results["inventory"] = {"baseline": kernel_inventory(lab, baseline_cfg),
                            "final_eager": kernel_inventory(lab, best_eager_cfg)}

    print("[batching]")
    results["batching"] = {"baseline": batching(lab, baseline_cfg, "baseline", repeats)}
    if best_eager_cfg != baseline_cfg:
        results["batching"]["best eager"] = batching(lab, best_eager_cfg, "best_eager", repeats)

    print("[attention]")
    results["attention"] = attention_experiment(lab)
    print("[launch overhead]")
    results["launch_overhead"] = launch_overhead()

    print("[baseline re-run]")
    rerun = measure(lab, baseline_cfg, HERE / "baseline_rerun", repeats)
    results["baseline_rerun"] = {k: rerun[k] for k in ("prefill", "decode")}

    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=str))
    write_findings(HERE / "FINDINGS.md", results)
    print(f"\nWrote {HERE / 'results.json'} and {HERE / 'FINDINGS.md'}")


if __name__ == "__main__":
    main()
