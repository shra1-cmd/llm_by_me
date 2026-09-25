"""
Phase 12 — inference benchmark report.

Measures every inference architecture built in Phases 3-11 with the
shared harness in src/inference/benchmark.py (same checkpoint,
tokenizer, prompts, max_new_tokens, greedy sampling, device, dtype and
warm-up policy for all), then answers four questions:

    Q1  How much does KV caching improve single-request generation?
    Q2  What happens when multiple requests are batched?
    Q3  Does continuous batching improve aggregate throughput?
    Q4  What is the memory cost of more concurrent requests?

Scenarios:

    headline   8 requests x --max-new-tokens, the four spec columns
               (naive / KV cache / batch / continuous)
    single     1 request, 2x --max-new-tokens          -> Q1
    uniform    8 requests, all --max-new-tokens         -> Q2, Q3
               (all six modes, incl. paged variants)
    mixed      8 requests, 16..128 new tokens           -> Q3
    sweep      16 mixed requests, continuous batching at
               batch sizes 1, 2, 4, 8, 16               -> Q4

Every scenario runs --warmup discarded runs then --iterations measured
runs per mode. Tables show the mean; the "spread" tables show
mean / median / min / max. Every mode must emit identical tokens
(greedy) or the benchmark fails.

Usage:
    PYTHONPATH="$(pwd)" python scripts/benchmark_inference.py
    PYTHONPATH="$(pwd)" python scripts/benchmark_inference.py --iterations 5 --output bench.json
    PYTHONPATH="$(pwd)" python scripts/benchmark_inference.py --scenarios single uniform
"""

import argparse
import json

import torch

from src.inference.benchmark import (
    MODES,
    BenchmarkContext,
    BenchmarkRequest,
    benchmark_mode,
    outputs_match,
)
from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.tokenizer.tokenizer import BPETokenizer

PROMPTS = [
    "Hello, my name is",
    "Once upon a time",
    "The little dog",
    "One day, a girl named Sue went to the park and",
    "Tom had a red ball",
    "The sun was shining and the birds were singing. Lily wanted to",
    "Mom said",
    "There was a big tree in the garden",
    "The cat sat on the",
    "Ben and his friend",
    "It was a rainy day, so",
    "The old man looked at the",
    "Sara found a shiny",
    "In the forest there lived a",
    "The boy wanted to fly",
    "Every morning, the bird",
]

MIXED_LENGTHS = [16, 128, 32, 96, 24, 112, 48, 64, 20, 80, 40, 120, 28, 72, 56, 104]

SCENARIOS = ["headline", "single", "uniform", "mixed", "sweep"]
HEADLINE_MODES = ["naive", "kv_cache", "batch", "continuous"]
SWEEP_BATCH_SIZES = [1, 2, 4, 8, 16]

TABLE_ROWS = [
    ("TTFT (mean)", "ttft_mean_s", "s", 3),
    ("Latency (mean)", "latency_mean_s", "s", 3),
    ("Total time", "total_s", "s", 3),
    ("ITL (mean)", "itl_mean_ms", "ms", 2),
    ("ITL (p50)", "itl_p50_ms", "ms", 2),
    ("Tokens/sec", "tokens_per_s", "", 1),
    ("Peak GPU memory", "peak_mb", "MB", 1),
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--output", type=str, default=None, help="write all results as JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


# --------------------------------------------------
# Workloads
# --------------------------------------------------


def uniform_workload(n, max_new_tokens):
    return [BenchmarkRequest(f"R{i:02d}", PROMPTS[i % len(PROMPTS)], max_new_tokens) for i in range(n)]


def mixed_workload(n):
    return [
        BenchmarkRequest(f"R{i:02d}", PROMPTS[i % len(PROMPTS)], MIXED_LENGTHS[i % len(MIXED_LENGTHS)])
        for i in range(n)
    ]


# --------------------------------------------------
# Printing
# --------------------------------------------------


def fmt(value, unit, digits):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{(' ' + unit) if unit else ''}"


def print_table(reports, columns, title):
    width = 14

    print(f"\n{title}")
    print("-" * (20 + width * len(columns)))
    print(f"{'':<20}" + "".join(f"{c:>{width}}" for c in columns))
    print("-" * (20 + width * len(columns)))

    for label, key, unit, digits in TABLE_ROWS:
        cells = []
        for report in reports:
            stat = report.stats().get(key)
            cells.append(fmt(stat["mean"] if stat else None, unit, digits))
        print(f"{label:<20}" + "".join(f"{c:>{width}}" for c in cells))

    first = reports[0].stats()
    print("-" * (20 + width * len(columns)))
    print(f"{'prompt / gen / total tokens':<20}  "
          f"{first['prompt_tokens']} / {first['generated_tokens']} / {first['total_tokens']}"
          f"   (identical for every column)")

    pools = [r.stats()["kv_pool_mb"] for r in reports]
    if any(pools):
        print(f"{'KV pool (paged)':<20}" + "".join(
            f"{(fmt(p, 'MB', 1) if p else '-'):>{width}}" for p in pools))


def print_spread(reports, columns):
    print(f"\n  spread over measured runs (mean / median / min / max):")

    for key, label, unit, digits in [
        ("total_s", "total", "s", 3),
        ("tokens_per_s", "tok/s", "", 1),
        ("ttft_mean_s", "TTFT", "s", 3),
    ]:
        for column, report in zip(columns, reports):
            s = report.stats()[key]
            print(f"    {label:<6} {column:<16} "
                  f"{s['mean']:.{digits}f} / {s['median']:.{digits}f} / "
                  f"{s['min']:.{digits}f} / {s['max']:.{digits}f} {unit}")


def mean(report, key):
    return report.stats()[key]["mean"]


# --------------------------------------------------
# Scenarios
# --------------------------------------------------


def run_scenario(ctx, workload, modes, args):
    reports = [
        benchmark_mode(ctx, workload, mode, warmup=args.warmup, iterations=args.iterations)
        for mode in modes
    ]

    return reports, outputs_match(reports)


def main():
    args = parse_args()

    print("=" * 76)
    print("PHASE 12 — INFERENCE BENCHMARK")
    print("=" * 76)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    num_params = sum(p.numel() for p in model.parameters())
    dtype = next(model.parameters()).dtype
    weights_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2

    print(f"\nModel:\n    V1 / {checkpoint_path}  ({num_params / 1e6:.1f}M params, {dtype}, "
          f"{weights_mb:.1f} MB weights)")
    print(f"Device:\n    {args.device}"
          + (f"  ({torch.cuda.get_device_name()})" if args.device.startswith("cuda") else ""))
    print(f"Protocol:\n    greedy, batch size {args.batch_size}, "
          f"{args.warmup} warm-up + {args.iterations} measured run(s) per mode, "
          f"all requests arrive at t=0")

    ctx = BenchmarkContext(model=model, tokenizer=tokenizer, device=args.device,
                           batch_size=args.batch_size)

    results = {}
    all_match = True

    # --------------------------------------------------
    if "headline" in args.scenarios or "uniform" in args.scenarios:
        workload = uniform_workload(8, args.max_new_tokens)
        modes = list(MODES) if "uniform" in args.scenarios else HEADLINE_MODES
        reports, match = run_scenario(ctx, workload, modes, args)
        all_match &= match
        by_mode = dict(zip(modes, reports))
        results["uniform"] = by_mode

        if "headline" in args.scenarios:
            print_table([by_mode[m] for m in HEADLINE_MODES],
                        ["Naive", "KV Cache", "Batch", "Continuous"],
                        f"Headline: 8 requests x {args.max_new_tokens} new tokens")
            print("  note: Batch uses Phase 9 contiguous KV, Continuous uses Phase 11 paged KV.")
            print("        For a same-KV comparison use 'batch paged' vs 'continuous' below.")

        if "uniform" in args.scenarios:
            cols = ["naive", "kv", "kv paged", "batch", "batch paged", "continuous"]
            print_table(reports, cols, f"Uniform: all six modes (same 8 x {args.max_new_tokens} workload)")
            print_spread(reports, cols)

        print(f"\n  outputs identical across modes: {match}")

    # --------------------------------------------------
    if "single" in args.scenarios:
        workload = uniform_workload(1, 2 * args.max_new_tokens)
        modes = ["naive", "kv_cache", "kv_cache_paged"]
        reports, match = run_scenario(ctx, workload, modes, args)
        all_match &= match
        results["single"] = dict(zip(modes, reports))

        cols = ["naive", "kv", "kv paged"]
        print_table(reports, cols, f"Single request x {2 * args.max_new_tokens} new tokens")
        print_spread(reports, cols)
        print(f"\n  outputs identical across modes: {match}")

    # --------------------------------------------------
    if "mixed" in args.scenarios:
        workload = mixed_workload(8)
        modes = ["kv_cache", "batch", "batch_paged", "continuous"]
        reports, match = run_scenario(ctx, workload, modes, args)
        all_match &= match
        results["mixed"] = dict(zip(modes, reports))

        cols = ["kv", "batch", "batch paged", "continuous"]
        lengths = ", ".join(str(r.max_new_tokens) for r in workload)
        print_table(reports, cols, f"Mixed lengths: 8 requests, new tokens = [{lengths}]")
        print_spread(reports, cols)
        print(f"\n  outputs identical across modes: {match}")

    # --------------------------------------------------
    sweep = []

    if "sweep" in args.scenarios:
        workload = mixed_workload(16)
        reference = None

        print(f"\nConcurrency sweep: continuous batching, 16 mixed-length requests")
        print(f"    {'batch':>5} {'tok/s':>9} {'total':>8} {'TTFT':>8} {'latency':>8} "
              f"{'ITL':>8} {'peak MB':>9} {'KV pool MB':>11}")

        for batch_size in SWEEP_BATCH_SIZES:
            sweep_ctx = BenchmarkContext(model=model, tokenizer=tokenizer,
                                         device=args.device, batch_size=batch_size)
            report = benchmark_mode(sweep_ctx, workload, "continuous",
                                    warmup=args.warmup, iterations=args.iterations)

            if reference is None:
                reference = report.outputs
            match = report.deterministic and report.outputs == reference
            all_match &= match

            st = report.stats()
            sweep.append((batch_size, report))

            peak = st["peak_mb"]["mean"] if "peak_mb" in st else None
            print(f"    {batch_size:>5} {st['tokens_per_s']['mean']:>9.1f} "
                  f"{st['total_s']['mean']:>7.3f}s {st['ttft_mean_s']['mean']:>7.3f}s "
                  f"{st['latency_mean_s']['mean']:>7.3f}s {st['itl_mean_ms']['mean']:>6.2f}ms "
                  f"{fmt(peak, '', 1):>9} {st['kv_pool_mb']:>11.1f}"
                  f"{'' if match else '   OUTPUT MISMATCH'}")

        results["sweep"] = {f"batch_{b}": r for b, r in sweep}

    # --------------------------------------------------
    # Answers, computed from the numbers above
    # --------------------------------------------------

    print("\n" + "=" * 76)
    print("FINDINGS (derived from this run)")
    print("=" * 76)

    if "single" in results:
        s = results["single"]
        print(f"\nQ1  KV cache, single request ({2 * args.max_new_tokens} tokens):")
        print(f"    latency  naive {mean(s['naive'], 'latency_mean_s'):.3f}s -> "
              f"kv {mean(s['kv_cache'], 'latency_mean_s'):.3f}s  "
              f"({mean(s['naive'], 'latency_mean_s') / mean(s['kv_cache'], 'latency_mean_s'):.2f}x faster)")
        print(f"    ITL      naive {mean(s['naive'], 'itl_mean_ms'):.2f}ms -> "
              f"kv {mean(s['kv_cache'], 'itl_mean_ms'):.2f}ms")
        print(f"    paging   kv paged is "
              f"{mean(s['kv_cache_paged'], 'tokens_per_s') / mean(s['kv_cache'], 'tokens_per_s'):.2f}x "
              f"the throughput of contiguous kv")

    if "uniform" in results:
        u = results["uniform"]
        print(f"\nQ2  Batching, 8 uniform requests (batch size {args.batch_size}):")
        print(f"    tok/s    kv {mean(u['kv_cache'], 'tokens_per_s'):.1f} -> "
              f"batch {mean(u['batch'], 'tokens_per_s'):.1f}  "
              f"({mean(u['batch'], 'tokens_per_s') / mean(u['kv_cache'], 'tokens_per_s'):.2f}x)")
        print(f"    mean latency  kv {mean(u['kv_cache'], 'latency_mean_s'):.3f}s -> "
              f"batch {mean(u['batch'], 'latency_mean_s'):.3f}s")
        print(f"    ITL      kv {mean(u['kv_cache'], 'itl_mean_ms'):.2f}ms -> "
              f"batch {mean(u['batch'], 'itl_mean_ms'):.2f}ms  (each token waits for the whole batch)")
        if "batch_paged" in u:
            print(f"    paging   batch paged is "
                  f"{mean(u['batch_paged'], 'tokens_per_s') / mean(u['batch'], 'tokens_per_s'):.2f}x "
                  f"the throughput of contiguous batch")

    for name in ("uniform", "mixed"):
        if name in results and {"continuous", "batch_paged"} <= set(results[name]):
            r = results[name]
            label = "Q3" if name == "uniform" else "  "
            print(f"\n{label}  Continuous vs static batching, {name} lengths:")
            print(f"    tok/s    batch paged {mean(r['batch_paged'], 'tokens_per_s'):.1f} -> "
                  f"continuous {mean(r['continuous'], 'tokens_per_s'):.1f}  "
                  f"({mean(r['continuous'], 'tokens_per_s') / mean(r['batch_paged'], 'tokens_per_s'):.2f}x, "
                  f"same paged KV)")
            print(f"    mean latency  batch paged {mean(r['batch_paged'], 'latency_mean_s'):.3f}s -> "
                  f"continuous {mean(r['continuous'], 'latency_mean_s'):.3f}s")

    if sweep:
        base_peak = sweep[0][1].stats().get("peak_mb", {}).get("mean")
        top_b, top = sweep[-1]
        top_peak = top.stats().get("peak_mb", {}).get("mean")
        print(f"\nQ4  Memory cost of concurrency (continuous, 16 requests):")
        print(f"    KV pool  {sweep[0][1].stats()['kv_pool_mb']:.1f} MB at batch 1 -> "
              f"{top.stats()['kv_pool_mb']:.1f} MB at batch {top_b}")
        if base_peak is not None and top_peak is not None:
            print(f"    peak     {base_peak:.1f} MB -> {top_peak:.1f} MB  "
                  f"(+{(top_peak - base_peak) / (top_b - 1):.2f} MB per extra concurrent request; "
                  f"model weights alone are {weights_mb:.1f} MB)")
        print(f"    tok/s    {sweep[0][1].stats()['tokens_per_s']['mean']:.1f} -> "
              f"{top.stats()['tokens_per_s']['mean']:.1f}")

    print("\nCorrectness:")
    print(f"    All outputs match: {'PASS' if all_match else 'FAIL'}")

    if args.output:
        payload = {
            "checkpoint": str(checkpoint_path),
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "all_outputs_match": all_match,
            "scenarios": {
                scenario: {
                    mode: {
                        "stats": report.stats(),
                        "runs": [run.metrics() for run in report.runs],
                    }
                    for mode, report in reports.items()
                }
                for scenario, reports in results.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults written to {args.output}")

    print("\n" + "=" * 76)

    if not all_match:
        print("PHASE 12 FAILED")
        print("=" * 76)
        raise RuntimeError("Implementations disagree on generated tokens.")

    print("PHASE 12 PASSED")
    print("=" * 76)


if __name__ == "__main__":
    main()
