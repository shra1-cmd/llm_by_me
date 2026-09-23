"""
Phase 11 — continuous batching vs static batching vs sequential.

What this script does, on the real checkpoint:

    1. Builds a workload of requests with very different generation
       lengths (default 10 requests, 10..120 new tokens). Mixed
       lengths are exactly where static batching wastes slots: short
       requests finish and their slot sits empty until the longest
       member of the batch is done.
    2. Runs the same workload three ways, all submitted at t=0:
           sequential   Phase 8  one request at a time
           static       Phase 9  FIFO batches of --batch-size, run to
                                 completion before the next batch
           continuous   Phase 11 --batch-size slots, refilled every step
    3. Reports total completion time, aggregate tokens/sec, per-request
       latency (submit -> finished), peak GPU memory, number of forward
       passes and average decode-batch occupancy.
    4. Checks every request produced identical tokens in all three
       modes (greedy) — batching must not change outputs.
    5. Prints the first steps of the continuous batch-membership
       timeline so you can watch requests enter and leave.

Latency caveat: in static mode our engine hands results back when the
whole batch finishes, so every member's latency is its batch's end
time — that is the behavior being compared, not a measurement bug.

Usage:
    PYTHONPATH="$(pwd)" python scripts/run_continuous_batching.py
    PYTHONPATH="$(pwd)" python scripts/run_continuous_batching.py --batch-size 8 --num-requests 16
"""

import argparse
import statistics
import time

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.continuous_batching import ContinuousBatchingEngine
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache_manager import KVCacheManager
from src.inference.model_runner import ModelRunner
from src.inference.request import SamplingParams
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
]

GEN_LENGTHS = [10, 120, 20, 80, 15, 100, 30, 60, 25, 90]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def is_cuda(device):
    return str(device).startswith("cuda") and torch.cuda.is_available()


def sync(device):
    if is_cuda(device):
        torch.cuda.synchronize()


def workload(n):
    return [
        (f"R{i:02d}", PROMPTS[i % len(PROMPTS)], GEN_LENGTHS[i % len(GEN_LENGTHS)])
        for i in range(n)
    ]


def make_manager(model, device):
    return KVCacheManager.for_model(model, device=device)


# --------------------------------------------------
# The three modes. Each returns (tokens by id, metrics).
# --------------------------------------------------


def start_run(device):
    sync(device)
    if is_cuda(device):
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def finish_metrics(start, finish_times, generated, forward_passes, occupancy, device):
    sync(device)
    total = time.perf_counter() - start

    latencies = sorted(t - start for t in finish_times.values())

    return {
        "total": total,
        "tokens_per_second": generated / total,
        "lat_mean": statistics.mean(latencies),
        "lat_p50": statistics.median(latencies),
        "lat_max": latencies[-1],
        "peak_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if is_cuda(device) else None,
        "forward_passes": forward_passes,
        "occupancy": occupancy,
    }


def run_sequential(runner, tokenizer, sampler, model, jobs, device):
    engine = InferenceEngine(runner, tokenizer, sampler, device=device,
                             kv_cache_manager=make_manager(model, device))
    requests = [engine.submit(engine.create_request(p, max_new_tokens=n, request_id=rid))
                for rid, p, n in jobs]

    start = start_run(device)
    finish_times = {}
    forward_passes = 0

    while engine.scheduler.has_waiting():
        result = engine.step()
        engine.pop_result(result["request"].request_id)
        finish_times[result["request"].request_id] = time.perf_counter()
        forward_passes += result["request"].num_generated     # 1 prefill + (n-1) decodes

    generated = sum(r.num_generated for r in requests)
    metrics = finish_metrics(start, finish_times, generated, forward_passes, 1.0, device)

    return {r.request_id: r.all_tokens for r in requests}, metrics


def run_static(runner, tokenizer, sampler, model, jobs, batch_size, device):
    engine = InferenceEngine(runner, tokenizer, sampler, device=device,
                             kv_cache_manager=make_manager(model, device))
    requests = [engine.submit(engine.create_request(p, max_new_tokens=n, request_id=rid))
                for rid, p, n in jobs]

    start = start_run(device)
    finish_times = {}
    forward_passes = 0
    decode_rows = 0
    decode_steps = 0

    while engine.scheduler.has_waiting():
        results = engine.step_batch(batch_size)
        now = time.perf_counter()

        steps = results[0]["stats"]["decode_steps"]
        forward_passes += 1 + steps
        decode_steps += steps

        for result in results:
            engine.pop_result(result["request"].request_id)
            finish_times[result["request"].request_id] = now
            decode_rows += max(result["request"].num_generated - 1, 0)

    generated = sum(r.num_generated for r in requests)
    occupancy = decode_rows / decode_steps if decode_steps else 0.0
    metrics = finish_metrics(start, finish_times, generated, forward_passes, occupancy, device)

    return {r.request_id: r.all_tokens for r in requests}, metrics


def run_continuous(runner, tokenizer, sampler, model, jobs, batch_size, device):
    engine = ContinuousBatchingEngine(runner, tokenizer, sampler, device=device,
                                      kv_cache_manager=make_manager(model, device),
                                      max_batch_size=batch_size)
    requests = [engine.submit(engine.create_request(p, max_new_tokens=n, request_id=rid))
                for rid, p, n in jobs]

    start = start_run(device)
    finish_times = {}

    while engine.has_work():
        for result in engine.step():
            engine.pop_result(result["request"].request_id)
            finish_times[result["request"].request_id] = time.perf_counter()

    history = engine.history
    forward_passes = sum(bool(h.admitted) + bool(h.decoded) for h in history)
    decode_sizes = [len(h.decoded) for h in history if h.decoded]
    occupancy = statistics.mean(decode_sizes) if decode_sizes else 0.0

    generated = sum(r.num_generated for r in requests)
    metrics = finish_metrics(start, finish_times, generated, forward_passes, occupancy, device)

    return {r.request_id: r.all_tokens for r in requests}, metrics, engine


def print_timeline(engine, max_steps):
    print(f"\nContinuous batch membership (first {max_steps} steps with changes + last):")
    print(f"    {'step':>4}  {'admitted':<14} {'finished':<14} active after")

    shown = 0
    for h in engine.history:
        changed = h.admitted or h.finished
        is_last = h is engine.history[-1]

        if (changed and shown < max_steps) or is_last:
            print(f"    {h.step:>4}  {','.join(h.admitted) or '-':<14} "
                  f"{','.join(h.finished) or '-':<14} [{' '.join(h.active_after)}]")
            shown += 1


def main():
    args = parse_args()

    print("=" * 72)
    print("V1 CONTINUOUS BATCHING (Phase 11)")
    print("=" * 72)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)

    print(f"\nCheckpoint:\n    {checkpoint_path}")

    model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    sampler = SamplingParams(greedy=True).to_sampler()
    runner = ModelRunner(model=model, tokenizer=tokenizer, sampler=sampler, device=args.device)

    jobs = workload(args.num_requests)

    print(f"\nWorkload: {len(jobs)} requests, batch size {args.batch_size}, greedy")
    for rid, prompt, n in jobs:
        print(f"    {rid}  new_tokens={n:>3}  \"{prompt}\"")

    # Warm-up all three paths.
    warm = jobs[:2]
    warm = [(rid, p, 4) for rid, p, _ in warm]
    run_sequential(runner, tokenizer, sampler, model, warm, args.device)
    run_static(runner, tokenizer, sampler, model, warm, 2, args.device)
    run_continuous(runner, tokenizer, sampler, model, warm, 2, args.device)

    seq_tokens, seq = run_sequential(runner, tokenizer, sampler, model, jobs, args.device)
    sta_tokens, sta = run_static(runner, tokenizer, sampler, model, jobs, args.batch_size, args.device)
    con_tokens, con, con_engine = run_continuous(
        runner, tokenizer, sampler, model, jobs, args.batch_size, args.device
    )

    rows = [("sequential", seq), (f"static b={args.batch_size}", sta),
            (f"continuous b={args.batch_size}", con)]

    print(f"\nResults:")
    print(f"    {'mode':<16} {'total':>7} {'tok/s':>8} {'lat mean':>9} {'lat p50':>8} "
          f"{'lat max':>8} {'fwd':>5} {'occup':>6} {'peak MB':>8}")

    for name, m in rows:
        peak = f"{m['peak_mb']:8.1f}" if m["peak_mb"] is not None else "     n/a"
        print(f"    {name:<16} {m['total']:>6.2f}s {m['tokens_per_second']:>8.1f} "
              f"{m['lat_mean']:>8.2f}s {m['lat_p50']:>7.2f}s {m['lat_max']:>7.2f}s "
              f"{m['forward_passes']:>5} {m['occupancy']:>6.2f} {peak}")

    print(f"\n    fwd   = model forward passes (prefill + decode)")
    print(f"    occup = average number of requests per decode forward")

    print(f"\n    continuous vs static:  {sta['total'] / con['total']:.2f}x faster overall, "
          f"mean latency {sta['lat_mean'] / con['lat_mean']:.2f}x lower")

    print_timeline(con_engine, max_steps=12)

    static_match = sta_tokens == seq_tokens
    continuous_match = con_tokens == seq_tokens

    print(f"\nCorrectness (vs sequential, per request):")
    print(f"    static tokens match     = {static_match}")
    print(f"    continuous tokens match = {continuous_match}")

    if not continuous_match:
        for rid in seq_tokens:
            if seq_tokens[rid] != con_tokens[rid]:
                print(f"    MISMATCH {rid}")

    passed = static_match and continuous_match

    print("\n" + "=" * 72)

    if not passed:
        print("PHASE 11 FAILED")
        print("=" * 72)
        raise RuntimeError("Batched tokens differ from sequential tokens.")

    print("PHASE 11 PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
