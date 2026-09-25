"""
Phase 13 — profile what PyTorch actually executes during inference.

Runs the Phase 13 experiments on the real checkpoint with
src/inference/profiler.py, prints a report, and writes:

    profiles/
    ├── prefill_128/        trace.json  ops.txt  shapes.txt
    ├── prefill_8/
    ├── decode_20/
    ├── batch_1/  batch_2/  batch_4/  batch_4_paged/
    ├── naive_20/  cached_20/
    ├── memory/snapshot.pickle      (https://pytorch.org/memory_viz)
    └── FINDINGS.md                 findings generated from this run

Open any trace.json in https://ui.perfetto.dev (or chrome://tracing)
to see the CPU timeline (our regions -> aten ops -> kernel launches)
above the GPU stream (the kernels themselves).

Experiments:

    A  prefill_128      one 128-token prompt, one forward
    B  decode_20        20 single-token decode steps on a 128-token
                        KV cache (the prefill is setup, not profiled)
    C  prefill_8        one 8-token prompt — compare with A
    D  batch_{1,2,4}    10 batched decode steps at batch 1, 2, 4
                        (contiguous KV), plus batch_4_paged (Phase 10
                        paged KV) to see what paging adds
    E  naive_20 vs cached_20
                        20 generated tokens from a 32-token prompt:
                        full recompute (Phase 3) vs prefill + decode
    M  memory           weights, KV cache per token, transient
                        working set of prefill/decode/batches, and an
                        allocation snapshot

Nothing here changes the model or kernels: Phase 13 is observation.

Usage:
    PYTHONPATH="$(pwd)" python scripts/profile_inference.py
    PYTHONPATH="$(pwd)" python scripts/profile_inference.py --out-dir profiles --experiments A B
"""

import argparse
from pathlib import Path

import torch

from src.inference.batch import build_prefill_batch
from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache_manager import KVCacheManager
from src.inference.model_runner import ModelRunner
from src.inference.profiler import (
    capture_memory_snapshot,
    measure_peak_memory,
    profile_workload,
)
from src.inference.request import SamplingParams
from src.model.profiling import region
from src.tokenizer.tokenizer import BPETokenizer

TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play "
    "outside in the sunshine with her dog, Max. One day, they went to the park "
    "and saw a big red ball under a tree. Lily ran to the ball and kicked it high "
    "into the sky. Max barked and chased it across the grass. Then a boy named Tom "
    "came to play with them. They laughed and played all day until the sun went "
    "down, and then they walked home together, happy and tired. "
)

# Disjoint regions: their GPU times add up to (almost) all kernel time.
LEAF_REGIONS = [
    "embedding",
    "rmsnorm",
    "attention/qkv_proj",
    "attention/rope",
    "attention/kv_cache",
    "attention/gqa_repeat",
    "attention/mask",
    "attention/sdpa",
    "attention/out_proj",
    "mlp/gate_up_proj",
    "mlp/act_mul",
    "mlp/down_proj",
    "lm_head",
    "sampling",
    "batch/build",
    "kv_cache/batch_gather",
    "kv_cache/scatter",
]

# Nested inside attention/kv_cache — shown for detail, not summed.
KV_DETAIL_REGIONS = ["kv_cache/cat", "kv_cache/paged_write", "kv_cache/paged_read"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--out-dir", type=str, default="profiles")
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C", "D", "E", "M"],
                        choices=["A", "B", "C", "D", "E", "M"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


# ======================================================================
# Report printing
# ======================================================================


def print_result(r, note=""):
    print(f"\n{'-' * 76}")
    print(f"{r.name}  ({r.steps} step{'s' if r.steps > 1 else ''}){'  — ' + note if note else ''}")
    print(f"{'-' * 76}")
    print(f"  wall (unprofiled)   {r.wall_ms_per_step:8.3f} ms/step")
    print(f"  GPU busy            {r.gpu_busy_ms_per_step:8.3f} ms/step   "
          f"-> GPU utilization {100 * r.gpu_utilization:5.1f}%")
    print(f"  kernel launches     {r.launches_per_step:8.1f} /step")

    total_gpu = r.gpu_busy_ms or 1e-9

    print(f"\n  {'region':<24} {'calls':>6} {'CPU ms':>9} {'GPU ms':>9} {'GPU %':>7}")
    labelled = 0.0
    for name in LEAF_REGIONS:
        st = r.region(name)
        if st.calls == 0:
            continue
        labelled += st.gpu_ms
        print(f"  {name:<24} {st.calls:>6} {st.cpu_ms:>9.3f} {st.gpu_ms:>9.3f} "
              f"{100 * st.gpu_ms / total_gpu:>6.1f}%")
    print(f"  {'(unlabelled)':<24} {'':>6} {'':>9} {r.gpu_busy_ms - labelled:>9.3f} "
          f"{100 * (r.gpu_busy_ms - labelled) / total_gpu:>6.1f}%")

    for name in KV_DETAIL_REGIONS:
        st = r.region(name)
        if st.calls:
            print(f"    within kv_cache: {name:<19} {st.calls:>4} calls "
                  f"{st.cpu_ms:>8.3f} CPU ms {st.gpu_ms:>8.3f} GPU ms")

    print(f"\n  GPU time by kernel category:")
    for category, ms in r.categories().items():
        print(f"    {category:<14} {ms:>8.3f} ms  {100 * ms / total_gpu:>5.1f}%")

    print(f"\n  top kernels:")
    for k in r.top_kernels(6):
        print(f"    {k.gpu_ms:>7.3f} ms  x{k.calls:<4} [{k.category}] {k.name[:70]}")

    if r.ops_by_shape:
        print(f"\n  top matmul / attention / KV ops by input shape:")
        for op in r.ops_by_shape[:6]:
            print(f"    {op.gpu_ms:>7.3f} GPU ms  x{op.calls:<4} {op.name:<36} {op.shapes[:60]}")

    if r.files:
        print(f"\n  files: {r.files.get('trace')}")


# ======================================================================
# Experiments
# ======================================================================


class Lab:
    def __init__(self, model, tokenizer, device, out_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.out = Path(out_dir)

        self.sampler = SamplingParams(greedy=True).to_sampler()
        self.runner = ModelRunner(model, tokenizer, self.sampler, device=device)

        base = tokenizer.encode(TEXT)
        self.token_pool = (base * (1 + 512 // len(base)))[:512]

    def ids(self, n, batch=1):
        return torch.tensor([self.token_pool[:n]] * batch, dtype=torch.long, device=self.device)

    def prompt_text(self, n):
        return self.tokenizer.decode(self.token_pool[:n])

    def profile(self, name, fn, steps=1, setup=None):
        return profile_workload(name, fn, self.device, steps=steps, setup=setup,
                                out_dir=self.out / name)

    # ---- A / C: prefill
    def prefill(self, n):
        ids = self.ids(n)

        @torch.inference_mode()
        def fn():
            out = self.runner.prefill(ids)
            with region("sampling"):
                out.logits.argmax(dim=-1).item()

        return self.profile(f"prefill_{n}", fn)

    # ---- B: decode on a warm cache
    def decode(self, context, steps):
        ids = self.ids(context)

        @torch.inference_mode()
        def setup():
            out = self.runner.prefill(ids)
            return out.kv_cache, out.logits.argmax(dim=-1, keepdim=True)

        @torch.inference_mode()
        def fn(state):
            cache, token = state
            for _ in range(steps):
                out = self.runner.decode(token, cache)
                with region("sampling"):
                    token = out.logits.argmax(dim=-1, keepdim=True)
                    token.item()

        return self.profile(f"decode_{steps}", fn, steps=steps, setup=setup)

    # ---- D: batched decode through the engine
    def batch_decode(self, batch_size, steps, paged=False, context=32):
        manager = KVCacheManager.for_model(self.model, num_blocks=8 * batch_size + 8,
                                           device=self.device)

        def setup():
            engine = InferenceEngine(self.runner, self.tokenizer, self.sampler,
                                     device=self.device, paged_kv=paged,
                                     kv_cache_manager=manager)
            requests = [
                engine.create_request_from_ids(self.token_pool[i:i + context + i * 3],
                                               max_new_tokens=steps + 2)
                for i in range(batch_size)
            ]
            engine.prefill_batch(requests)
            return engine, requests

        def fn(state):
            engine, requests = state
            for _ in range(steps):
                engine.decode_batch(requests)
            for request in requests:
                engine._release(request)

        name = f"batch_{batch_size}{'_paged' if paged else ''}"
        return self.profile(name, fn, steps=steps, setup=setup)

    # ---- E: naive vs cached, same tokens
    def naive_vs_cached(self, prompt_len, new_tokens):
        text = self.prompt_text(prompt_len)
        ids = self.ids(len(self.tokenizer.encode(text)))

        def naive():
            return self.runner.generate(text, max_new_tokens=new_tokens)["token_ids"]

        @torch.inference_mode()
        def cached():
            out = self.runner.prefill(ids)
            tokens = ids[0].tolist()
            with region("sampling"):
                token = out.logits.argmax(dim=-1, keepdim=True)
                tokens.append(token.item())
            for _ in range(new_tokens - 1):
                out = self.runner.decode(token, out.kv_cache)
                with region("sampling"):
                    token = out.logits.argmax(dim=-1, keepdim=True)
                    tokens.append(token.item())
            return tokens

        same = naive() == cached()

        return (
            self.profile(f"naive_{new_tokens}", naive, steps=new_tokens),
            self.profile(f"cached_{new_tokens}", cached, steps=new_tokens),
            same,
        )

    # ---- M: memory
    def memory(self):
        mb = 1024 ** 2
        config = self.model.config
        head_dim = config.hidden_dim // config.num_q_heads
        weights = sum(p.numel() * p.element_size() for p in self.model.parameters())
        kv_per_token = 2 * config.num_layers * config.num_kv_heads * head_dim * 4

        report = {
            "weights_mb": weights / mb,
            "kv_bytes_per_token": kv_per_token,
            "kv_mb_per_full_sequence": kv_per_token * config.max_seq_len / mb,
        }

        def kv_mb(cache):
            return sum(
                t.numel() * t.element_size()
                for layer in range(cache.num_layers)
                for t in cache.get(layer)
            ) / mb

        @torch.inference_mode()
        def prefill(n):
            return lambda: self.runner.prefill(self.ids(n))

        for n in (8, 128):
            stats = measure_peak_memory(prefill(n), self.device)
            report[f"prefill_{n}"] = stats

        cache_128 = self.runner.prefill(self.ids(128)).kv_cache
        report["kv_cache_128_tokens_mb"] = kv_mb(cache_128)

        @torch.inference_mode()
        def decode_one():
            token = self.ids(1)
            return self.runner.decode(token, cache_128)

        report["decode_step_ctx_128"] = measure_peak_memory(decode_one, self.device)

        for b in (1, 2, 4):
            requests = [
                InferenceEngine(self.runner, self.tokenizer, self.sampler, device=self.device,
                                paged_kv=False,
                                kv_cache_manager=KVCacheManager.for_model(self.model, num_blocks=1,
                                                                          device=self.device))
                .create_request_from_ids(self.token_pool[:64], max_new_tokens=4)
                for _ in range(b)
            ]
            batch = build_prefill_batch(requests, device=self.device)

            @torch.inference_mode()
            def prefill_batch(batch=batch):
                return self.runner.prefill_batch(batch)

            report[f"batch_{b}_prefill_64"] = measure_peak_memory(prefill_batch, self.device)

        @torch.inference_mode()
        def snapshot_workload():
            out = self.runner.prefill(self.ids(128))
            token = out.logits.argmax(dim=-1, keepdim=True)
            for _ in range(20):
                out = self.runner.decode(token, out.kv_cache)
                token = out.logits.argmax(dim=-1, keepdim=True)

        report["snapshot"] = capture_memory_snapshot(
            snapshot_workload, self.out / "memory" / "snapshot.pickle", self.device
        )

        return report


# ======================================================================
# Findings document
# ======================================================================


def pct(part, whole):
    return 100 * part / whole if whole else 0.0


def region_share(r, names):
    return pct(sum(r.region(n).gpu_ms for n in names), r.gpu_busy_ms)


def gemm_share(r):
    c = r.categories()
    return pct(c.get("gemm", 0) + c.get("gemv", 0), r.gpu_busy_ms)


def write_findings(path, results, memory, device_name, checkpoint, naive_same, vocab_size):
    lines = [
        "# Phase 13 — PyTorch execution profiling findings",
        "",
        f"Generated by `scripts/profile_inference.py` on **{device_name}**, checkpoint "
        f"`{checkpoint}`. All numbers come from this run; rerun the script to refresh them.",
        "",
        "Wall time is measured without the profiler and with regions off; GPU times are "
        "summed kernel durations from the profiled run. `GPU utilization` = GPU busy / wall.",
        "",
        "## Per-experiment summary",
        "",
        "| experiment | steps | wall ms/step | GPU busy ms/step | GPU util | kernels/step "
        "| matmul (gemm+gemv) share | attention/sdpa share |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results.values():
        lines.append(
            f"| {r.name} | {r.steps} | {r.wall_ms_per_step:.3f} | {r.gpu_busy_ms_per_step:.3f} "
            f"| {100 * r.gpu_utilization:.1f}% | {r.launches_per_step:.0f} "
            f"| {gemm_share(r):.1f}% | {region_share(r, ['attention/sdpa']):.1f}% |"
        )

    def region_table(r):
        out = ["| region | CPU ms | GPU ms | GPU % |", "|---|---|---|---|"]
        for name in LEAF_REGIONS:
            st = r.region(name)
            if st.calls:
                out.append(f"| {name} | {st.cpu_ms:.3f} | {st.gpu_ms:.3f} "
                           f"| {pct(st.gpu_ms, r.gpu_busy_ms):.1f}% |")
        return out

    def category_line(r):
        return ", ".join(f"{c} {pct(ms, r.gpu_busy_ms):.0f}%" for c, ms in r.categories().items())

    lines += ["", "## Findings", ""]

    n = 1

    def add(title, body):
        nonlocal n
        lines.extend([f"### {n}. {title}", "", *body, ""])
        n += 1

    if "prefill_128" in results and "decode_20" in results:
        p, d = results["prefill_128"], results["decode_20"]
        add("Prefill and decode are different workloads", [
            f"- **Prefill (128 tokens)**: {p.wall_ms_per_step:.2f} ms wall, GPU busy "
            f"{p.gpu_busy_ms_per_step:.2f} ms (**{100 * p.gpu_utilization:.0f}%** utilization). "
            f"Kernels: {category_line(p)}.",
            f"- **Decode (1 token, 128-token cache)**: {d.wall_ms_per_step:.2f} ms/step wall, "
            f"GPU busy {d.gpu_busy_ms_per_step:.3f} ms/step (**{100 * d.gpu_utilization:.0f}%**). "
            f"Kernels: {category_line(d)}.",
            f"- Prefill processes 128x more tokens but costs only "
            f"{p.wall_ms_per_step / d.wall_ms_per_step:.1f}x the wall time of one decode step: "
            f"per token, prefill is {128 * d.wall_ms_per_step / p.wall_ms_per_step:.0f}x cheaper.",
            f"- Both launch about the same number of kernels per forward "
            f"({p.launches_per_step:.0f} vs {d.launches_per_step:.0f}); decode's kernels are just "
            f"tiny. In decode every Linear has M=1 row, so cuBLAS picks **gemv** "
            f"(matrix-vector) kernels instead of gemm.",
        ])

    if "decode_20" in results:
        d = results["decode_20"]
        cpu_per_step = sum(d.region(n).cpu_ms for n in ("embedding", "rmsnorm", "attention",
                                                        "mlp", "lm_head", "sampling")) / d.steps
        add("CPU vs GPU time: decode is launch/dispatch-bound", [
            f"- Per decode step the GPU works {d.gpu_busy_ms_per_step:.3f} ms out of "
            f"{d.wall_ms_per_step:.3f} ms; the rest of the time it waits for the CPU to issue "
            f"~{d.launches_per_step:.0f} kernel launches through Python and the PyTorch dispatcher.",
            f"- Under the profiler the CPU spends ~{cpu_per_step:.2f} ms/step inside the model's "
            f"regions (inflated by profiler overhead) — far more than the GPU time.",
            "- Implication for Phase 14/15: faster GEMM kernels alone would barely move decode "
            "latency for this model; fewer launches (fusion, CUDA graphs, torch.compile) would.",
        ])
        add("Where decode GPU time goes (per region)", region_table(d))

    if "prefill_128" in results:
        p = results["prefill_128"]
        add("Where prefill GPU time goes (per region)", region_table(p) + [
            "",
            f"Matmuls (q/k/v/out projections, MLP, LM head) are {gemm_share(p):.0f}% of prefill "
            f"GPU time; the LM head alone (hidden x vocab={vocab_size}) is "
            f"{region_share(p, ['lm_head']):.0f}%.",
            "",
            f"Prefill computes logits for **all** {128} positions "
            f"([1, 128, {vocab_size}] = {128 * vocab_size * 4 / 1024 ** 2:.1f} MB fp32) although "
            f"generation only reads the last one — so the LM-head matmul does 128x more work "
            f"than needed and dominates prefill's transient memory. (Observation only; a "
            f"candidate optimization for later phases.)",
        ])

    if "prefill_8" in results and "prefill_128" in results:
        s, p = results["prefill_8"], results["prefill_128"]
        add("Short vs long prompt", [
            f"- 8 tokens: {s.wall_ms_per_step:.2f} ms wall, {s.gpu_busy_ms_per_step:.3f} ms GPU "
            f"({100 * s.gpu_utilization:.0f}% util).",
            f"- 128 tokens: {p.wall_ms_per_step:.2f} ms wall, {p.gpu_busy_ms_per_step:.3f} ms GPU "
            f"({100 * p.gpu_utilization:.0f}% util).",
            f"- 16x the tokens costs {p.wall_ms_per_step / s.wall_ms_per_step:.2f}x the wall time: "
            f"a short prompt is dominated by fixed per-forward launch overhead.",
        ])

    batch_keys = [k for k in ("batch_1", "batch_2", "batch_4") if k in results]
    if batch_keys:
        rows = ["| batch | wall ms/step | GPU busy ms/step | GPU util | kernels/step | ms per token |",
                "|---|---|---|---|---|---|"]
        for k in batch_keys + (["batch_4_paged"] if "batch_4_paged" in results else []):
            r = results[k]
            b = int(k.split("_")[1])
            rows.append(f"| {k} | {r.wall_ms_per_step:.3f} | {r.gpu_busy_ms_per_step:.3f} "
                        f"| {100 * r.gpu_utilization:.0f}% | {r.launches_per_step:.0f} "
                        f"| {r.wall_ms_per_step / b:.3f} |")
        body = rows + [""]
        if "batch_1" in results and "batch_4" in results:
            b1, b4 = results["batch_1"], results["batch_4"]
            body.append(
                f"Going from batch 1 to 4 multiplies tokens per step by 4 but wall time per step "
                f"only by {b4.wall_ms_per_step / b1.wall_ms_per_step:.2f}x: the same launches now "
                f"carry 4 rows, which is why batching raised throughput in Phase 9/12.")
        if "batch_4_paged" in results and "batch_4" in results:
            bp, b4 = results["batch_4_paged"], results["batch_4"]
            extra = bp.launches_per_step - b4.launches_per_step
            body.append(
                f"Paged KV (Phase 10) at batch 4 adds ~{extra:.0f} kernel launches per step "
                f"(block gathers/scatters, index ops) and "
                f"{bp.wall_ms_per_step - b4.wall_ms_per_step:.2f} ms/step — the paging overhead "
                f"Phase 12 measured, now attributed to kv_cache/paged_* and batch_gather.")
        add("Batching: same kernels, more rows", body)

    if "naive_20" in results and "cached_20" in results:
        nv, cv = results["naive_20"], results["cached_20"]
        add("Naive vs KV cache", [
            f"- Outputs identical: {naive_same}.",
            f"- 20 tokens: naive {nv.wall_ms:.1f} ms vs cached {cv.wall_ms:.1f} ms wall; GPU busy "
            f"{nv.gpu_busy_ms:.2f} ms vs {cv.gpu_busy_ms:.2f} ms.",
            "- Naive re-runs the whole sequence every step, so its matmuls stay gemm-shaped with "
            "M = current length and attention is a full causal square; with the cache, M=1 "
            "(gemv) and attention is one query against the cached keys.",
            "- The cache adds `kv_cache/cat` (torch.cat growth) and an explicit attention mask "
            "(`attention/mask`), and removes almost all recomputation.",
            f"- Because this model is small and sequences are short, both are launch-bound "
            f"(GPU util naive {100 * nv.gpu_utilization:.0f}%, cached "
            f"{100 * cv.gpu_utilization:.0f}%), so the wall-clock gain is modest even though "
            f"GPU work drops.",
        ])

    if memory:
        m = memory
        body = [
            f"- Model weights: **{m['weights_mb']:.1f} MB** (fp32).",
            f"- KV cache: {m['kv_bytes_per_token']} bytes/token "
            f"(2 x layers x kv_heads x head_dim x 4 B) = "
            f"{m['kv_mb_per_full_sequence']:.1f} MB per full 512-token sequence; measured "
            f"{m['kv_cache_128_tokens_mb']:.2f} MB for 128 tokens. GQA (2 KV heads instead of 8) "
            f"makes this 4x smaller than full multi-head attention.",
        ]
        for key in ("prefill_8", "prefill_128", "decode_step_ctx_128",
                    "batch_1_prefill_64", "batch_2_prefill_64", "batch_4_prefill_64"):
            st = m.get(key)
            if st and st["transient_mb"] is not None:
                body.append(f"- {key}: transient working set {st['transient_mb']:.2f} MB "
                            f"(peak {st['peak_mb']:.1f} MB)")
        if m.get("snapshot"):
            body.append(f"- Allocation snapshot: `{m['snapshot']}` "
                        f"(drag into https://pytorch.org/memory_viz).")
        body.append("- Activations are short-lived and small next to the weights; what grows with "
                    "concurrency is the KV cache (Phase 10's pool), linearly in tokens held.")
        add("Memory", body)

    lines += [
        "## Traces",
        "",
        "Each experiment directory under `profiles/` has `trace.json` (open in "
        "https://ui.perfetto.dev), `ops.txt` (aten ops by GPU time) and `shapes.txt` "
        "(same, grouped by input shape).",
        "",
    ]

    Path(path).write_text("\n".join(lines))


# ======================================================================


def main():
    args = parse_args()

    print("=" * 76)
    print("PHASE 13 — PYTORCH EXECUTION PROFILING")
    print("=" * 76)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    model, _ = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    device_name = torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU"
    print(f"\nModel:  V1 / {checkpoint_path}")
    print(f"Device: {device_name}   torch {torch.__version__}")

    lab = Lab(model, tokenizer, args.device, args.out_dir)
    results = {}
    memory = None
    naive_same = None

    if "A" in args.experiments:
        results["prefill_128"] = lab.prefill(128)
        print_result(results["prefill_128"], "Experiment A: prefill, 128-token prompt")

    if "B" in args.experiments:
        results["decode_20"] = lab.decode(context=128, steps=20)
        print_result(results["decode_20"], "Experiment B: decode, 20 steps on a 128-token cache")

    if "C" in args.experiments:
        results["prefill_8"] = lab.prefill(8)
        print_result(results["prefill_8"], "Experiment C: prefill, 8-token prompt")

    if "D" in args.experiments:
        for b in (1, 2, 4):
            results[f"batch_{b}"] = lab.batch_decode(b, steps=10)
            print_result(results[f"batch_{b}"], f"Experiment D: batched decode, batch {b}")
        results["batch_4_paged"] = lab.batch_decode(4, steps=10, paged=True)
        print_result(results["batch_4_paged"], "Experiment D: batched decode, batch 4, paged KV")

    if "E" in args.experiments:
        nv, cv, naive_same = lab.naive_vs_cached(prompt_len=32, new_tokens=20)
        results["naive_20"], results["cached_20"] = nv, cv
        print_result(nv, "Experiment E: naive (full recompute), 20 tokens")
        print_result(cv, "Experiment E: KV cache, 20 tokens")
        print(f"\n  naive and cached outputs identical: {naive_same}")

    if "M" in args.experiments:
        memory = lab.memory()
        print(f"\n{'-' * 76}\nMemory\n{'-' * 76}")
        print(f"  weights               {memory['weights_mb']:.1f} MB")
        print(f"  KV cache              {memory['kv_bytes_per_token']} B/token, "
              f"{memory['kv_mb_per_full_sequence']:.1f} MB per 512-token sequence")
        for key, value in memory.items():
            if isinstance(value, dict) and value.get("transient_mb") is not None:
                print(f"  {key:<22}transient {value['transient_mb']:7.2f} MB   "
                      f"peak {value['peak_mb']:7.1f} MB")
        if memory.get("snapshot"):
            print(f"  snapshot              {memory['snapshot']}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    findings = out / "FINDINGS.md"
    write_findings(findings, results, memory, device_name, checkpoint_path, naive_same,
                   vocab_size=model.config.vocab_size)

    ok = naive_same is not False and all(r.kernel_launches > 0 or not args.device.startswith("cuda")
                                         for r in results.values())

    print(f"\nFindings written to {findings}")
    print("\n" + "=" * 76)
    print("PHASE 13 PASSED" if ok else "PHASE 13 FAILED")
    print("=" * 76)

    if not ok:
        raise RuntimeError("Profiling checks failed.")


if __name__ == "__main__":
    main()
