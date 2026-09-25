"""
Phase 16 — freeze our engine and compare it with vLLM on the same workload.

Three jobs, one script:

    freeze (16.1)   record exactly what "our engine, final version" is:
                    model, checkpoint, tokenizer, engine components,
                    scheduler / batching / KV / sampling settings,
                    CUDA/runtime configuration, source-file hashes, and
                    the stored Phase 13-15 headline numbers
    ours   (16.17)  run the shared workload through our engine with the
                    Phase 12 benchmark harness (src/inference/benchmark.py):
                      baseline   Phase 6 KV cache (1 request) and Phase 11
                                 continuous batching, as built
                      optimized  same modes with the Phase 15 final
                                 configuration (fast paths + torch.compile,
                                 read from profiles/phase15/results.json)
    vllm   (16.17)  run the same workload through vLLM (V1 engine) on the
                    Llama export of the same checkpoint
                    (profiles/phase16/export_llama.py), stepping
                    LLMEngine by hand so every token is timestamped the
                    same way as ours

Same model weights, prompt token ids, max_new_tokens, greedy sampling,
EOS token, GPU and dtype (fp32) on both sides; tokens are compared.

    workload.json          shared prompts (token ids) + generation settings
    engine_freeze.json     the freeze record
    results_ours.json      our engine's metrics
    results_vllm.json      vLLM's metrics (only if run)
    COMPARISON.md          generated from whatever results exist

vLLM is not installable next to this project's environment (it pins its
own PyTorch/CUDA build), so run the vLLM side in its own environment:

    python profiles/phase16/export_llama.py                # once: models/v1-llama
    python profiles/phase16/engine_vs_vllm.py              # freeze + ours + report
    # in a separate venv with vllm installed, from the repo root:
    python profiles/phase16/engine_vs_vllm.py --engine vllm
    python profiles/phase16/engine_vs_vllm.py --report     # regenerate COMPARISON.md

The vLLM path needs only vllm + torch (no project imports), so it runs
in a clean vLLM environment.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

WORKLOAD = HERE / "workload.json"
FREEZE = HERE / "engine_freeze.json"
OURS = HERE / "results_ours.json"
VLLM = HERE / "results_vllm.json"
REPORT = HERE / "COMPARISON.md"

PROMPT_TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play "
    "outside in the sunshine with her dog, Max. One day, they went to the park "
    "and saw a big red ball under a tree. Lily ran to the ball and kicked it high "
    "into the sky. Max barked and chased it across the grass. Then a boy named Tom "
    "came to play with them. They laughed and played all day until the sun went "
    "down, and then they walked home together, happy and tired. "
)


# ======================================================================
# Shared helpers (no project imports: the vLLM side runs in its own env)
# ======================================================================


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def trace_metrics(t0: float, t_end: float, traces: dict[str, list[float]], peak_bytes) -> dict:
    """Same definitions as src/inference/benchmark.py RunResult.metrics()."""

    ttfts = [times[0] - t0 for times in traces.values() if times]
    itls = [(b - a) * 1e3 for times in traces.values() for a, b in zip(times, times[1:])]
    tokens = sum(len(t) for t in traces.values())
    total = t_end - t0
    return {
        "total_s": total,
        "generated_tokens": tokens,
        "tokens_per_s": tokens / total if total > 0 else 0.0,
        "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
        "itl_mean_ms": statistics.mean(itls) if itls else None,
        "itl_p50_ms": statistics.median(itls) if itls else None,
        "peak_mb": peak_bytes / 2 ** 20 if peak_bytes is not None else None,
    }


def aggregate(runs: list[dict]) -> dict:
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    return {k: summarize([r[k] for r in runs if r[k] is not None]) for k in keys}


# ======================================================================
# Workload + freeze
# ======================================================================


def build_workload(args) -> dict:
    sys.path.insert(0, str(ROOT))
    from src.tokenizer.tokenizer import BPETokenizer

    tokenizer = BPETokenizer(str(ROOT / args.tokenizer_path))
    pool = tokenizer.encode(PROMPT_TEXT * 8)

    def text_of(n, offset=0):
        # decode/encode round trip so both engines see exactly these ids
        ids = pool[offset:offset + n]
        text = tokenizer.decode(ids)
        return text, tokenizer.encode(text)

    single_text, single_ids = text_of(args.prompt_len)
    batch = []
    for i in range(args.batch_requests):
        n = 32 + (i * 37) % (args.prompt_len - 31)          # 32..prompt_len, mixed
        text, ids = text_of(n, offset=11 * i)
        batch.append({"request_id": f"b{i}", "prompt": text, "prompt_token_ids": ids})

    return {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.token_to_id("<eos>"),
        "sampling": "greedy",
        "dtype": "float32",
        "batch_size": args.batch_size,
        "scenarios": {
            "single": [{"request_id": "s0", "prompt": single_text, "prompt_token_ids": single_ids}],
            "batch": batch,
        },
    }


def file_sha(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def git(*cmd) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def stored_phase_results() -> dict:
    """Headline numbers already produced by Phases 13-15 (read, not re-measured)."""

    out = {}
    p15 = ROOT / "profiles" / "phase15" / "results.json"
    if p15.exists():
        r = json.loads(p15.read_text())
        fin, base = r["final"], r["ladder"][0]["metrics"]
        out["phase15"] = {
            "source": "profiles/phase15/results.json",
            "final_config": fin["config"],
            "paired_vs_baseline": fin.get("paired_vs_baseline"),
            "baseline_decode_launches_per_step": base["decode"]["launches_per_step"],
            "final_decode_launches_per_step": fin["metrics"]["decode"]["launches_per_step"],
            "gate_passed": fin["gate"]["passed"],
            "gate_max_abs_error": fin["gate"]["max_abs_error"],
            "ladder": [{"name": s["name"], "decision": s["decision"]} for s in r["ladder"]],
        }
    p14 = ROOT / "profiles" / "phase14" / "results.json"
    if p14.exists():
        r = json.loads(p14.read_text())
        out["phase14"] = {"source": "profiles/phase14/results.json", "roofline": r["roofline"],
                          "forward_estimate": r["forward_estimate"]}
    p13 = ROOT / "profiles" / "FINDINGS.md"
    if p13.exists():
        out["phase13"] = {"source": "profiles/FINDINGS.md",
                          "summary_rows": [l for l in p13.read_text().splitlines()
                                           if l.startswith("| prefill_128") or l.startswith("| decode_20")]}
    return out


def freeze(args) -> dict:
    sys.path.insert(0, str(ROOT))
    import torch
    from src.inference.checkpoint_loader import find_latest_checkpoint
    from src.inference.kv_cache_manager import KVCacheManager
    from src.inference.request import SamplingParams
    from src.tokenizer.tokenizer import BPETokenizer
    from src.model import fast_paths

    checkpoint = Path(args.checkpoint or find_latest_checkpoint(ROOT / args.checkpoint_dir))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer = BPETokenizer(str(ROOT / args.tokenizer_path))
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    for_model_defaults = KVCacheManager.for_model.__defaults__

    sources = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "configs" / "v1.py"]
    return {
        "name": "our engine — final version",
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": {"commit": git("rev-parse", "--short", "HEAD"), "branch": git("branch", "--show-current"),
                "dirty": bool(git("status", "--porcelain", "--", "src", "configs"))},
        "model": {"class": "src.model.model.V1LanguageModel", "config": ckpt.get("model_config"),
                  "architecture": "decoder-only, GQA attention + RoPE (interleaved pairs), "
                                  "pre-RMSNorm, SwiGLU MLP, tied LM head, no biases"},
        "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256_12": file_sha(checkpoint),
                       "step": ckpt.get("step"), "loss": ckpt.get("loss")},
        "tokenizer": {"path": args.tokenizer_path, "sha256_12": file_sha(ROOT / args.tokenizer_path),
                      "type": "byte-level BPE (src/tokenizer/tokenizer.py)",
                      "vocab_size": tokenizer.vocab_size,
                      "special_tokens": {t: tokenizer.token_to_id(t)
                                         for t in ("<pad>", "<unk>", "<bos>", "<eos>")}},
        "engine": {
            "request": "src/inference/request.py InferenceRequest (WAITING→PREFILLING→DECODING→FINISHED/ABORTED)",
            "scheduler": "src/inference/scheduler.py FIFO Scheduler (waiting queue + running dict)",
            "batching": "src/inference/batch.py right-padded prefill batches, [B,1] decode batches",
            "continuous_batching": "src/inference/continuous_batching.py ADMIT→PREFILL→DECODE→RETIRE per step",
            "kv_cache": {"paged": "src/inference/kv_cache_manager.py KVCacheManager + PagedKVCache",
                         "contiguous": "src/inference/kv_cache.py KVCache (torch.cat) / StaticKVCache (Phase 15)",
                         "default_block_size": for_model_defaults[1],
                         "default_pool": f"{for_model_defaults[3]} full-length sequences"},
            "model_runner": "src/inference/model_runner.py prefill / decode / prefill_batch / decode_batch",
            "sampling": {"class": "src/inference/sampler.py Sampler",
                         "methods": ["greedy", "temperature", "top_k", "top_p",
                                     "repetition_penalty", "repeat_ngram_size"],
                         "defaults": SamplingParams().__dict__},
            "fast_paths": fast_paths.NAMES,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": props.name if props else None,
            "compute_capability": f"{props.major}.{props.minor}" if props else None,
            "sm_count": props.multi_processor_count if props else None,
            "gpu_memory_mb": props.total_memory // 2 ** 20 if props else None,
            "driver": git_free_driver(),
            "fp32_matmul_precision": torch.get_float32_matmul_precision(),
            "sdpa": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                     "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "math": torch.backends.cuda.math_sdp_enabled()},
        },
        "source_hashes": {str(p.relative_to(ROOT)): file_sha(p, 8) for p in sources},
        "stored_results": stored_phase_results(),
    }


def git_free_driver() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ======================================================================
# Our engine
# ======================================================================


def run_ours(args, workload: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    import torch
    from src.inference.benchmark import BenchmarkContext, BenchmarkRequest, benchmark_mode
    from src.inference.checkpoint_loader import find_latest_checkpoint, load_inference_checkpoint
    from src.inference.model_runner import ModelRunner
    from src.model import fast_paths
    from src.tokenizer.tokenizer import BPETokenizer

    checkpoint = args.checkpoint or find_latest_checkpoint(ROOT / args.checkpoint_dir)
    model, _ = load_inference_checkpoint(checkpoint, device="cuda")
    tokenizer = BPETokenizer(str(ROOT / args.tokenizer_path))

    p15 = json.loads((ROOT / "profiles" / "phase15" / "results.json").read_text())["final"]["config"]
    flags = {f: True for f in p15["flags"]}

    def requests(name):
        return [BenchmarkRequest(r["request_id"], r["prompt"], workload["max_new_tokens"])
                for r in workload["scenarios"][name]]

    ctx = BenchmarkContext(model, tokenizer, "cuda", batch_size=workload["batch_size"])

    fast_paths.prepare(model)
    opt_model = torch.compile(model, dynamic=True) if p15["compile"] else model
    if p15["compile"]:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    opt_runner = ModelRunner(opt_model, tokenizer, ctx.sampler, device="cuda", kv_cache=p15["kv_cache"])
    opt_runner.max_seq_len = model.config.max_seq_len

    results = {"phase15_config": p15, "engines": {}}
    plan = [("ours: baseline", ctx.runner, {}), ("ours: Phase 15 optimized", opt_runner, flags)]
    for label, runner, active in plan:
        ctx.runner = runner
        engine = {}
        with fast_paths.enabled(**active):
            for scenario, mode in (("single", "kv_cache"), ("batch", "continuous")):
                print(f"  {label}: {scenario} ({mode})")
                report = benchmark_mode(ctx, requests(scenario), mode,
                                        warmup=args.warmup, iterations=args.iterations)
                engine[scenario] = {
                    "mode": mode,
                    "stats": report.stats(),
                    "tokens": {k: v[len(tokenizer.encode(next(r["prompt"] for r in workload["scenarios"][scenario]
                                                             if r["request_id"] == k))):]
                               for k, v in report.outputs.items()},
                    "deterministic": report.deterministic,
                }
        results["engines"][label] = engine
    return results


# ======================================================================
# vLLM (runs in its own environment; imports only vllm + torch)
# ======================================================================


def run_vllm(args, workload: dict) -> dict:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # EngineCore in-process: same-process peak memory
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    llm = LLM(
        model=str(ROOT / args.vllm_model),
        skip_tokenizer_init=True,          # our BPE tokenizer: send token ids
        dtype="float32",
        max_model_len=512,
        max_num_seqs=workload["batch_size"],
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enforce_eager=args.vllm_eager,
        seed=0,
    )
    engine = llm.llm_engine
    params = SamplingParams(temperature=0.0, max_tokens=workload["max_new_tokens"],
                            stop_token_ids=[workload["eos_token_id"]], detokenize=False,
                            output_kind=RequestOutputKind.CUMULATIVE)   # token ids so far, every step

    def run(scenario) -> tuple[dict, dict]:
        reqs = workload["scenarios"][scenario]
        traces = {r["request_id"]: [] for r in reqs}
        tokens = {r["request_id"]: [] for r in reqs}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for r in reqs:
            engine.add_request(r["request_id"], TokensPrompt(prompt_token_ids=r["prompt_token_ids"]), params)
        while engine.has_unfinished_requests():
            for out in engine.step():
                now = time.perf_counter()
                ids = list(out.outputs[0].token_ids)
                new = len(ids) - len(tokens[out.request_id])
                traces[out.request_id].extend([now] * max(new, 0))
                tokens[out.request_id] = ids
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        return trace_metrics(t0, t_end, traces, torch.cuda.max_memory_allocated()), tokens

    results = {"vllm_version": vllm.__version__, "torch": torch.__version__,
               "enforce_eager": args.vllm_eager,
               "gpu_memory_utilization": args.vllm_gpu_memory_utilization, "scenarios": {}}
    for scenario in ("single", "batch"):
        for _ in range(args.warmup):
            run(scenario)
        runs, tokens = [], None
        for _ in range(args.iterations):
            m, tokens = run(scenario)
            runs.append(m)
        results["scenarios"][scenario] = {"stats": aggregate(runs), "tokens": tokens}
        print(f"  vllm {scenario}: {aggregate(runs)['tokens_per_s']['mean']:.0f} tok/s")
    return results


# ======================================================================
# Report
# ======================================================================


def write_report(workload, freeze_rec, ours, vllm_res):
    lines = []
    add = lines.append
    add("# Phase 16 — our engine vs vLLM\n")
    add("Generated by `profiles/phase16/engine_vs_vllm.py` from `workload.json`, "
        "`engine_freeze.json`, `results_ours.json` and `results_vllm.json` (if present).\n")

    rt = freeze_rec["runtime"] if freeze_rec else {}
    add(f"Workload: greedy, fp32, `max_new_tokens={workload['max_new_tokens']}`, EOS id "
        f"{workload['eos_token_id']}. **single** = 1 request with a "
        f"{len(workload['scenarios']['single'][0]['prompt_token_ids'])}-token prompt; **batch** = "
        f"{len(workload['scenarios']['batch'])} requests (prompts "
        f"{min(len(r['prompt_token_ids']) for r in workload['scenarios']['batch'])}–"
        f"{max(len(r['prompt_token_ids']) for r in workload['scenarios']['batch'])} tokens) arriving "
        f"together, max {workload['batch_size']} running at once. GPU: {rt.get('gpu')}.\n")

    rows = []
    if ours:
        for label, eng in ours["engines"].items():
            for scenario, r in eng.items():
                rows.append((label, scenario, r["stats"]))
    if vllm_res:
        label = f"vLLM {vllm_res['vllm_version']}" + (" (eager)" if vllm_res["enforce_eager"] else "")
        for scenario, r in vllm_res["scenarios"].items():
            rows.append((label, scenario, r["stats"]))

    def cell(stats, key, fmt, scale=1.0):
        v = stats.get(key)
        return "—" if not v else fmt.format(v["mean"] * scale)

    for scenario in ("single", "batch"):
        add(f"## {scenario}\n")
        add("| engine | TTFT ms (mean) | ITL ms (mean) | ITL ms (p50) | tokens/s | total s | peak MB |")
        add("|---|---|---|---|---|---|---|")
        for label, sc, st in rows:
            if sc == scenario:
                add(f"| {label} | {cell(st, 'ttft_mean_s', '{:.1f}', 1e3)} | {cell(st, 'itl_mean_ms', '{:.2f}')} | "
                    f"{cell(st, 'itl_p50_ms', '{:.2f}')} | {cell(st, 'tokens_per_s', '{:.0f}')} | "
                    f"{cell(st, 'total_s', '{:.3f}')} | {cell(st, 'peak_mb', '{:.0f}')} |")
        add("")

    if ours and vllm_res:
        base = ours["engines"]["ours: baseline"]
        same = {sc: base[sc]["tokens"] == vllm_res["scenarios"][sc]["tokens"] for sc in ("single", "batch")}
        add(f"Greedy tokens identical to vLLM: single {same['single']}, batch {same['batch']}. "
            "Small fp32 differences between kernels can flip a near-tie argmax late in a sequence; "
            "a mismatch there is not a bug by itself.\n")
    elif not vllm_res:
        add("**vLLM side not run yet.** vLLM cannot be installed into this project's environment "
            "(it pins its own PyTorch/CUDA build; this env has PyTorch "
            f"{rt.get('torch')} on Python {rt.get('python')}). Run "
            "`python profiles/phase16/engine_vs_vllm.py --engine vllm` in a separate vLLM "
            "environment, then `--report`. The model is already exported as a standard "
            "`LlamaForCausalLM` (`models/v1-llama`, verified against our model by "
            "`export_llama.py`), so vLLM runs it with no custom code.\n")

    add("## Which mechanisms explain the difference\n")
    add("| mechanism | our engine | vLLM | effect on this workload |")
    add("|---|---|---|---|")
    for row in MECHANISMS:
        add("| " + " | ".join(row) + " |")
    add("")
    add("See `Docs/FINAL_ARCHITECTURE.md` §18 for the component-by-component mapping with vLLM "
        "source references.\n")
    REPORT.write_text("\n".join(lines) + "\n")


MECHANISMS = [
    ("kernel launches per decode step",
     "~424 eager (Phase 13); ~130 with torch.compile (Phase 15)",
     "torch.compile'd model + CUDA graph replay: a decode step is ~1 graph launch",
     "the dominant factor at this model size: decode is launch-bound (Phase 14/15)"),
    ("batch layout", "prefill right-padded to [B,T_max] + separate [B,1] decode forward",
     "one flattened token batch per step (no padding), prefill + decode tokens mixed",
     "no wasted FLOPs on padding; one forward per step instead of two"),
    ("KV read for attention", "paged: gather blocks back into contiguous K/V every step; "
     "batched decode: pad + torch.cat all rows every step",
     "paged-attention kernel reads blocks in place via block_table; new K/V written by slot_mapping",
     "removes O(B·L) copies per step; grows with context and batch size"),
    ("attention kernel", "SDPA memory-efficient fp32 kernel, repeat_interleave for GQA, masks",
     "FlashAttention / FlashInfer / Triton varlen paged kernels with native GQA",
     "fewer launches and bytes; fp32 limits which backends apply"),
    ("LM head", "all positions (last-only is a Phase 15 opt-in)", "only logits_indices (last scheduled token)",
     "prefill LM head is ~26% of prefill FLOPs (Phase 14)"),
    ("sampling", "Python loop, one Sampler call per request", "one batched GPU sampler for all requests",
     "per-request Python overhead scales with batch size"),
    ("scheduling", "per step: admit (slots + KV), prefill newcomers, decode actives; no token budget",
     "token budget (max_num_batched_tokens), chunked prefill, preemption, prefix-cache lookup",
     "better TTFT/ITL balance under load; neutral for a 16-request offline batch"),
    ("process structure", "one Python process, synchronous", "EngineCore busy loop (optionally its own "
     "process), async output processing / detokenization",
     "CPU work overlaps GPU work"),
]


# ======================================================================
# Main
# ======================================================================


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--engine", choices=["ours", "vllm"], default="ours")
    parser.add_argument("--report", action="store_true", help="only regenerate COMPARISON.md")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-requests", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--vllm-model", type=str, default="models/v1-llama")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-eager", action="store_true", help="disable CUDA graphs / compile in vLLM")
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    args = parse_args()

    if not args.report:
        if args.engine == "ours":
            workload = build_workload(args)
            WORKLOAD.write_text(json.dumps(workload, indent=2))
            print("[freeze]")
            FREEZE.write_text(json.dumps(freeze(args), indent=2, default=str))
            print("[our engine]")
            OURS.write_text(json.dumps(run_ours(args, workload), indent=2, default=str))
        else:
            workload = load(WORKLOAD)
            if workload is None:
                sys.exit("workload.json missing: run the 'ours' side first")
            print("[vLLM]")
            VLLM.write_text(json.dumps(run_vllm(args, workload), indent=2, default=str))

    workload = load(WORKLOAD)
    if workload is None:
        sys.exit("nothing to report: run the 'ours' side first")
    write_report(workload, load(FREEZE), load(OURS), load(VLLM))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
