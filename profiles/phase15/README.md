# Phase 15 — Kernel-level optimization

Phase 13 found where decode time goes. Phase 14 explained why: decode is **launch-bound**
(~424 tiny kernels per step, each costing more CPU time than it runs on the GPU), and its
M=1 GEMVs are memory-bound. Phase 15 makes the engine faster **without changing its
outputs**, one measured change at a time.

```text
baseline ──► make ONE change ──► correctness gate ──► paired benchmark ──► profile ──► keep / reject
   ▲                                                                                  │
   └──────────────────────── kept changes become the new "previous" ◄─────────────────┘
```

## What was added to the engine (all opt-in, defaults unchanged)

| change | where | how to enable |
|---|---|---|
| Preallocated KV cache (no `torch.cat`) | `StaticKVCache` in `src/inference/kv_cache.py` | `ModelRunner(..., kv_cache="static")` |
| Precomputed interleaved RoPE tables | `src/model/rope.py` | flag `rope_cache` |
| Fused RMSNorm (`F.rms_norm`) | `src/model/rmsnorm.py` | flag `fused_rmsnorm` |
| No mask for a single decode query | `src/model/attention.py` | flag `decode_no_mask` |
| SDPA `enable_gqa` instead of `repeat_interleave` | `src/model/attention.py` | flag `sdpa_gqa` |
| One q+k+v matmul | `src/model/attention.py` | flag `fused_qkv` (needs `fast_paths.prepare`) |
| One gate+up matmul | `src/model/swiglu.py` | flag `fused_gate_up` (needs `fast_paths.prepare`) |
| LM head on the last position only | `src/model/model.py` | flag `last_token_logits` |
| `torch.compile(model, dynamic=True)` | harness | wrap the model |

```python
from src.model import fast_paths

fast_paths.prepare(model)          # fused q/k/v and gate/up weights (views: no extra memory)
fast_paths.set_flags(rope_cache=True, fused_rmsnorm=True, decode_no_mask=True,
                     fused_qkv=True, last_token_logits=True)
runner = ModelRunner(torch.compile(model, dynamic=True), tokenizer, sampler, kv_cache="static")
```

With every flag off, `kv_cache="dynamic"` and no compile, the model runs exactly the
Phase 13 code path. That path is the frozen baseline.

## Correctness gate

`src/inference/correctness.py` compares any variant against the baseline:

1. prefill logits
2. teacher-forced decode logits (both variants are fed the same tokens)
3. greedy tokens
4. EOS stop step
5. cached decode vs a full recompute
6. prompts of different lengths through the continuous-batching engine: they must match the
   reference, and each must match its own single-request run

Logits need max abs error ≤ 1e-3 and mean abs error ≤ 1e-4. The observed errors are
~3e-5, the same scale as Phase 5's 3.05e-5. Tokens must match exactly.

## Files

| path | contents |
|---|---|
| `phase15_benchmark.py` | the ladder and side experiments; writes everything below |
| `results.json` | every number |
| `FINDINGS.md` | baseline, ladder table, and answers to the Phase 15 questions (generated) |
| `baseline/`, `kv_preallocated/`, `rope_cache/`, …, `torch_compile/` | per-rung `prefill/` and `decode/` profiles |
| `batching/` | decode profiles for B = 1…16 |
| `baseline_rerun/` | the baseline measured again at the end (drift check) |

Each profile directory has an `ops.txt`. The `trace.json` files (open them in
https://ui.perfetto.dev) and `shapes.txt` are written too, but git-ignored because of
their size.

## Run

```bash
python profiles/phase15/phase15_benchmark.py                 # full run (~10-15 min, RTX 3050 Laptop)
python profiles/phase15/phase15_benchmark.py --quick         # fewer repeats / paired rounds
python profiles/phase15/phase15_benchmark.py --skip-compile  # no torch.compile rung
pytest tests/test_phase15.py                                 # cache, flags and gate tests (CPU)
```

## Measurement notes

- **Paired timing.** Laptop clocks drift by tens of percent over a run. In one run the same
  baseline measured 3.3 ms/token at the start and 4.7 ms at the end. So every keep/reject
  decision, and the headline baseline→final numbers, interleave the two configs (A B B A …)
  and use the median per-round ratio. The absolute numbers of a single rung are only
  comparable within that rung.
- **Keep rule.** A change is kept only if the gate passes and decode or prefill gets at
  least 3% faster, with no regression beyond 3% (`--threshold`).
- **torch.compile.** The first calls compile (seconds), so the report gives cold-start and
  steady-state numbers separately. Profiling it runs with the named regions off, so the
  `record_function` calls don't trigger a recompile.
