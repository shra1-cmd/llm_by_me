# Phase 14 — CUDA / GEMM / cuBLAS investigation

Phase 13 showed **where** GPU time goes. Phase 14 explains **why**: it traces what an
`nn.Linear` in the V1 model turns into on the GPU, then measures those matrix
multiplications on their own. Nothing is optimized here. That is Phase 15's job.

Rule: **observe → trace → measure → explain.**

```text
Inference engine (prefill: 128 tokens | decode: 1 token)
  -> nn.Linear / x @ E.T            [M,K] x [K,N],  M = 128 | 1
  -> aten::linear -> aten::t (view) + aten::matmul -> aten::mm
  -> at::cuda::blas::gemm -> cuBLAS / cuBLASLt (column-major, C^T = W · x^T, "tn")
  -> kernel selection by (M, N, K, dtype, layout, math mode, sm_86)
       M = 1      internal::gemvx / gemv2T_kernel_val     (GEMV, memory-bound)
       M = 2..16  gemmSN_TN_kernel                        (small-N GEMM)
       M >= 32    ampere_sgemm_<tile>_tn (+ splitKreduce) (tiled GEMM, compute-bound)
  -> SMs: FP32 FFMA on CUDA cores (TF32 would use Tensor Cores: cutlass_80_tensorop_s1688gemm)
```

## Files

| file | what it is |
|---|---|
| `gemm_benchmark.py` | runs every experiment, writes `results.json` and `findings.md` |
| `results.json` | raw numbers: roofline, traces, tensor layouts, M sweep, TF32, batching, KV `cat` |
| `findings.md` | answers to the ten Phase 14 questions, generated from `results.json` |

## Run

```bash
python profiles/phase14/gemm_benchmark.py            # ~10 s on an RTX 3050 Laptop
python profiles/phase14/gemm_benchmark.py --quick    # fewer timed iterations
pytest tests/test_gemm_benchmark.py                  # checks the tooling itself
```

It needs a CUDA GPU. It loads the latest checkpoint in `checkpoints/v1` (used only to
inspect real tensor layouts) and falls back to random weights if none is found. The
Phase 13 numbers it cross-checks against are read from `profiles/FINDINGS.md`.

## Experiments → spec sections

| experiment | spec | method |
|---|---|---|
| `trace` | 14.3 | `TorchDispatchMode` logs every aten op with shapes/strides; `torch.profiler` op tree shows which kernel each op launched |
| `model_tensors` | 14.4, 14.9 | forward hooks on layer 0's Linears + LM head during a real prefill/decode: shape, stride, dtype, contiguity |
| `layer_table` | 14.4, 14.5 | M, K, N, FLOPs = 2MKN, minimum bytes, arithmetic intensity per model matmul |
| `roofline` | 14.6, 14.10 | measured FP32 / TF32 peak, DRAM bandwidth, CPU cost of one tiny kernel launch |
| `m_sweep` | 14.6, 14.8, 14.11 | every model Linear shape at M = 1…512: wall time, GPU kernel time, kernels, TFLOPS, GB/s, CTAs vs SMs, split-K |
| `tf32_sweep` | 14.10 | same shapes with TF32 allowed (restored afterwards): which kernels change |
| `batching` | 14.12 | B separate `[1,K]` calls vs one `[B,K]` call |
| `forward_estimate` | 14.5, 14.6 | isolated kernel times summed over a full forward, compared with Phase 13 |
| `kv_cat` | 14.13 | `torch.cat` cost vs cache length, allocations per decode step, total bytes copied |

## Measurement notes

- **wall us**: CPU time per call over back-to-back calls with one sync at the end, i.e.
  max(CPU issue cost, GPU time). **GPU / kernel us**: kernel durations from
  `torch.profiler`. When wall is much larger than kernel time, the op is launch-bound.
- Laptop GPU clocks change with power and temperature, so numbers move by around ±10%
  between runs. Look at ratios and trends, not the exact digits.
- The isolated loops reuse the same weights, so weights of 1 MB or less can stay in
  the 1 MB L2. In the full model they get evicted between calls.
- Model precision is untouched (fp32, `float32_matmul_precision = "highest"`). The
  TF32 run only switches precision inside a context manager.
