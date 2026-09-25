# llm_by_me — Final Architecture

This is the final document of the project (Phase 16). It follows one generated token through
the whole system: from the user's request through the scheduler, the KV cache, PyTorch, ATen,
cuBLAS and the GPU, and back. It then maps every component onto vLLM's V1 engine. Training
and infrastructure are covered separately in `Docs/V1_DETAILED_DOCUMENTATION.md`; this document
is about inference.

Conventions:

- `path:line` refers to this repository at the Phase 16 freeze (`profiles/phase16/engine_freeze.json`).
- `vllm/...:line` refers to vLLM commit `8b84e15` (2026-09-24), V1 engine.
- Numbers come from stored results. Each one names its source file.
- Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere, sm_86, 20 SMs), fp32.

```text
                          REQUEST ("Once upon a time")
                                     │  tokenizer.encode
                                     ▼
       InferenceRequest (WAITING → PREFILLING → DECODING → FINISHED/ABORTED)
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                 ▼
  Scheduler + ContinuousBatchingEngine            KVCacheManager (paged blocks)
  "which requests run this step?"                 "where does their K/V live?"
            └────────────────────────┬────────────────────────┘
                                     ▼
                  Batch (prefill [B,T] padded | decode [B,1])
                                     ▼
                  ModelRunner → V1LanguageModel.forward
                                     ▼
       PyTorch ops (nn.Linear, SDPA, RMSNorm, RoPE, SiLU, cat, ...)
                                     ▼
       ATen (aten::linear → aten::mm, aten::_efficient_attention_forward, ...)
                                     ▼
       CUDA backend → cuBLAS/cuBLASLt (GEMM/GEMV) | CUTLASS fmha | elementwise kernels
                                     ▼
       GPU: 20 SMs, FP32 CUDA cores (Tensor Cores only with TF32/FP16/BF16)
                                     ▼
       logits [B, vocab] → Sampler → next token → KV cache updated → scheduler again
```

---

## 1. Model architecture

`V1LanguageModel` (`src/model/model.py:32`) is a Llama-family decoder. Its configuration is in
`configs/v1.py`; the checkpoint stores the real vocabulary size.

| | |
|---|---|
| parameters | 30.48M (fp32 weights ≈ 116 MB) |
| layers / hidden / FFN | 8 / 512 / 1408 (SwiGLU) |
| attention | GQA: 8 query heads and 2 KV heads, head_dim 64, RoPE (base 10 000, interleaved pairs) |
| norm | pre-RMSNorm (eps 1e-6), plus a final norm |
| vocab / context | 15 485 / 512 |
| LM head | tied: `logits = x @ token_embedding.weight.T` (`model.py:175`) |

A block is `x += attn(attn_norm(x)); x += mlp(ffn_norm(x))` (`src/model/block.py:57`).

Attention (`src/model/attention.py:104`) runs these steps:

1. q/k/v projections
2. split into heads, `[B,H,T,D]`
3. RoPE on the **new** tokens only, at their absolute positions
4. `kv_cache.update` appends the new K/V and returns past + new
5. `repeat_interleave` expands K/V from 2 heads to 8 for GQA
6. `F.scaled_dot_product_attention`: causal with no past, an explicit mask with a past or with padding
7. output projection

Two small modules support later phases:

- `src/model/profiling.py`: named profiler regions, which cost nothing when off (Phase 13).
- `src/model/fast_paths.py`: opt-in Phase 15 fast paths.

Because the architecture is Llama's, the checkpoint exports losslessly to a standard
`LlamaForCausalLM` (§18.9). The only change needed is permuting the q/k rows to switch RoPE
from interleaved pairs to half-split.

## 2. Tokenization

A byte-level BPE (HF `tokenizers`), trained on 1% of TinyStories
(`src/tokenizer/train_tokenizer.py`). It has 15 485 tokens and 15 379 merges. Special tokens:
`<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3. There is no post-processor, so `<bos>`/`<eos>` are
never added automatically. In the training data, `<eos>` is appended after each story.

`BPETokenizer` (`src/tokenizer/tokenizer.py:11`) wraps it: `encode`, `decode`, `token_to_id`.
The engine uses `<eos>` to stop generation and `<pad>` as the padding id.

## 3. Inference request

`InferenceRequest` (`src/inference/request.py:129`) holds all per-generation state: prompt
tokens, generated tokens, `SamplingParams`, `max_new_tokens`, EOS id, `max_seq_len`, its KV
cache handle, status and finish reason, and an `on_token` callback. The engine keeps no state
between requests.

```text
WAITING ──► PREFILLING ──► DECODING ──► FINISHED   (EOS | MAX_NEW_TOKENS | MAX_SEQ_LEN)
   │             │             │
   └─────────────┴─────────────┴──────► ABORTED    (OUT_OF_KV_BLOCKS | error)
```

The transitions are enforced by `_transition` (`request.py:102-119, 206`).

`check_stop` (`request.py:229`) runs after every appended token, in this order:

1. EOS (the EOS token is kept)
2. `max_new_tokens`
3. `position >= max_seq_len`

The last sampled token is never fed back through the model, so the final KV length is
`prompt + generated - 1`. The engine asserts this before every decode
(`inference_engine.py:25-28, 322`).

## 4. Scheduler

`Scheduler` (`src/inference/scheduler.py:45`) keeps a FIFO `waiting` deque and a `running`
dict. `next_batch(n, can_schedule)` takes requests from the head of the queue until the first
one that doesn't fit, so there is no queue-jumping. There are no priorities, no token budget
and no preemption. Its question is "which requests run next?". It is not asking "how many
tokens should the GPU compute this step?" — that is the key difference from vLLM (§18.4).

## 5. Batching

Static batching (Phase 9, `src/inference/batch.py`):

- **Prefill batch:** prompts are right-padded to `[B, T_max]`, with per-row `position_ids` and
  a combined causal + padding mask `[B,1,T,T]`.
- **Decode batch:** one token per row, `[B, 1]`, with each row's own position and a key mask
  `[B,1,1,L_max+1]`.

A batch runs until all of its members finish. Rows never share KV state: in batched decode,
`BatchedKVCache` pads and concatenates the rows' caches for that one forward pass only
(`kv_cache.py:234`), and the runner scatters each row's new K/V back to its own cache
(`model_runner.py:281-290`).

## 6. Continuous batching

`ContinuousBatchingEngine.step()` (`src/inference/continuous_batching.py:175`) is one serving
iteration:

```text
ADMIT    free slots = max_batch_size - |active|; admit waiting requests FIFO while the KV pool
         passes a one-step lookahead (reserve next-step blocks for every active request, then
         admit a newcomer only if prompt+1 tokens fit)
PREFILL  newcomers: one padded [B,T] forward
DECODE   requests that were already active: one [B',1] forward (a separate forward pass)
RETIRE   finished requests leave; their blocks return to the pool immediately
```

A finished request's slot is refilled on the next step. On mixed-length workloads this beats
static batching (Phase 11/12). Limitations (`continuous_batching.py:49-51`): no chunked prefill,
prefill and decode never share a forward pass, and there is no preemption, so a running request
that later cannot grow is aborted.

## 7. KV cache

There are three implementations with one interface: `get_seq_length / get / update / append`.

| class | file | how K/V grows | read for attention |
|---|---|---|---|
| `KVCache` (Phase 4) | `kv_cache.py:32` | `torch.cat([old, new])` every step | the tensor itself |
| `StaticKVCache` (Phase 15) | `kv_cache.py:123` | preallocated `[B,H,512,D]`, in-place `copy_` at the cursor | a view `[:, :, :len]` |
| `PagedKVCache` (Phase 10) | `kv_cache_manager.py:365` | scatter into pool blocks | **gather** blocks back into a contiguous tensor (`kv_cache_manager.py:129-145`) |

The paged cache is managed by `KVCacheManager` (`kv_cache_manager.py:153`):

- **Pool** (`KVBlockPool`): a fixed set of blocks, each `block_size`=16 tokens, stored per layer
  as `[num_blocks, H_kv, 16, D]`. The default pool holds 16 full-length sequences (512 blocks).
- **Operations:**
  - `allocate` takes `ceil(prompt/16)` blocks, all or nothing.
  - `grow` adds a block when a request's length crosses a block boundary.
  - `release` returns blocks to a free deque.
- **Addressing:** token `p` lives in block `block_ids[p // 16]` at offset `p % 16`.
- **Limitation:** there is no paged-attention kernel. Each step re-gathers each request's blocks
  into a contiguous tensor. The file's docstring notes this makes paged batched decode slower
  than contiguous (`kv_cache_manager.py:49-57`).
- **Not implemented:** prefix caching, block sharing, eviction and preemption. Running out of
  blocks aborts that request only.

Phase 14 measured the cost of `torch.cat` growth: generating 384 tokens after a 128-token prompt
copies **2.0 GB** in order to append **3 MB** of new K/V, so copy traffic grows quadratically
with length (`profiles/phase14/results.json` `kv_cat`).

## 8. Prefill

`ModelRunner.prefill` (`model_runner.py:141`) runs one forward pass over the whole prompt
`[1, N]`, fills a fresh cache with N tokens, and returns the last position's logits. The work is
dominated by matrix-matrix products: every Linear is `[N, K] × [K, N_out]`.

At N = 128 one forward is **7.80 GFLOP** of matmul, at 48 FLOP/byte for gate/up. That is above
this GPU's ridge point of 25.8 FLOP/byte, so prefill is compute-bound. Phase 13 measured
**87.4%** GPU utilization, with matmuls taking 72.5% of prefill GPU time
(`profiles/FINDINGS.md`, `profiles/phase14/findings.md`).

## 9. Decode

`ModelRunner.decode` (`model_runner.py:170`) takes one new token `[B, 1]`, runs it against the
cache, and appends one K/V row per layer. Every Linear becomes `[1, K] × [K, N]`, a
matrix-vector product:

- one forward is **61 MFLOP**, but it still reads all **116 MB** of weights
- arithmetic intensity is **0.5 FLOP/byte**
- so decode is **memory-bound**

And at this model size it is also **launch-bound**: 424 kernels per step, each costing more CPU
time to launch than it runs on the GPU. The result was 36.7% GPU utilization (Phase 13).
Section 19 has the numbers.

## 10. Sampling

`Sampler` (`src/inference/sampler.py:26`) is applied per request, in this order:

1. repetition penalty
2. no-repeat n-gram ban
3. greedy argmax, or: temperature → top-k → top-p → softmax → `multinomial`

`SamplingParams` (`request.py:50`) is a frozen dataclass, so each request carries its own and
the engine caches one `Sampler` per distinct parameter set. In batched modes every row is still
sampled by its own request's sampler, in a Python loop (`inference_engine.py:158-164`), and each
sampled token is read back with `.item()`, which forces a host sync on every token.

The sampler decides *which token*. It never decides *which work runs*; that is the scheduler's
job.

## 11. ModelRunner

`ModelRunner` (`src/inference/model_runner.py:96`) answers "how do I execute the model?" and
nothing more:

- `prefill`, `decode`
- `prefill_batch`: a padded forward, then per-row logits at `seq_len-1`, then per-row K/V scattered into each request's cache
- `decode_batch`: build `BatchedKVCache`, run the forward, scatter the new rows back
- the naive Phase 3 `generate`, which recomputes the whole sequence every step

With `kv_cache="static"` (Phase 15) the private caches it creates are `StaticKVCache`.
`InferenceEngine` (`inference_engine.py:98`) answers "how do I run a request?": it walks the
lifecycle, calls the KV manager, samples, and decides when a request stops.

## 12. PyTorch execution

All model code is eager PyTorch. Each Python-level op (`F.linear`, `x * cos`, `torch.cat`,
`F.scaled_dot_product_attention`, ...) goes through the dispatcher and usually launches one or
more CUDA kernels **asynchronously**: the CPU enqueues work and moves on, and the GPU executes
it later. Two consequences follow:

- If the CPU cannot enqueue kernels as fast as the GPU finishes them, the GPU idles. This is
  Phase 13's "launch/dispatch-bound".
- `.item()` / `.tolist()` block until the GPU catches up, so they mark a synchronization point
  per token.

`torch.compile` (Phase 15) traces the forward into a graph, and Inductor fuses chains of
elementwise ops into generated Triton kernels. Matmuls stay on cuBLAS.

## 13. ATen

ATen is PyTorch's C++ operator library. It is what the profiler shows as `aten::*`. Phase 14
traced `nn.Linear(512, 1408)` on a `[1,128,512]` input:

```text
nn.Linear.forward → F.linear → aten::linear   (composite: decomposes, launches nothing itself)
    ├─ aten::t          weight [1408,512] → view [512,1408], strides (1,512): free, no copy
    ├─ aten::matmul → aten::view [1,128,512] → [128,512]: free
    │               → aten::mm [128,512]×[512,1408]  ⇒  cuBLAS: ampere_sgemm_128x64_tn
    └─ aten::_unsafe_view [128,1408] → [1,128,1408]
```

Attention is `aten::scaled_dot_product_attention` → `aten::_efficient_attention_forward` →
one CUTLASS `fmha_cutlassF_f32_aligned_64x64_rf_sm80` kernel. RMSNorm and RoPE become several
elementwise ATen ops each (Phase 15 inventory):

- RMSNorm: 6 kernels
- RoPE: 8 kernels per call, one call each for q and k

## 14. CUDA

The CUDA backend of each ATen op picks and launches a kernel. The launches that matter here:

- `cudaLaunchKernel` for elementwise, reduction and copy kernels (at::native TensorIterator kernels)
- cuBLAS/cuBLASLt for `mm`
- CUTLASS for SDPA
- `cudaMemsetAsync` for cuBLAS workspaces

Kernels on one stream run in order. Measured on this machine (Phase 15, 15.14): each launch
costs **≈4.1 µs** of CPU, and even a 1-element kernel takes ≈1.1 µs on the GPU. Splitting the
same work into 1000 launches made it **60× slower** in wall time.

## 15. cuBLAS / cuBLASLt

cuBLAS is column-major, so PyTorch computes `C = x·Wᵀ` as `Cᵀ = W·xᵀ`. It passes the transposed
weight view with op `T` and the activations with op `N`, which gives the `_tn` suffix in kernel
names. cuBLASLt is the newer matmul API: it adds layouts, epilogues, workspaces and split-K
algorithms. Its reduction kernel is `cublasLt::splitKreduce_kernel`.

The library picks an algorithm from (M, N, K), dtype, layout, math mode and architecture
(Phase 14 M sweep, gate/up shape):

| M (rows) | kernel | GPU µs | TFLOPS |
|---|---|---|---|
| 1 | `internal::gemvx::kernel` (GEMV) | 27.7 | 0.05 |
| 2–16 | `gemmSN_TN_kernel` (small-N GEMM) | 29–56 | 0.1–0.4 |
| 32–64 | `ampere_sgemm_64x32_sliced1x4_tn`, `ampere_sgemm_128x32_tn` | 40–52 | 1.2–1.8 |
| 128–512 | `ampere_sgemm_128x64_tn` (+ split-K for some shapes) | 76–308 | 2.4–2.9 |

`ampere_sgemm_128x64_tn` breaks down as: Ampere, single-precision **S**GEMM on CUDA cores,
128×64 output tile per thread block, and T/N operand ops. Split-K runs when there are too few
output tiles to fill 20 SMs: the K dimension is split across thread blocks, and
`splitKreduce_kernel` sums the partial results. At M=128 that reduction took up to a third of
the q/out projection time (`profiles/phase14/findings.md` §8).

## 16. GPU kernels

A decode step of the baseline, by kernel category (Phase 15 ladder, GPU ms per step):

| gemv | attention | elementwise | copy/cat | reduction | index |
|---|---|---|---|---|---|
| 1.125 | 0.302 | 0.294 | 0.185 | 0.049 | 0.018 |

GEMV is where the GPU time goes. Elementwise and copy kernels are where the *launches* go:

- RoPE: 128 launches per step
- RMSNorm: 102
- masked SDPA (mask kernels + `fill_`): 72
- the `torch.cat` KV cache: 16

(`profiles/phase15/FINDINGS.md`, "Kernel launches".)

## 17. GPU hardware

The RTX 3050 6GB Laptop is Ampere, compute capability 8.6, with 20 SMs, a 1 MB L2 cache and
6 GB of GDDR6.

| measured (Phase 14) | value |
|---|---|
| peak FP32 GEMM (CUDA cores, FFMA) | 2.86 TFLOPS |
| peak TF32 GEMM (Tensor Cores, `cutlass_80_tensorop_s1688gemm`) | 5.97 TFLOPS |
| DRAM bandwidth | 111 GB/s |
| FP32 ridge point | 25.8 FLOP/byte |

The model runs in fp32 with `float32_matmul_precision="highest"`, so every matmul uses the
**CUDA cores**. Tensor Cores would need TF32/FP16/BF16. TF32 would speed up prefill GEMMs
1.5–2.2×, but it can't help the M=1 GEMVs, which are limited by bytes, not math.

---

## 18. vLLM comparison

The source studied is vLLM commit `8b84e15`, V1 engine. Paths below are relative to the vLLM
repo root.

### 18.1 The frozen reference

`profiles/phase16/engine_freeze.json` records "our engine — final version". It contains:

- model config, checkpoint and tokenizer, with sha256 hashes
- every engine component and its default settings (block size 16, a pool of 16 full
  sequences, FIFO scheduling, sampler defaults, fast-path flags)
- the runtime: Python, torch, CUDA, cuDNN, GPU, driver, matmul precision, SDPA backends
- a hash of every `src/` file
- the stored Phase 13–15 headline numbers

Phase 12 left no stored results (its `--output` is optional), so Phase 16 re-measures with the
same harness (`src/inference/benchmark.py`).

### 18.2 Architecture map

| our engine | vLLM V1 | why the component exists |
|---|---|---|
| `InferenceRequest` (`request.py:129`) | `Request` (`vllm/v1/request.py`), 12 `RequestStatus` values incl. `PREEMPTED` and the `FINISHED_*` reasons | one object owns a generation's tokens, parameters, progress and cache handle, so the engine can stay stateless |
| `Scheduler` + `ContinuousBatchingEngine.step` | `Scheduler.schedule` / `update_from_output` (`vllm/v1/core/sched/scheduler.py:557, 1967`) | decides what work the GPU runs this iteration |
| `Batch` (padded `[B,T]` / `[B,1]`) | `SchedulerOutput` → flattened token batch in `GPUModelRunner._prepare_inputs` (`vllm/v1/worker/gpu_model_runner.py:1951`) | turns a set of requests into one forward pass |
| prefill / decode (two code paths) | one path: every request gets `num_new_tokens` (prompt chunk or 1) | processes new tokens against cached ones |
| `KVCacheManager` + `PagedKVCache` | `KVCacheManager` → `KVCacheCoordinator` → `BlockPool` (`vllm/v1/core/*.py`) + worker `BlockTable` | variable-length sequences without fragmentation |
| continuous batching | the same scheduler with a token budget, chunked prefill and preemption | keeps the GPU busy as requests come and go |
| `ModelRunner` | `Worker` → `GPUModelRunner` + persistent `InputBatch` | turns scheduled work into tensor operations |
| `Sampler` (per request) | `Sampler` (`vllm/v1/sample/sampler.py`), batched on the GPU | turns logits into tokens |
| PyTorch eager (+ `torch.compile`) | `torch.compile` with a custom Inductor backend, piecewise compiled + CUDA graphs | runs the model's math |
| CUDA via ATen, cuBLAS, SDPA | the same, plus FlashAttention/FlashInfer/Triton paged kernels and vLLM custom ops | the kernels themselves |

### 18.3 One request through both engines: "Hello, my name is"

| step | our engine | vLLM V1 |
|---|---|---|
| entry | `engine.generate(prompt)` or `submit` | `LLM.generate` → `_add_request` (`vllm/entrypoints/offline_utils.py:560`) → `LLMEngine.add_request` (`vllm/v1/engine/llm_engine.py:222`) |
| tokenize | `tokenizer.encode` in `create_request` | renderer (`offline_utils.py:110-140`); `InputProcessor` validates the params and builds an `EngineCoreRequest` (`vllm/v1/engine/input_processor.py:476`) |
| hand-off | same process, same call stack | ZMQ to **EngineCore in its own process** by default (`core_client.py:124-130`; `VLLM_ENABLE_V1_MULTIPROCESSING=1`). The IO threads overlap serialization with the GPU forward (`vllm/v1/engine/core.py:1172-1176`) |
| queue | `Scheduler.add` → WAITING | `Scheduler.add_request` → `waiting` |
| schedule | admit if a slot is free and the prompt blocks fit | prefix-cache lookup, then `allocate_slots` for as many prompt tokens as the budget allows |
| KV allocation | `create_cache(prompt_len)`: `ceil(len/16)` blocks | `allocate_slots`: `cdiv(tokens, bs)` new blocks, reusing cached prefix blocks |
| prefill | padded `[B,T]` forward; K/V scattered into blocks | the prompt tokens share a flat forward with other requests' decode tokens |
| sample | `Sampler` per row; `.item()` | batched GPU sampler; one pinned D→H copy per step |
| decode loop | `grow` a block if needed → `[B,1]` forward → sample → `check_stop` | same request, next step: `num_new_tokens=1` |
| stop | `check_stop`: EOS, `max_new_tokens`, `max_seq_len` (`request.py:229`) | `check_stop` in the engine core: EOS, stop ids, length, repetition (`vllm/v1/core/sched/utils.py:98-140`); stop *strings* in the frontend's detokenizer, which then aborts the request in the core |
| finish | release blocks; FINISHED | `_free_request` frees blocks (they remain prefix-cached); the output processor returns a `RequestOutput` |

**Structurally the same:** request objects with explicit states, a scheduler loop, a
block-based KV allocator, a runner that executes scheduled work, and a separate sampler.

**Different in vLLM:** the frontend and the engine core are separate processes; the scheduler
reasons in tokens rather than requests; one forward pass per step; and requests can be
preempted instead of aborted.

### 18.4 The scheduler: "what token computation runs this iteration?"

vLLM's scheduler (`scheduler.py:559-568`) says it directly:

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the
> num_computed_tokens and num_tokens_with_spec … At each step, the scheduler tries to assign
> tokens to the requests so that each request's num_computed_tokens can catch up.

- **Two limits per step:**
  - `token_budget = max_num_batched_tokens` (default 2048) bounds the *compute*
    (`scheduler.py:577`, `vllm/config/scheduler.py:42-54`).
  - `max_num_seqs` (default 128) bounds the *concurrency*.
  - Ours only has `max_batch_size`, a seat limit. A seat limit alone lets one long prompt
    inflate the step's latency.
- **Order:** RUNNING requests first, then WAITING (`scheduler.py:624, 868`). Waiting requests
  are skipped entirely in any step that had to preempt (`:869`). Running requests already hold
  KV, so serving them first guarantees progress. Ours is the opposite: newcomers are
  prefilled first, then actives decode.
- **Output:**
  - `SchedulerOutput` is a flat `{req_id: num_tokens}` map, plus block ids and finished ids
    (`vllm/v1/core/sched/output.py:231-315`).
  - New requests are sent once in full (`NewRequestData`), and later only as diffs
    (`CachedRequestData`).
  - `num_computed_tokens` is advanced right after scheduling (`scheduler.py:1584-1597`), which
    lets step N+1 be planned while step N runs (async scheduling, on by default).
- **Preemption:**
  - If a running request cannot grow, the scheduler preempts `running[-1]` (the lowest
    priority under the priority policy) (`:759-806`).
  - The victim's blocks are freed, `num_computed_tokens=0`, status → PREEMPTED, and it goes to
    the *front* of `waiting` (`:1539-1582`).
  - This is recompute, not swap (there is no swap in `vllm/v1/core/`). It is cheap because the
    freed blocks usually stay in the prefix cache.
  - Ours aborts with `OUT_OF_KV_BLOCKS`.

### 18.5 Continuous batching, properly

Both engines re-decide batch membership every step, so finished requests leave and waiting
requests join immediately. The differences are in what a step *is*:

| | our engine | vLLM |
|---|---|---|
| admission | FIFO while slots + a one-step KV lookahead allow | FIFO/priority while the budget, `max_num_seqs` and `allocate_slots` allow; optional full-sequence reservation (`kv_cache_manager.py:515-531`) and watermark |
| removal | after `check_stop`, blocks released | the same, and the finished ids reach the workers through the next `SchedulerOutput` |
| composition | newcomers (prefill) and actives (decode), as two forwards | one flat token list mixing prefill chunks and decode tokens |
| prefill/decode interaction | a newcomer's whole prompt runs before this step's decodes | a prompt is chunked to the leftover budget, so decodes keep flowing |

### 18.6 Paged KV cache: logical → physical

```text
logical positions  0..15 | 16..31 | 32..47          request A, 40 tokens, block_size 16
logical blocks        0  |    1   |    2
block table (req A)  [ 7 ,    2   ,    9 ]           physical block ids in the pool
slot(p) = block_table[p // 16] * 16 + p % 16          ← vLLM's slot_mapping
```

- **The logical → physical map (both engines).** Ours keeps `block_ids` per request and
  addresses token `p` at `block_ids[p // 16]`, offset `p % 16` (`kv_cache_manager.py:39-42`).
  vLLM keeps `req_to_blocks` per request in the scheduler process
  (`single_type_kv_cache_manager.py:127-130`) and mirrors the ids into a worker-side
  `BlockTable`, a `[max_reqs, max_blocks]` int32 GPU tensor (`vllm/v1/worker/block_table.py:114-121`).
- **Writing new K/V.** A Triton kernel computes `slot_mapping` per token (`block_table.py:445-478`).
  `reshape_and_cache_flash` then scatters the new K/V to those slots
  (`vllm/v1/attention/backends/flash_attn.py:1510-1544`).
- **Reading K/V — the key difference.** vLLM's `flash_attn_varlen_func` takes
  `block_table=...` and reads the paged cache **in place** (`flash_attn.py:1453-1478`). Ours
  gathers each request's blocks into a contiguous tensor on every step, in every layer
  (`kv_cache_manager.py:129-145`). In batched decode it then pads and concatenates all rows
  (`kv_cache.py:268-295`).

### 18.7 Why paging matters, and what each engine does

The problem is that sequences have unknown, different, growing lengths. Contiguous
per-request buffers force a choice: reserve `max_len` for everyone, which wastes memory
(internal fragmentation), or reallocate and copy as they grow, which is O(L²) copies
(Phase 14: 2 GB for one 512-token generation) and fragments the heap. Fixed-size blocks make
every allocation the same size, so any free block fits any request. At most one partial block
per request is wasted, and a finished request's blocks are immediately reusable.

**What we implemented**

- a fixed pool of blocks
- per-request block lists
- all-or-nothing allocation
- growth by one block at a boundary
- release to a free list
- a lookahead reservation for admission
- OOM handling that affects only the one request

**What vLLM adds**

- **a paged-attention kernel**: no gather, no copy
- a pool sized from *profiled* free memory:
  - `gpu_memory_utilization` × total memory, minus peak activations and CUDA-graph memory
    (`vllm/v1/worker/gpu_worker.py:566-726`)
  - divided by the page size (`kv_cache_utils.py:1760-1763`)
- one int8 backing buffer with per-layer strided views
- ref-counted blocks, shared across requests
- an intrusive doubly-linked LRU free queue (`FreeKVCacheBlockQueue`, `kv_cache_utils.py:246`), so a cache hit can pull a block out of the middle in O(1)
- prefix caching
- recompute preemption
- a watermark
- hybrid managers for sliding-window and Mamba layers
- a CPU-only manager in the scheduler, with the worker receiving only ids

**What we simplified**

- a manually sized pool
- one attention type
- one block size, used both for bookkeeping and for storage

**What we did not implement**

- a paged-attention kernel
- sharing
- prefix caching
- eviction
- preemption
- swapping (vLLM V1 doesn't swap either)
- KV offload / transfer

### 18.8 Prefix caching

Once KV lives in blocks, a block of K/V is fully determined by the tokens *up to and including*
that block. vLLM hashes every **full** block as
`hash(parent_hash, block_token_ids, extra_keys)` (`kv_cache_utils.py:649-679`). The extra keys
are LoRA, multimodal inputs and `cache_salt`. Because the hash is chained, it identifies the
whole prefix.

- **Lookup.** On admission, `get_computed_blocks` walks a request's block hashes until the
  first miss (`kv_cache_manager.py:264-321`). It stops at `num_tokens-1`, because the last
  token must be recomputed to produce logits. The hit blocks are `touch`ed (`ref_cnt += 1`),
  which also removes them from the free queue. The request starts with `num_computed_tokens`
  already advanced, and only the uncached suffix costs budget.
- **Why only full blocks.** A partial block is still being written by its owner. Sharing it
  would need copy-on-write. Full blocks are immutable, so sharing them is pure ref-counting.
- **Freeing.** A freed block keeps its K/V and its hash until the block is actually reused
  (lazy eviction, `block_pool.py:723-744`). "Free" therefore also means "cached, LRU".

For two prompts "The capital of France is …", the second request skips the prefill of every
shared full block. Our Phase 10 manager already has everything this needs except the hash map
and ref-counts (§21 item 4).

### 18.9 ModelRunner → vLLM model execution

vLLM keeps request management (scheduler, CPU), execution planning (`SchedulerOutput`) and
model execution (worker, GPU) apart. Everything meets in `GPUModelRunner.execute_model`
(`gpu_model_runner.py:4149`):

| input | where it comes from |
|---|---|
| model weights | loaded once by the Worker; the model is compiled and CUDA-graph-captured at startup (`gpu_worker.py:806-855`) |
| KV cache | per-layer views into one preallocated buffer, bound to the attention layers |
| input tokens | `token_ids_cpu[req, pos]` gathered into one flat `[T]` tensor (`:1998-2011`) |
| positions | `num_computed_tokens[req] + query_pos` (`:1984-1987`) |
| attention metadata | `query_start_loc`, `seq_lens`, `block_table`, `slot_mapping` (`:2060-2066, 2179-2188`) |
| sampling information | `SamplingMetadata`: per-row tensors, rebuilt only when the batch changes (`gpu_input_batch.py:836-958`) |

- **Persistent state.** The `InputBatch` persists across steps and only deltas are applied
  (`_update_states`, `:1192-1562`). That keeps the per-step CPU cost proportional to what
  changed, which matters because decode is host-bound.
- **Logits.** They are computed only at `logits_indices = query_start_loc[1:] - 1`, the last
  scheduled token of each request (`:2231`). This is what our Phase 15 `last_token_logits`
  flag does for the unpadded case.

In one sentence: the scheduler determines *what* needs to be executed, and the model runner
turns that into tensor operations.

**Same checkpoint in both engines.** V1 is architecturally a Llama. `profiles/phase16/export_llama.py`
exports it as `LlamaForCausalLM`, permuting the q/k rows so that interleaved RoPE pairs become
half-split. q·k is invariant under a shared permutation, so the model is unchanged: max
|Δlogit| 1.7e-5, identical greedy tokens. vLLM therefore runs *our* model with its built-in
`LlamaForCausalLM` (`vllm/model_executor/models/llama.py`).

### 18.10 Prefill vs decode in production

We run prefill and decode as two separate forward passes. vLLM makes them the same operation
(§18.4), and that is what makes **mixed workloads** possible: one step can carry 50 decode
tokens and a 500-token prompt chunk. The `FlashAttention` backend reorders the batch so that
decodes come first (`vllm/v1/attention/backends/utils.py:878-891`). The varlen kernel handles
the prompt's q_len=500 and each decode's q_len=1 in one launch, via `cu_seqlens_q`.

The risk this manages is the one ours has. A large prompt's prefill occupies the GPU for that
whole step, so every running decode's inter-token latency jumps.

### 18.11 Chunked prefill

With `enable_chunked_prefill` (default on), a waiting request is scheduled for
`min(remaining_prompt, leftover_budget)` tokens (`scheduler.py:1128`). The rest continues in
later steps through the running loop. `long_prefill_token_threshold` caps any single request's
chunk so that one long prompt cannot starve the others (`:606-622`). It is disabled when
there is only one eligible request, since then "there is nobody to starve".

The trade-off:

- **Smaller budget or chunks:** steady inter-token latency for running requests, but a longer
  time-to-first-token for the long prompt and slightly less GEMM efficiency.
- **Larger budget or chunks:** the opposite.

In our Phase 14 terms, a chunk of ≥128 tokens is still well into the compute-bound GEMM regime
(`ampere_sgemm_128x64` at ~2.4 TFLOPS), so chunking costs little efficiency.

### 18.12 Sampling

Sampling has the same pipeline in both engines: logits → per-request parameters → penalties →
temperature → top-k/top-p filtering → distribution → token. vLLM differs in how it runs it:

- **One batched GPU call** per step, over `[num_reqs, vocab]` (`sampler.py:21-59`).
- **Per-row parameter tensors** (`SamplingMetadata`).
- **Greedy and random rows** are computed together and merged with `torch.where`.
- **No `torch.multinomial`:** it uses exponential-noise `argmax(probs/q)`, or FlashInfer
  rejection sampling, which avoids a CPU sync (`ops/topk_topp_sampler.py:544-605`).
- **Penalties** run as a CUDA op, and are skipped when nobody uses them.
- **A single pinned device→host copy** of the sampled ids per step (`gpu_model_runner.py:7456-7469`).

Ours loops over requests in Python and calls `.item()` for each one.

The separation is identical in both: the scheduler decides *what work executes*, and the
sampler decides *which token this request generates*.

### 18.13 Abstraction boundaries

| responsibility | same concept | different implementation | missing in ours | production complexity in vLLM |
|---|---|---|---|---|
| request | state machine, per-request params, stop rules | vLLM tracks `num_computed_tokens`, block hashes and async placeholders | PREEMPTED, blocked states, stop strings | multi-process request ids, n>1 fan-out, streaming |
| scheduling | iteration-level, FIFO | tokens vs seats | token budget, priorities, preemption | spec decode, structured output, KV connectors, async scheduling |
| batching | one forward per group | flat unpadded tokens vs padded `[B,T]` | mixed prefill+decode | persistent `InputBatch`, reordering, cascade attention |
| continuous batching | admit/retire every step | budget-driven | chunked prefill | fairness thresholds, full-sequence reservation, watermark |
| KV cache | fixed-size blocks, block tables | in-place kernel reads vs gather | sharing, prefix cache, LRU, preemption | memory profiling, hybrid layouts, offload, KV transfer |
| prefill / decode | prompt pass, then 1-token steps | one unified path | chunking | varlen kernels, CUDA-graph dispatch per batch shape |
| model execution | runner builds tensors, calls forward | fixed buffers, compiled model | CUDA graphs, fixed addresses | TP/PP, piecewise compile, custom fusion passes |
| sampling | same pipeline | batched GPU vs per-row Python | logprobs, seeds, min-p, stop strings | logits processors, rejection sampling, spec-decode verification |
| GPU execution | PyTorch → ATen → cuBLAS/SDPA | eager vs compiled + graphs | FlashAttention, fp16/bf16/fp8 | kernel libraries per hardware (CUDA, ROCm, TPU, CPU) |

### 18.14 What we don't have, and why production is much bigger

- **Memory management:** profiled pool sizing, prefix caching, sharing, LRU eviction,
  preemption, CPU/remote KV offload.
- **Kernels:** paged FlashAttention/FlashInfer, fused RMSNorm+residual, RoPE+KV-write fusion
  (`vllm/compilation/passes/pass_manager.py:141-230`), quantized GEMMs, CUDA graphs.
- **Distributed:** tensor, pipeline, data and expert parallelism, with the KV configuration
  kept consistent across ranks.
- **Quantization:** fp8, int8, int4 weights and KV.
- **Scheduling:** token budgets, chunked prefill, priorities, spec decode, structured output,
  disaggregated prefill.
- **Serving:** an OpenAI-compatible API, streaming, detokenization, metrics, admission control,
  fault tolerance.
- **Models:** hundreds of architectures, multimodal inputs, LoRA; hardware backends beyond
  NVIDIA.

Each item exists because production traffic breaks one of our assumptions: one GPU, one
model, short contexts, FIFO fairness, a few requests at a time.

### 18.15 All the way down

```text
REQUEST ─► vLLM / our engine
             ├─ Scheduler (what tokens run)      ├─ KV manager (where their K/V lives)
             └──────────────┬────────────────────┘
                     Model Runner (flat tokens | padded batch)
                            ▼
                PyTorch (compiled + CUDA graph | eager / torch.compile)
                            ▼
                ATen (aten::mm, attention custom op | aten::_efficient_attention_forward)
                            ▼
                CUDA backend
          ┌─────────────────┴──────────────────┐
     cuBLAS/cuBLASLt                   attention kernels
     GEMM (prefill) / GEMV (decode)    flash_attn_varlen + block_table | CUTLASS fmha fp32
          └─────────────────┬──────────────────┘
                     CUDA kernels (graph replay | ~130-424 launches per step)
                            ▼
                GPU SMs: CUDA cores (FP32) | Tensor Cores (TF32/FP16/BF16/FP8)
```

### 18.16 Source study: one request through the components

| component (vLLM file) | problem it solves | data it owns | called by → calls | GPU |
|---|---|---|---|---|
| `LLM` (`entrypoints/llm.py`) | synchronous offline API | request counter, renderer | user → `LLMEngine.add_request` / `step` loop | none |
| `LLMEngine` (`v1/engine/llm_engine.py`) | frontend loop: input → core → output | processors, core client | `LLM` → InputProcessor, EngineCoreClient, OutputProcessor | none |
| `InputProcessor` (`v1/engine/input_processor.py`) | validate, build `EngineCoreRequest` | configs, renderer | LLMEngine → `SamplingParams.verify` | none |
| `EngineCoreClient` (`v1/engine/core_client.py`) | hide in-process vs separate process | ZMQ sockets, output queue thread | LLMEngine → EngineCoreProc over ZMQ | none |
| `EngineCore` (`v1/engine/core.py`) | the busy loop: schedule → execute → update | scheduler, executor, batch queue | client → Scheduler, Executor | indirect |
| `Scheduler` (`v1/core/sched/scheduler.py`) | tokens to compute per request this step | `requests`, `waiting`, `running` | EngineCore → KVCacheManager | none (CPU) |
| `KVCacheManager` / `BlockPool` (`v1/core/`) | block allocation, prefix cache, eviction | block metadata, hash map, LRU queue | Scheduler → Coordinator → BlockPool | none: block ids index the GPU buffer |
| `Worker` (`v1/worker/gpu_worker.py`) | one GPU: memory profiling, KV allocation, compile/capture | device, model runner | Executor (RPC) → GPUModelRunner | yes |
| `GPUModelRunner` (`v1/worker/gpu_model_runner.py`) | `SchedulerOutput` → flat tensors → forward → sample | `InputBatch`, persistent buffers, KV views, sampler | Worker → attention builders, CUDA graph dispatcher, model, Sampler | yes: the only layer that launches work |
| `BlockTable` (`v1/worker/block_table.py`) | `[req, logical block] → physical id`, `slot_mapping` | CPU/GPU buffers | runner → Triton slot-mapping kernel | H→D copy per step |
| FlashAttention backend (`v1/attention/backends/flash_attn.py`) | varlen paged attention; KV write | graph-safe metadata buffers | model's `Attention` layer → `reshape_and_cache_flash`, `flash_attn_varlen_func` | yes |
| `Sampler` (`v1/sample/sampler.py`) | batched token selection + logprobs | none (all parameters in `SamplingMetadata`) | runner → penalties, top-k/top-p ops | yes |
| `OutputProcessor` (`v1/engine/output_processor.py`) | detokenize, stop strings, `RequestOutput` | per-request detokenizer state | LLMEngine → detokenizer | none |

### 18.17 Experiment: our engine vs vLLM on the same workload

`profiles/phase16/engine_vs_vllm.py`, results in `profiles/phase16/COMPARISON.md`. The setup is
identical on both sides:

- same weights
- same prompt token ids
- 64 new tokens, greedy, fp32, EOS id 3
- same GPU

| scenario | engine | TTFT ms | ITL ms | tokens/s | peak MB |
|---|---|---|---|---|---|
| single (128-token prompt) | ours, baseline | 5.1 | 3.18 | 311 | 141 |
| single | ours, Phase 15 optimized | 4.3 | 1.88 | 522 | 136 |
| batch (16 requests, 32–125-token prompts) | ours, baseline | 54.4 | 17.74 | 867 | 296 |
| batch | ours, Phase 15 optimized | 43.9 | 17.25 | 898 | 296 |

Both configurations produce identical tokens. The **vLLM row** comes from
`--engine vllm` run in a separate vLLM environment; vLLM pins its own PyTorch/CUDA, so it
can't share this environment. `COMPARISON.md` merges it once it has been run.

What the numbers already show:

1. **Single request.** Phase 15's launch reductions give 1.68× tokens/s. Decode is
   launch-bound, and that is exactly what compile and fusion attack. vLLM's CUDA-graph replay
   for uniform decode batches (`vllm/v1/cudagraph_dispatcher.py:233-322`, `FULL_AND_PIECEWISE`
   by default) goes one step further and turns ~130 launches into one graph launch.
2. **Batch.** Phase 15 changes almost nothing here (1.04×), because the batched path's cost
   isn't launches. It is **KV data movement**:
   - every step, every layer gathers each request's paged blocks back into contiguous K/V;
   - it then pads and concatenates all 16 rows into `[16,H,L_max,D]`;
   - it pads prefill prompts to the longest one;
   - and `torch.compile` graph-breaks on `position_ids.max().item()`.

   vLLM's design removes each of these:
   - the paged kernel reads blocks in place;
   - the flat unpadded batch removes the padding;
   - fixed buffers and CUDA graphs remove the per-launch cost.

   This is the mechanism to look for in the vLLM row: its advantage should be **larger in
   the batch scenario than in the single one**.
3. **Memory.** Our peak is the working set, 136–296 MB. vLLM *reserves*
   `gpu_memory_utilization` × 6 GB up front and fills it with KV blocks, so its "peak" is a
   capacity decision, not a cost.

### 18.18 One token, forwards and backwards

**Forwards:**

1. user → `InferenceRequest`
2. scheduler admits it
3. batch assembled
4. KV blocks allocated
5. decode chosen
6. `ModelRunner` builds `[B,1]`
7. `F.linear` → `aten::mm` → cuBLAS `gemv2T_kernel_val` (`[1,512]×[512,15485]` for the LM head)
8. SDPA → `fmha_cutlassF`
9. SMs stream 116 MB of weights
10. logits `[B, 15485]`
11. sampler → next token
12. K/V appended to the request's block
13. scheduler, next iteration

**Backwards:**

1. The profiler says the GEMV kernel is expensive (57.7% of decode GPU time).
2. Why? Decode multiplies `[1,K] × [K,N]`: 0.5 FLOP/byte, memory-bound.
3. Why M=1? One token per request per step.
4. Why not increase M? Batch requests together: M=B. Phase 15 measured 9.3× aggregate
   throughput at B=16.
5. What makes batching possible? The scheduler plus continuous batching, admitting and
   retiring requests every step.
6. What makes variable-length requests batchable? KV-cache management: blocks, so every
   request can grow independently.
7. What makes the blocks cheap to attend over? A paged-attention kernel. That is the piece we
   stopped short of, and it is the reason vLLM exists in the form it does.

---

## 19. Performance findings

**Phase 13 — where time goes** (`profiles/FINDINGS.md`):

| step | wall | GPU busy | GPU util | kernels/step | largest cost |
|---|---|---|---|---|---|
| prefill (128 tokens) | 7.13 ms | 6.23 ms | 87.4% | 403 | GEMM 72.5% |
| decode (1 token, 128 cached) | 5.84 ms | 2.14 ms | 36.7% | 424 | GEMV 57.7% |

Per token, prefill is 105× cheaper than decode.

**Phase 14 — why** (`profiles/phase14/findings.md`):

- **Prefill is compute-bound.** Matmul intensity is ~48 FLOP/byte, above the ridge point.
- **Decode is memory-bound.** At 0.5 FLOP/byte, the 116 MB of weights take ~1.1 ms per token
  at full bandwidth.
- **Decode is also launch-bound.** Each launch takes ~14 µs of wall time on average, while the
  average kernel runs ~5 µs.
- **Batching is the lever.** In isolation, 32 separate `[1,K]` calls vs one `[32,K]` call is
  ~21× faster.

**Phase 15 — what helped** (`profiles/phase15/FINDINGS.md`). Paired A/B timing was used,
because laptop clocks drift ~40% during a run:

| change | Δ decode | decision |
|---|---|---|
| preallocated KV (no `cat`) | −0.2% | within noise at 128 tokens of context |
| RoPE tables pre-interleaved | +10.0% | kept |
| fused RMSNorm | +14.5% | kept |
| no decode mask | +17.7% | kept |
| SDPA `enable_gqa` | −24.9% | rejected: fp32 falls back to the math backend, adding launches |
| fused q/k/v, fused gate/up | +2.2%, +0.2% | within noise |
| last-token LM head | +7.6% prefill | kept |
| `torch.compile` | +26.6% | kept |

- **Final configuration:** decode **2.05×** (3.92 → 1.91 ms/token), prefill 1.22×, launches
  424 → 132, GPU utilization 50.9% → 72.9%.
- **Correctness:** the gate passed with max |Δlogit| 3.05e-5, and tokens identical to the
  baseline.
- **Batching the decode (B=1→16):** aggregate throughput 262 → 2436 tokens/s (9.3×), while each
  request's latency grew 1.7×.

**Phase 16 — end-to-end through the engine** (`profiles/phase16/COMPARISON.md`; 64 new
tokens, greedy, fp32):

| scenario | tokens/s, baseline → Phase 15 optimized | speedup |
|---|---|---|
| single request | 311 → 522 | 1.68× |
| 16 concurrent requests | 867 → 898 | 1.04× |

Tokens are identical in every case. The batched path is dominated by KV data movement, not
launches (§18.17), which is why the launch-focused Phase 15 changes barely move it.

## 20. Limitations

- **One model, one GPU, fp32.** No tensor or pipeline parallelism, no quantization, no
  FP16/BF16 inference path.
- **Attention.** Eager SDPA with explicit masks, plus `repeat_interleave` for GQA. There is no
  paged-attention kernel: the paged cache is gathered back into contiguous K/V every step, and
  batched decode pads and concatenates every row's cache every step.
- **Scheduling.**
  - FIFO only.
  - No token budget and no chunked prefill, so one long prompt delays every decode that step.
  - No preemption: running out of KV aborts the request.
  - No priorities, no fairness.
  - Prefill and decode are separate forward passes.
- **KV cache.** No prefix caching, no block sharing, no eviction. The pool is sized by hand,
  not from measured free memory.
- **Execution.**
  - One Python process, synchronous.
  - `.item()` after every token.
  - No CUDA graphs.
  - `torch.compile` graph-breaks on `position_ids.max().item()` in the batched path
    (`model.py:123`).
- **Sampling.** A per-request Python loop, not batched on the GPU. No logprobs, beam search,
  min-p, stop strings or seeded sampling.
- **Serving.** No HTTP/OpenAI API, no streaming, no detokenization pipeline, no metrics or
  admission control.

## 21. Future work

In the order the measurements suggest:

1. **CUDA graphs for decode.** Decode is launch-bound, and graph replay would make a step
   roughly one launch. This needs static shapes: full-capacity attention with a length mask
   over `StaticKVCache`, and no `.item()` inside the forward.
2. **A paged-attention decode kernel** (Triton) that reads K/V through a block table and
   removes the per-step gather/cat. That is the step from "paged bookkeeping" to "paged
   attention".
3. **A flattened, unpadded token batch.** Mix prefill chunks and decode tokens in one forward,
   plus a token budget and chunked prefill in the scheduler (vLLM's V1 design, §18.4).
4. **Prefix caching:** hash full blocks, reference-count shared blocks, and make the free list
   LRU.
5. **Preemption** by recompute, instead of aborting requests when KV runs out.
6. **Lower precision:** BF16 weights halve the bytes every decode step has to stream, and TF32
   or BF16 uses the Tensor Cores for prefill.
7. **Batched GPU sampling**, and removing the per-token host sync.
