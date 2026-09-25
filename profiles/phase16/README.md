# Phase 16 — Deep mapping against vLLM

The final phase is mostly **study**, not new engine code. We freeze the engine we built,
map every component onto vLLM's V1 engine by reading vLLM's source, and run both engines
on the same workload. The goal is to explain *which mechanisms* cause the difference, not
just how big it is.

The write-up is **`Docs/FINAL_ARCHITECTURE.md`**. It covers the whole system from request
to GPU and back, the component-by-component vLLM mapping with source references, the
performance findings from Phases 12–16, the limitations and future work.

## Files

| file | purpose |
|---|---|
| `export_llama.py` | exports the V1 checkpoint as a standard `LlamaForCausalLM` (`models/v1-llama`) and checks that it reproduces our logits and greedy tokens |
| `engine_vs_vllm.py` | freezes our engine (16.1), writes the shared workload, benchmarks our engine (baseline and Phase 15 optimized) and, in a vLLM environment, vLLM (16.17) |
| `workload.json` | shared prompts (token ids), `max_new_tokens`, EOS id, batch size |
| `engine_freeze.json` | the frozen "our engine — final version" record: model, checkpoint and tokenizer hashes, components, settings, runtime, source hashes, stored Phase 13–15 results |
| `results_ours.json` / `results_vllm.json` | metrics per engine and scenario (TTFT, ITL, tokens/s, peak memory) plus the generated tokens |
| `COMPARISON.md` | generated comparison table, and the mechanisms behind each difference |

## Why the model runs in vLLM unchanged

V1 is a Llama-family decoder: GQA, RoPE, pre-RMSNorm, SwiGLU, no biases, tied embeddings.
The one difference is RoPE layout. V1 rotates interleaved pairs `(2i, 2i+1)`, while
Llama rotates halves `(i, i+D/2)`. `export_llama.py` reorders each head's q/k projection
rows so the interleaved pairs become halves. Because q·k is invariant under a shared
permutation, the exported model computes the same function. Measured result: max
|Δlogit| 1.7e-5 and identical greedy tokens. Our BPE tokenizer isn't an HF tokenizer, so
vLLM gets token ids directly (`skip_tokenizer_init=True`, `TokensPrompt`).

## Run

```bash
python profiles/phase16/export_llama.py           # once: models/v1-llama (~120 MB)
python profiles/phase16/engine_vs_vllm.py         # freeze + our engine + COMPARISON.md

# vLLM pins its own PyTorch/CUDA build, so it lives in a separate environment:
python -m venv .venv-vllm && . .venv-vllm/bin/activate && pip install vllm
python profiles/phase16/engine_vs_vllm.py --engine vllm       # add --vllm-eager to disable CUDA graphs
python profiles/phase16/engine_vs_vllm.py --report            # merge into COMPARISON.md
```

On a 6 GB GPU, keep `--vllm-gpu-memory-utilization` around 0.5. vLLM preallocates that
fraction for weights + KV cache, so its "peak MB" is a reservation rather than a working
set. Running with `--vllm-eager` next to the default run separates the effect of CUDA
graphs from the rest of vLLM.

The vLLM source studied is commit `8b84e15` (2026-09-24) of
https://github.com/vllm-project/vllm (V1 engine: `vllm/v1/...`).
