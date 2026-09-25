"""
Phase 13 — profiling instrumentation tests.

Checks the observation tooling itself, so the findings it produces
can be trusted:

    - profiler regions are OFF by default, cost nothing, and never
      change model outputs (on vs off gives identical logits)
    - when enabled, every labelled section of the model and engine
      appears in a torch.profiler trace, with sensible call counts
      (one "layer_i" per layer per forward, 2 rmsnorms per layer + 1)
    - KV-cache regions show up for the path actually used:
      kv_cache/cat for contiguous caches, kv_cache/paged_write/read for
      paged ones, batch_gather/scatter for batched decode
    - kernel names are bucketed into the right categories
    - profile_workload keeps setup() out of the profile, writes trace
      files, and its per-step accounting is consistent

Every profiler here is started with profiler_activities(): a CPU-only
session would stop later CUDA sessions in this process from seeing any
kernels (a Kineto limitation).

Runs on CPU with a tiny model (no CUDA kernels there, so GPU totals
are zero); the CUDA test at the bottom checks kernel collection.
"""

import json

import pytest
import torch

from configs.v1 import ModelConfig
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache import KVCache
from src.inference.model_runner import ModelRunner
from src.inference.profiler import categorize_kernel, profile_workload, profiler_activities
from src.inference.sampler import Sampler
from src.model import profiling
from src.model.model import V1LanguageModel


class IdTokenizer:
    def encode(self, text):
        return [int(t) for t in text.split()]

    def decode(self, token_ids):
        return " ".join(str(i) for i in token_ids)

    def token_to_id(self, token):
        return None


def make_model(num_layers=2, device="cpu"):
    torch.manual_seed(0)

    config = ModelConfig(
        vocab_size=64,
        max_seq_len=64,
        hidden_dim=32,
        num_layers=num_layers,
        num_q_heads=4,
        num_kv_heads=2,
        ffn_dim=64,
    )

    model = V1LanguageModel(config).to(device)
    model.eval()

    return model


def region_names(prof):
    return {
        e.key: e.count
        for e in prof.key_averages()
        if getattr(e, "is_user_annotation", False)
    }


# --------------------------------------------------
# Switch
# --------------------------------------------------


def test_regions_off_by_default_and_noop():
    assert not profiling.is_enabled()

    ctx_a = profiling.region("anything")
    ctx_b = profiling.region("else")
    assert ctx_a is ctx_b                      # shared no-op, no allocation

    with profiling.enabled():
        assert profiling.is_enabled()
        assert profiling.region("x") is not ctx_a

    assert not profiling.is_enabled()


def test_regions_do_not_change_outputs():
    model = make_model()
    ids = torch.randint(0, 64, (2, 9))

    with torch.no_grad():
        off, _ = model(ids)
        with profiling.enabled():
            on, _ = model(ids)

    assert torch.equal(off, on)


def test_enabled_restores_previous_state_on_error():
    with pytest.raises(RuntimeError):
        with profiling.enabled():
            raise RuntimeError("boom")

    assert not profiling.is_enabled()


def test_no_regions_recorded_when_disabled():
    model = make_model()
    ids = torch.randint(0, 64, (1, 5))

    with torch.profiler.profile(activities=profiler_activities()) as prof:
        with torch.no_grad():
            model(ids)

    assert "attention/sdpa" not in region_names(prof)


# --------------------------------------------------
# Labels appear with the right structure
# --------------------------------------------------


def test_model_regions_and_counts():
    num_layers = 3
    model = make_model(num_layers=num_layers)
    ids = torch.randint(0, 64, (1, 6))

    with profiling.enabled():
        with torch.profiler.profile(activities=profiler_activities()) as prof:
            with torch.no_grad():
                model(ids)

    names = region_names(prof)

    for i in range(num_layers):
        assert names[f"layer_{i}"] == 1

    for leaf in ("attention/qkv_proj", "attention/rope", "attention/gqa_repeat",
                 "attention/mask", "attention/sdpa", "attention/out_proj",
                 "mlp/gate_up_proj", "mlp/act_mul", "mlp/down_proj", "attention", "mlp"):
        assert names[leaf] == num_layers, leaf

    assert names["rmsnorm"] == 2 * num_layers + 1
    assert names["embedding"] == 1
    assert names["lm_head"] == 1
    assert "attention/kv_cache" not in names   # no cache passed


def test_contiguous_kv_cache_regions():
    model = make_model()
    runner = ModelRunner(model, IdTokenizer(), Sampler(greedy=True), device="cpu")
    ids = torch.randint(0, 64, (1, 5))

    with profiling.enabled():
        with torch.profiler.profile(activities=profiler_activities()) as prof:
            out = runner.prefill(ids)
            runner.decode(out.logits.argmax(-1, keepdim=True), out.kv_cache)

    names = region_names(prof)

    assert names["attention/kv_cache"] == 2 * 2      # 2 forwards x 2 layers
    assert names["kv_cache/cat"] == 2                # only decode grows by cat
    assert "kv_cache/paged_write" not in names


def test_paged_and_batched_kv_regions():
    model = make_model()
    tokenizer = IdTokenizer()
    sampler = Sampler(greedy=True)
    runner = ModelRunner(model, tokenizer, sampler, device="cpu")
    engine = InferenceEngine(runner, tokenizer, sampler, device="cpu")

    requests = [engine.create_request(p, max_new_tokens=5) for p in ("1 2 3", "4 5 6 7 8")]

    with profiling.enabled():
        with torch.profiler.profile(activities=profiler_activities()) as prof:
            engine.prefill_batch(requests)
            engine.decode_batch(requests)

    names = region_names(prof)

    for name in ("kv_cache/paged_write", "kv_cache/paged_read", "kv_cache/batch_gather",
                 "kv_cache/scatter", "batch/build", "sampling"):
        assert name in names, name

    assert names["sampling"] == 4                    # 2 requests x (prefill + decode)
    assert names["batch/build"] == 2


def test_naive_generate_labels_sampling():
    model = make_model()
    runner = ModelRunner(model, IdTokenizer(), Sampler(greedy=True), device="cpu")

    with profiling.enabled():
        with torch.profiler.profile(activities=profiler_activities()) as prof:
            runner.generate("1 2 3", max_new_tokens=4)

    assert region_names(prof)["sampling"] == 4


# --------------------------------------------------
# Kernel categories
# --------------------------------------------------


@pytest.mark.parametrize("name, category", [
    ("ampere_sgemm_128x64_tn", "gemm"),
    ("void cublasLt::splitKreduce_kernel<32, 16, int, float>", "gemm"),
    ("void gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false, cublasGemvTensorStridedBatched>", "gemm"),
    ("std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float>", "gemv"),
    ("void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16>", "gemv"),
    ("fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel)", "attention"),
    ("flash_fwd_kernel<Flash_fwd_kernel_traits>", "attention"),
    ("void at::native::(anonymous namespace)::CatArrayBatchedCopy<float>", "copy/cat"),
    ("void at::native::index_elementwise_kernel<128, 4>", "index/gather"),
    ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float>>", "reduction"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>>", "elementwise"),
    ("some_custom_kernel", "other"),
])
def test_categorize_kernel(name, category):
    assert categorize_kernel(name) == category


# --------------------------------------------------
# profile_workload
# --------------------------------------------------


def test_profile_workload_setup_is_excluded_and_files_written(tmp_path):
    model = make_model()
    runner = ModelRunner(model, IdTokenizer(), Sampler(greedy=True), device="cpu")
    ids = torch.randint(0, 64, (1, 6))

    setups = []

    @torch.inference_mode()
    def setup():
        setups.append(1)
        out = runner.prefill(ids)
        return out.kv_cache, out.logits.argmax(-1, keepdim=True)

    @torch.inference_mode()
    def fn(state):
        cache, token = state
        for _ in range(3):
            token = runner.decode(token, cache).logits.argmax(-1, keepdim=True)

    result = profile_workload("decode", fn, "cpu", steps=3, setup=setup,
                              out_dir=tmp_path / "decode", warmup=1, timing_repeats=2)

    # warm-up + 2 timed + 1 profiled, each with its own fresh setup
    assert len(setups) == 4

    # The profiled run saw 3 decode forwards and no prefill: each
    # decode grows the cache by cat, prefill never does.
    assert result.region("layer_0").calls == 3
    assert result.region("kv_cache/cat").calls == 3 * model.config.num_layers

    assert result.steps == 3
    assert result.wall_ms > 0
    assert result.wall_ms_per_step == pytest.approx(result.wall_ms / 3)
    assert result.gpu_busy_ms == 0 and result.kernels == []   # CPU only

    trace = json.loads((tmp_path / "decode" / "trace.json").read_text())
    assert any(e.get("name") == "attention/sdpa" for e in trace["traceEvents"])
    assert (tmp_path / "decode" / "ops.txt").read_text()
    assert (tmp_path / "decode" / "shapes.txt").read_text()

    assert any(op.name == "aten::mm" or op.name == "aten::linear" for op in result.ops)
    assert not profiling.is_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profile_workload_collects_cuda_kernels(tmp_path):
    model = make_model(device="cuda")
    ids = torch.randint(0, 64, (1, 16), device="cuda")

    @torch.inference_mode()
    def fn():
        model(ids, kv_cache=KVCache(model.config.num_layers))

    result = profile_workload("prefill", fn, "cuda", out_dir=tmp_path / "p")

    assert result.kernel_launches > 0
    assert result.gpu_busy_ms > 0
    assert 0 < result.gpu_utilization <= 1.5
    assert result.region("attention/sdpa").gpu_ms > 0
    assert "gemm" in result.categories() or "gemv" in result.categories()

    labelled = sum(result.region(n).gpu_ms for n in (
        "embedding", "rmsnorm", "attention/qkv_proj", "attention/rope", "attention/kv_cache",
        "attention/gqa_repeat", "attention/mask", "attention/sdpa", "attention/out_proj",
        "mlp/gate_up_proj", "mlp/act_mul", "mlp/down_proj", "lm_head"))
    # Leaf regions account for (almost) every kernel.
    assert labelled == pytest.approx(result.gpu_busy_ms, rel=0.1)
