"""
Phase 14 — GEMM investigation tooling tests.

Checks the pieces of profiles/phase14/gemm_benchmark.py that the
findings rely on, so its numbers can be trusted:

    - FLOPs / bytes / arithmetic intensity follow 2*M*K*N and
      4*(MK + KN + MN)
    - the model's matmul table matches the real V1 model: every
      nn.Linear shape, 57 matmuls per forward, tied LM head
    - output-tile (CTA) counts are parsed from cuBLAS kernel names
      using cuBLAS's column-major C^T[N, M] view
    - kernel names are shortened correctly (templates, CUTLASS wrappers)
    - nn.Linear on a 3-D input reaches the backend as t -> view -> mm
      -> _unsafe_view, with the weight passed as a transposed view

Runs on CPU; the CUDA test at the bottom checks that an M=1 Linear
really dispatches to a gemv-type kernel and M=128 to a GEMM.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from configs.v1 import ModelConfig
from src.model.model import V1LanguageModel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "gemm_benchmark", ROOT / "profiles" / "phase14" / "gemm_benchmark.py"
)
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)


def test_gemm_cost_prefill_vs_decode():
    prefill = gb.gemm_cost(128, 512, 1408)
    decode = gb.gemm_cost(1, 512, 1408)

    assert prefill["flops"] == 2 * 128 * 512 * 1408
    assert decode["flops"] == 2 * 512 * 1408
    assert prefill["bytes"] == 4 * (128 * 512 + 512 * 1408 + 128 * 1408)
    assert prefill["weight_bytes"] == decode["weight_bytes"] == 4 * 512 * 1408

    # M=1: every weight used once -> ~0.5 FLOP/byte (2 FLOPs per 4-byte weight)
    assert decode["arithmetic_intensity"] == pytest.approx(0.5, abs=0.01)
    assert prefill["arithmetic_intensity"] > 40


def test_linear_shapes_match_model():
    config = ModelConfig(vocab_size=1000)
    model = V1LanguageModel(config)
    shapes = {s["name"]: s for s in gb.linear_shapes(config)}

    block = model.blocks[0]
    expected = {
        "q_proj": block.attention.q_proj,
        "k_proj/v_proj": block.attention.k_proj,
        "out_proj": block.attention.out_proj,
        "gate_proj/up_proj": block.ffn.gate_proj,
        "down_proj": block.ffn.down_proj,
    }
    for name, linear in expected.items():
        N, K = linear.weight.shape
        assert (shapes[name]["K"], shapes[name]["N"]) == (K, N), name

    assert tuple(block.attention.v_proj.weight.shape) == tuple(block.attention.k_proj.weight.shape)
    assert tuple(block.ffn.up_proj.weight.shape) == tuple(block.ffn.gate_proj.weight.shape)

    V, H = model.token_embedding.weight.shape
    assert (shapes["lm_head"]["K"], shapes["lm_head"]["N"]) == (H, V)

    n_linears = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert sum(s["count"] for s in shapes.values()) == n_linears + 1 == 57


def test_output_tiles_uses_column_major_view():
    # 128 along N, 64 along M
    assert gb.output_tiles("ampere_sgemm_128x64_tn", M=128, N=1408) == 11 * 2
    assert gb.output_tiles("ampere_sgemm_32x32_sliced1x4_tn", M=128, N=128) == 4 * 4
    assert gb.output_tiles("internal::gemvx::kernel", M=1, N=1408) is None


def test_short_kernel():
    assert gb.short_kernel("void gemv2T_kernel_val<int, int, float>(params)") == "gemv2T_kernel_val"
    assert gb.short_kernel(
        "std::enable_if<!(false), void>::type internal::gemvx::kernel<int, float>(x)"
    ) == "internal::gemvx::kernel"
    assert gb.short_kernel(
        "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(P::Params)"
    ) == "cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4"
    assert gb.short_kernel("ampere_sgemm_128x64_tn") == "ampere_sgemm_128x64_tn"


def test_linear_trace_reaches_mm_with_transposed_weight():
    trace = gb.trace_linear(K=64, N=96, M=8, device="cpu")
    ops = [c["op"] for c in trace["dispatch_ops"]]

    assert ops == ["t", "view", "mm", "_unsafe_view"]

    mm = trace["dispatch_ops"][ops.index("mm")]
    x2d, w_t = mm["inputs"]
    assert x2d["shape"] == [8, 64] and x2d["contiguous"]
    # weight [96, 64] row-major, passed as a [64, 96] view with swapped strides
    assert w_t["shape"] == [64, 96] and w_t["stride"] == [1, 64]
    assert not w_t["contiguous"]
    assert mm["output"]["shape"] == [8, 96]

    assert trace["profiler_tree"][0]["name"] == "aten::linear"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_decode_uses_gemv_and_prefill_uses_gemm():
    decode = gb.measure_linear(1, 512, 1408, iters=5)
    prefill = gb.measure_linear(128, 512, 1408, iters=5)

    assert "gemv" in decode["main_kernel"].lower()
    assert "gemm" in prefill["main_kernel"].lower()
    assert prefill["tflops"] > decode["tflops"]
    assert decode["kernel_us"] > 0 and prefill["kernel_count"] >= 1
