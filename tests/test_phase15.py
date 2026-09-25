"""
Phase 15 — optimization correctness tests.

Every Phase 15 change is opt-in; these tests pin down that each one
keeps the model's outputs and that the tooling that judges them works:

    StaticKVCache      same interface as KVCache, writes in place,
                       bit-identical logits, overflow is an error,
                       works with ModelRunner / InferenceEngine /
                       batched decode (BatchedKVCache)
    fast_paths         all flags off by default; enabled() scopes and
                       restores them; rope_cache / decode_no_mask are
                       exact, the rest match within fp32 noise;
                       prepare() fuses weights as views (no copy);
                       last_token_logits only applies when safe
    correctness gate   passes an equivalent candidate, fails a broken
                       one (logits and tokens)

Runs on CPU with a tiny random model.
"""

import pytest
import torch

from configs.v1 import ModelConfig
from src.inference.continuous_batching import ContinuousBatchingEngine
from src.inference.correctness import Tolerance, Variant, run_gate
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache import KVCache, StaticKVCache
from src.inference.kv_cache_manager import KVCacheManager
from src.inference.model_runner import ModelRunner
from src.inference.sampler import Sampler
from src.model import fast_paths
from src.model.model import V1LanguageModel


class IdTokenizer:
    def encode(self, text):
        return [int(t) for t in text.split()]

    def decode(self, token_ids):
        return " ".join(str(i) for i in token_ids)

    def token_to_id(self, token):
        return None


def make_model(seed=0):
    torch.manual_seed(seed)
    config = ModelConfig(vocab_size=64, max_seq_len=64, hidden_dim=32, num_layers=2,
                         num_q_heads=4, num_kv_heads=2, ffn_dim=64)
    return V1LanguageModel(config).eval()


def make_runner(model, kv_cache="dynamic"):
    return ModelRunner(model, IdTokenizer(), Sampler(greedy=True), device="cpu", kv_cache=kv_cache)


@torch.inference_mode()
def prefill_decode_logits(model, cache, prompt_len=10, steps=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 64, (2, prompt_len), generator=g)
    new = torch.randint(0, 64, (2, steps), generator=g)
    logits, _ = model(ids, kv_cache=cache)
    out = [logits[:, -1]]
    for i in range(steps):
        logits, _ = model(new[:, i:i + 1], kv_cache=cache)
        out.append(logits[:, -1])
    return torch.stack(out)


# ======================================================================
# StaticKVCache
# ======================================================================


def test_static_cache_writes_in_place_and_returns_views():
    cache = StaticKVCache(num_layers=1, capacity=8)
    k0, v0 = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)

    k, v = cache.update(0, k0, v0)
    buffer = cache.key_cache[0]
    assert buffer.shape == (1, 2, 8, 4)
    assert cache.get_seq_length() == 3
    assert torch.equal(k, k0) and torch.equal(v, v0)
    assert k.data_ptr() == buffer.data_ptr()          # a view, not a copy

    k1 = torch.randn(1, 2, 1, 4)
    k, _ = cache.update(0, k1, k1)
    assert cache.key_cache[0] is buffer               # no reallocation
    assert torch.equal(k, torch.cat([k0, k1], dim=2))
    assert cache.get(0)[0].shape[2] == 4


def test_static_cache_overflow_raises():
    cache = StaticKVCache(num_layers=1, capacity=4)
    cache.update(0, torch.randn(1, 1, 4, 2), torch.randn(1, 1, 4, 2))
    with pytest.raises(ValueError, match="overflow"):
        cache.update(0, torch.randn(1, 1, 1, 2), torch.randn(1, 1, 1, 2))


def test_static_cache_logits_identical_to_dynamic():
    model = make_model()
    dynamic = prefill_decode_logits(model, KVCache(2))
    static = prefill_decode_logits(model, StaticKVCache(2, capacity=64))
    assert torch.equal(dynamic, static)


def test_static_cache_from_tensors_copies():
    k = torch.randn(1, 2, 5, 4)
    cache = StaticKVCache.from_tensors([k], [k.clone()], capacity=16)
    assert cache.get_seq_length() == 5
    k.zero_()
    assert cache.get(0)[0].abs().sum() > 0


def test_runner_and_engine_use_configured_cache_kind():
    model = make_model()
    with pytest.raises(ValueError):
        make_runner(model, kv_cache="paged")

    runner = make_runner(model, kv_cache="static")
    out = runner.prefill(torch.tensor([[1, 2, 3]]))
    assert isinstance(out.kv_cache, StaticKVCache)
    assert out.kv_cache.capacity == model.config.max_seq_len

    engine = InferenceEngine(runner, IdTokenizer(), Sampler(greedy=True), device="cpu",
                             paged_kv=False)
    result = engine.generate_from_ids([1, 2, 3, 4], max_new_tokens=5)
    assert isinstance(result["kv_cache"], StaticKVCache)

    reference = InferenceEngine(make_runner(model), IdTokenizer(), Sampler(greedy=True),
                                device="cpu", paged_kv=False)
    assert result["token_ids"] == reference.generate_from_ids([1, 2, 3, 4], max_new_tokens=5)["token_ids"]


def test_static_cache_through_continuous_batching():
    model = make_model()
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11]]

    def run(kind):
        runner = make_runner(model, kv_cache=kind)
        engine = ContinuousBatchingEngine(
            runner, IdTokenizer(), Sampler(greedy=True), device="cpu", paged_kv=False,
            kv_cache_manager=KVCacheManager.for_model(model, num_blocks=1, device="cpu"),
            max_batch_size=3,
        )
        for i, p in enumerate(prompts):
            engine.submit(engine.create_request_from_ids(p, max_new_tokens=6, request_id=f"r{i}"))
        return {r["request"].request_id: r["token_ids"] for r in engine.run_until_complete()}

    assert run("static") == run("dynamic")


# ======================================================================
# fast_paths
# ======================================================================


def test_flags_default_off_and_scoped():
    assert fast_paths.FLAGS.active() == []
    with fast_paths.enabled(rope_cache=True):
        assert fast_paths.FLAGS.active() == ["rope_cache"]
        with fast_paths.enabled(fused_qkv=True):
            assert fast_paths.FLAGS.active() == ["fused_qkv"]
        assert fast_paths.FLAGS.active() == ["rope_cache"]
    assert fast_paths.FLAGS.active() == []
    with pytest.raises(ValueError):
        fast_paths.set_flags(warp_speed=True)


@pytest.mark.parametrize("flag", fast_paths.NAMES)
def test_each_flag_matches_baseline(flag):
    model = fast_paths.prepare(make_model())
    reference = prefill_decode_logits(model, KVCache(2))
    with fast_paths.enabled(**{flag: True}):
        candidate = prefill_decode_logits(model, KVCache(2))

    if flag in ("rope_cache", "decode_no_mask"):
        assert torch.equal(reference, candidate)
    else:
        assert torch.allclose(reference, candidate, atol=1e-5, rtol=0)


def test_all_flags_with_static_cache_match_baseline():
    model = fast_paths.prepare(make_model())
    reference = prefill_decode_logits(model, KVCache(2))
    with fast_paths.enabled(**{f: True for f in fast_paths.NAMES}):
        candidate = prefill_decode_logits(model, StaticKVCache(2, capacity=64))
    assert torch.allclose(reference, candidate, atol=1e-5, rtol=0)


def test_prepare_fuses_weights_as_views():
    model = make_model()
    before = [p.clone() for p in model.parameters()]
    fast_paths.prepare(model)
    fast_paths.prepare(model)                        # idempotent

    attn = model.blocks[0].attention
    fused = attn.qkv_weight
    assert fused.shape[0] == attn.q_proj.weight.shape[0] + 2 * attn.k_proj.weight.shape[0]
    assert attn.q_proj.weight.data_ptr() == fused.data_ptr()
    assert attn.k_proj.weight.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
    assert "blocks.0.attention.qkv_weight" not in model.state_dict()   # non-persistent

    for a, b in zip(before, model.parameters()):
        assert torch.equal(a, b)

    # an in-place weight update is seen by the fused path too
    with torch.no_grad():
        attn.v_proj.weight.add_(1.0)
    assert torch.equal(fused[-attn.v_proj.weight.shape[0]:], attn.v_proj.weight)


def test_last_token_logits_only_when_safe():
    model = make_model()
    ids = torch.randint(0, 64, (2, 7))
    with torch.no_grad(), fast_paths.enabled(last_token_logits=True):
        logits, _ = model(ids)
        assert logits.shape == (2, 1, 64)

        full, loss = model(ids, targets=ids)
        assert full.shape == (2, 7, 64) and loss is not None

        mask = torch.ones(2, 1, 7, 7, dtype=torch.bool).tril()
        masked, _ = model(ids, attention_mask=mask)
        assert masked.shape == (2, 7, 64)


# ======================================================================
# Correctness gate
# ======================================================================


PROMPTS = [[1, 2, 3], [5, 6, 7, 8, 9, 10, 11], [20, 21, 22, 23, 24]]


def test_gate_passes_equivalent_candidate():
    model = fast_paths.prepare(make_model())
    reference = Variant("baseline", make_runner(model))
    candidate = Variant("fast", make_runner(model, kv_cache="static"),
                        {f: True for f in fast_paths.NAMES})

    gate = run_gate(reference, candidate, PROMPTS, IdTokenizer(), max_new_tokens=8)
    assert gate.passed, [c for c in gate.failures()]
    assert {c.name for c in gate.checks} == {
        "prefill_logits", "decode_logits", "greedy_tokens", "eos", "kv_cache",
        "batched", "batched_independent",
    }
    assert gate.max_abs_error < 1e-5


def test_gate_fails_broken_candidate():
    reference = Variant("baseline", make_runner(make_model(seed=0)))
    broken = Variant("other weights", make_runner(make_model(seed=1)))

    gate = run_gate(reference, broken, PROMPTS, IdTokenizer(), max_new_tokens=8,
                    tolerance=Tolerance(max_abs=1e-4, mean_abs=1e-5))
    assert not gate.passed
    failed = {c.name for c in gate.failures()}
    assert {"prefill_logits", "decode_logits"} <= failed
