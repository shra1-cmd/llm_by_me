"""
Phase 10 — KV-cache manager tests.

What this file checks, in three groups:

1. Logical block bookkeeping (KVCacheManager with no physical pool,
   so no model or torch tensors involved):
       empty pool, allocation sizes, exact block boundaries, many
       requests without overlap, release, reuse, out-of-memory that
       changes nothing, growth 10 -> 20 -> 40 tokens, isolation.

2. Physical storage (KVBlockPool + PagedKVCache handle):
       K/V written into non-contiguous blocks reads back exactly the
       same as a plain contiguous KVCache; reused blocks never leak
       the previous owner's values.

3. Engine integration (tiny real model):
       requests get blocks at prefill, grow during decode, release on
       finish; batched and sequential generation still match the
       Phase 3 naive runner; OOM aborts only the request that didn't
       fit; admission control keeps FIFO and doesn't deadlock.
"""

import pytest
import torch

from configs.v1 import ModelConfig
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache import KVCache
from src.inference.kv_cache_manager import (
    KVCacheManager,
    KVCacheOutOfMemory,
    PagedKVCache,
)
from src.inference.model_runner import ModelRunner
from src.inference.request import FinishReason, RequestStatus
from src.inference.sampler import Sampler
from src.model.model import V1LanguageModel


# ======================================================================
# 1. Logical bookkeeping
# ======================================================================


def test_empty_manager():
    manager = KVCacheManager(num_blocks=8, block_size=16)

    assert manager.num_free_blocks == 8
    assert manager.num_allocated_blocks == 0
    assert manager.free_block_ids == list(range(8))
    assert manager.request_ids == []


def test_allocate_blocks():
    manager = KVCacheManager(num_blocks=8, block_size=16)

    blocks = manager.allocate("A", 20)

    assert len(blocks) == 2
    assert manager.get_block_ids("A") == blocks
    assert manager.get_num_tokens("A") == 20
    assert manager.num_free_blocks == 6
    assert all(manager.owner_of(b) == "A" for b in blocks)


@pytest.mark.parametrize(
    "num_tokens, expected_blocks",
    [(0, 0), (1, 1), (15, 1), (16, 1), (17, 2), (32, 2), (33, 3)],
)
def test_exact_block_boundaries(num_tokens, expected_blocks):
    manager = KVCacheManager(num_blocks=8, block_size=16)

    assert manager.blocks_needed(num_tokens) == expected_blocks
    assert len(manager.allocate("A", num_tokens)) == expected_blocks


def test_multiple_requests_do_not_overlap():
    manager = KVCacheManager(num_blocks=16, block_size=16)

    a = manager.allocate("A", 40)
    b = manager.allocate("B", 20)
    c = manager.allocate("C", 50)

    assert (len(a), len(b), len(c)) == (3, 2, 4)
    assert not (set(a) & set(b))
    assert not (set(a) & set(c))
    assert not (set(b) & set(c))
    assert manager.num_free_blocks == 16 - 9


def test_release_returns_blocks():
    manager = KVCacheManager(num_blocks=8, block_size=16)

    a = manager.allocate("A", 40)
    manager.allocate("B", 10)

    released = manager.release("A")

    assert released == a
    assert not manager.has("A")
    assert manager.num_free_blocks == 8 - 1
    assert all(manager.owner_of(block) is None for block in a)
    assert set(a) <= set(manager.free_block_ids)

    with pytest.raises(KeyError):
        manager.release("A")


def test_released_blocks_are_reused():
    manager = KVCacheManager(num_blocks=4, block_size=16)

    a = manager.allocate("A", 32)
    manager.allocate("B", 32)

    assert manager.num_free_blocks == 0

    manager.release("A")

    d = manager.allocate("D", 20)

    assert sorted(d) == sorted(a)
    assert all(manager.owner_of(block) == "D" for block in d)


def test_out_of_memory_fails_cleanly():
    manager = KVCacheManager(num_blocks=4, block_size=16)

    b_blocks = manager.allocate("B", 40)          # 3 blocks

    with pytest.raises(KVCacheOutOfMemory):
        manager.allocate("A", 20)                  # needs 2, only 1 free

    # Nothing changed: A has nothing, B untouched, free pool intact.
    assert not manager.has("A")
    assert manager.get_block_ids("B") == b_blocks
    assert manager.num_free_blocks == 1

    # A new request that still fits is unaffected by the failed one.
    assert len(manager.allocate("C", 10)) == 1
    assert manager.num_free_blocks == 0


def test_grow_out_of_memory_is_atomic():
    manager = KVCacheManager(num_blocks=4, block_size=16)

    manager.allocate("A", 16)       # 1 block
    manager.allocate("B", 32)       # 2 blocks  -> 1 free

    with pytest.raises(KVCacheOutOfMemory):
        manager.grow("A", 48)       # needs 2 more, only 1 free

    assert len(manager.get_block_ids("A")) == 1
    assert manager.get_num_tokens("A") == 16
    assert manager.num_free_blocks == 1

    # Fits exactly: uses the last free block.
    assert len(manager.grow("A", 32)) == 1
    assert manager.num_free_blocks == 0
    assert manager.can_allocate(32, request_id="A")
    assert not manager.can_allocate(33, request_id="A")


def test_growth_10_20_40():
    manager = KVCacheManager(num_blocks=8, block_size=16)

    first = manager.allocate("A", 10)
    assert len(first) == 1

    added = manager.grow("A", 20)
    assert len(added) == 1
    assert manager.get_block_ids("A") == first + added

    added_again = manager.grow("A", 40)
    assert len(added_again) == 1
    assert len(manager.get_block_ids("A")) == 3
    assert manager.get_num_tokens("A") == 40

    # Growing to a size it already covers allocates nothing.
    assert manager.grow("A", 41) == []
    assert manager.grow("A", 5) == []
    assert manager.get_num_tokens("A") == 41


def test_request_isolation():
    manager = KVCacheManager(num_blocks=12, block_size=4)

    manager.allocate("A", 5)
    manager.allocate("B", 9)
    manager.grow("A", 13)
    manager.allocate("C", 3)
    manager.grow("B", 16)

    owned = {rid: set(manager.get_block_ids(rid)) for rid in ("A", "B", "C")}

    assert not (owned["A"] & owned["B"])
    assert not (owned["A"] & owned["C"])
    assert not (owned["B"] & owned["C"])

    for rid, blocks in owned.items():
        for block in blocks:
            assert manager.owner_of(block) == rid

    # Owned + free is exactly the whole pool, with no duplicates.
    everything = [b for blocks in owned.values() for b in blocks] + manager.free_block_ids
    assert sorted(everything) == list(range(12))


def test_allocate_twice_rejected_and_bad_args():
    manager = KVCacheManager(num_blocks=4, block_size=16)
    manager.allocate("A", 10)

    with pytest.raises(ValueError):
        manager.allocate("A", 10)

    with pytest.raises(KeyError):
        manager.grow("unknown", 10)

    with pytest.raises(ValueError):
        KVCacheManager(num_blocks=0)

    with pytest.raises(RuntimeError):
        manager.create_cache("B", 4)   # no physical pool


# ======================================================================
# 2. Physical storage through PagedKVCache
# ======================================================================

LAYERS, HEADS, DIM = 2, 2, 8


def make_physical_manager(num_blocks=8, block_size=4):
    return KVCacheManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=LAYERS,
        num_kv_heads=HEADS,
        head_dim=DIM,
    )


def random_kv(T, seed):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, HEADS, T, DIM, generator=g),
        torch.randn(1, HEADS, T, DIM, generator=g),
    )


def test_paged_cache_matches_contiguous_cache():
    manager = make_physical_manager(block_size=4)

    # Take a block first so A's blocks are not [0, 1, 2, ...].
    manager.allocate("other", 4)

    paged = manager.create_cache("A", 0)
    plain = KVCache(num_layers=LAYERS)

    # Prefill of 6 tokens, then 5 single-token decodes: crosses block
    # boundaries at 4 and 8.
    steps = [6, 1, 1, 1, 1, 1]

    for i, T in enumerate(steps):
        for layer in range(LAYERS):
            k, v = random_kv(T, seed=100 * i + layer)

            pk, pv = paged.update(layer, k, v)
            ck, cv = plain.update(layer, k, v)

            assert torch.equal(pk, ck)
            assert torch.equal(pv, cv)

    assert paged.get_seq_length() == plain.get_seq_length() == 11
    assert len(paged.block_ids) == 3
    assert "other" not in paged.block_ids
    assert manager.get_block_ids("other")[0] not in paged.block_ids

    for layer in range(LAYERS):
        assert torch.equal(paged.get(layer)[0], plain.get(layer)[0])


def test_reused_blocks_do_not_leak_previous_values():
    manager = make_physical_manager(num_blocks=2, block_size=4)

    a = manager.create_cache("A", 0)
    for layer in range(LAYERS):
        a.update(layer, torch.full((1, HEADS, 8, DIM), 7.0), torch.full((1, HEADS, 8, DIM), 7.0))

    manager.release("A")
    assert a.is_released

    b = manager.create_cache("B", 0)
    k, v = random_kv(3, seed=1)

    for layer in range(LAYERS):
        bk, _ = b.update(layer, k, v)
        assert bk.shape[2] == 3
        assert torch.equal(bk, k)

    with pytest.raises(RuntimeError):
        a.get(0)


def test_paged_cache_rejects_batched_writes():
    manager = make_physical_manager()
    cache = manager.create_cache("A", 0)

    with pytest.raises(ValueError):
        cache.update(0, torch.zeros(2, HEADS, 1, DIM), torch.zeros(2, HEADS, 1, DIM))


# ======================================================================
# 3. Engine integration
# ======================================================================


class IdTokenizer:
    """Space-separated integer ids; `<eos>` maps to a configurable id."""

    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id

    def encode(self, text):
        return [int(t) for t in text.split()]

    def decode(self, token_ids):
        return " ".join(str(i) for i in token_ids)

    def token_to_id(self, token):
        return self.eos_token_id if token == "<eos>" else None


def make_model(seed=0, device="cpu"):
    torch.manual_seed(seed)

    config = ModelConfig(
        vocab_size=64,
        max_seq_len=64,
        hidden_dim=32,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        ffn_dim=64,
    )

    model = V1LanguageModel(config).to(device)
    model.eval()

    return model


def make_engine(model, num_blocks=None, block_size=4, device="cpu"):
    tokenizer = IdTokenizer()
    sampler = Sampler(greedy=True)

    runner = ModelRunner(model, tokenizer, sampler, device=device)
    manager = KVCacheManager.for_model(
        model, num_blocks=num_blocks, block_size=block_size, device=device
    )
    engine = InferenceEngine(
        runner, tokenizer, sampler, device=device, kv_cache_manager=manager
    )

    return runner, engine, manager


def prompt_of(length, offset=0):
    return " ".join(str((offset + 7 * i) % 64) for i in range(length))


PROMPTS = [prompt_of(5, 1), prompt_of(8, 2), prompt_of(12, 3)]


def test_request_gets_paged_cache_and_grows_by_blocks():
    model = make_model()
    _, engine, manager = make_engine(model, block_size=4)

    request = engine.create_request(prompt_of(6), max_new_tokens=8)

    engine.prefill(request)

    assert isinstance(request.kv_cache, PagedKVCache)
    assert request.num_cached_tokens == 6
    assert len(request.block_ids) == 2                # ceil(6 / 4)

    block_counts = [len(request.block_ids)]

    while request.status == RequestStatus.DECODING:
        engine.decode(request)
        if not request.is_finished:
            block_counts.append(len(request.block_ids))
            assert len(request.block_ids) == manager.blocks_needed(request.num_cached_tokens)

    # Released on finish; final length still readable for stats.
    assert request.block_ids == []
    assert not manager.has(request.request_id)
    assert manager.num_free_blocks == manager.num_blocks
    assert request.num_cached_tokens == 6 + 8 - 1
    assert block_counts == sorted(block_counts)
    assert block_counts[-1] > block_counts[0]


def test_generation_matches_naive_with_paged_cache():
    model = make_model()
    runner, engine, manager = make_engine(model, block_size=4)

    for prompt in PROMPTS:
        naive = runner.generate(prompt, max_new_tokens=20)
        paged = engine.generate(prompt, max_new_tokens=20)

        assert paged["token_ids"] == naive["token_ids"]
        assert paged["stats"]["kv_blocks_peak"] == manager.blocks_needed(
            len(naive["token_ids"]) - 1
        )

    assert manager.num_free_blocks == manager.num_blocks


def test_batch_requests_hold_disjoint_blocks():
    model = make_model()
    _, engine, manager = make_engine(model, block_size=4)

    requests = [engine.create_request(p, max_new_tokens=10) for p in PROMPTS]

    engine.prefill_batch(requests)

    block_sets = [set(r.block_ids) for r in requests]
    assert [len(b) for b in block_sets] == [2, 2, 3]   # 5, 8, 12 tokens
    assert not (block_sets[0] & block_sets[1] | block_sets[0] & block_sets[2] | block_sets[1] & block_sets[2])

    for _ in range(4):
        engine.decode_batch([r for r in requests if not r.is_finished])

        live = [set(r.block_ids) for r in requests]
        for i in range(3):
            for j in range(i + 1, 3):
                assert not (live[i] & live[j])


def test_batched_generation_matches_naive_and_releases_everything():
    model = make_model()
    runner, engine, manager = make_engine(model, block_size=4)

    requests = [engine.submit(engine.create_request(p, max_new_tokens=15)) for p in PROMPTS]
    results = engine.run_until_complete(max_batch_size=3)

    for prompt, result in zip(PROMPTS, results):
        assert result["token_ids"] == runner.generate(prompt, max_new_tokens=15)["token_ids"]

    assert all(r.block_ids == [] for r in requests)
    assert manager.num_free_blocks == manager.num_blocks
    assert manager.request_ids == []


def test_blocks_are_reused_across_batches():
    model = make_model()
    runner, engine, manager = make_engine(model, num_blocks=12, block_size=4)

    prompts = [prompt_of(6, i) for i in range(6)]

    for p in prompts:
        engine.submit(engine.create_request(p, max_new_tokens=6))

    # 12 blocks can't hold all 6 requests at once, so later batches
    # must run on blocks freed by earlier ones.
    results = engine.run_until_complete(max_batch_size=3)

    assert len(results) == 6
    assert all(r["request"].finish_reason == FinishReason.MAX_NEW_TOKENS for r in results)

    for p, result in zip(prompts, results):
        assert result["token_ids"] == runner.generate(p, max_new_tokens=6)["token_ids"]

    assert manager.num_free_blocks == 12


def test_decode_oom_aborts_only_that_request():
    """
    Pool of 5 blocks x 4 tokens. A (12 tokens) and B (5 tokens) both
    fit at prefill (3 + 2 blocks), but there is no spare block for
    either to grow into. Whichever needs a new block first is aborted
    with OUT_OF_KV_BLOCKS; once it releases, the other can continue
    and must still produce exactly the naive tokens.
    """

    model = make_model()
    runner, engine, manager = make_engine(model, num_blocks=5, block_size=4)

    prompt_a = prompt_of(12, 3)
    prompt_b = prompt_of(5, 1)

    a = engine.submit(engine.create_request(prompt_a, max_new_tokens=10))
    b = engine.submit(engine.create_request(prompt_b, max_new_tokens=10))

    engine.run_until_complete(max_batch_size=2)

    reasons = {a.finish_reason, b.finish_reason}
    assert FinishReason.OUT_OF_KV_BLOCKS in reasons

    survivor, prompt = (b, prompt_b) if a.finish_reason == FinishReason.OUT_OF_KV_BLOCKS else (a, prompt_a)
    victim = a if survivor is b else b

    assert victim.status == RequestStatus.ABORTED
    assert survivor.status == RequestStatus.FINISHED
    assert survivor.all_tokens == runner.generate(prompt, max_new_tokens=10)["token_ids"]

    assert manager.num_free_blocks == 5


def test_prompt_larger_than_pool_is_aborted_not_deadlocked():
    model = make_model()
    runner, engine, manager = make_engine(model, num_blocks=2, block_size=4)

    too_big = engine.submit(engine.create_request(prompt_of(12), max_new_tokens=3))
    fits = engine.submit(engine.create_request(prompt_of(3), max_new_tokens=3))

    results = engine.run_until_complete(max_batch_size=2)

    assert [r["request"] for r in results] == [too_big, fits]
    assert too_big.finish_reason == FinishReason.OUT_OF_KV_BLOCKS
    assert too_big.kv_cache is None
    assert fits.all_tokens == runner.generate(prompt_of(3), max_new_tokens=3)["token_ids"]
    assert manager.num_free_blocks == 2


def test_admission_is_fifo_and_bounded_by_free_blocks():
    model = make_model()
    _, engine, manager = make_engine(model, num_blocks=6, block_size=4)

    # Each prompt needs 2 blocks: only 3 fit in one batch.
    requests = [engine.submit(engine.create_request(prompt_of(8, i), max_new_tokens=1)) for i in range(5)]

    first = engine.step_batch(max_batch_size=5)
    assert [r["request"] for r in first] == requests[:3]

    second = engine.step_batch(max_batch_size=5)
    assert [r["request"] for r in second] == requests[3:]


def test_failed_execution_releases_blocks():
    model = make_model()
    _, engine, manager = make_engine(model, block_size=4)

    request = engine.submit(engine.create_request(prompt_of(6), max_new_tokens=5))

    original_decode = engine.runner.decode

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    engine.runner.decode = boom

    with pytest.raises(RuntimeError):
        engine.step()

    engine.runner.decode = original_decode

    assert request.status == RequestStatus.ABORTED
    assert manager.num_free_blocks == manager.num_blocks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_paged_cache_on_cuda_matches_naive():
    model = make_model(device="cuda")
    runner, engine, manager = make_engine(model, block_size=4, device="cuda")

    assert manager.pool.k[0].device.type == "cuda"

    for prompt in PROMPTS:
        naive = runner.generate(prompt, max_new_tokens=15)
        assert engine.generate(prompt, max_new_tokens=15)["token_ids"] == naive["token_ids"]

    requests = [engine.submit(engine.create_request(p, max_new_tokens=15)) for p in PROMPTS]
    results = engine.run_until_complete(max_batch_size=3)

    for prompt, result in zip(PROMPTS, results):
        assert result["token_ids"] == runner.generate(prompt, max_new_tokens=15)["token_ids"]

    assert manager.num_free_blocks == manager.num_blocks
