"""
Phase 10 — KV-cache manager, end-to-end on the real checkpoint.

What this script shows:

    1. A small block pool (default 20 blocks x 16 tokens) shared by
       every request, instead of one private KVCache per request.
    2. For each FIFO batch: which blocks each request got at prefill
       (ALLOCATE), how many it held at its peak (GROW), and that the
       free pool is full again after the batch (RELEASE).
    3. Later batches running on block ids freed by earlier ones
       (REUSE) — the default pool is too small to hold every request
       at once, so reuse is required (and checked) in that case.
    4. Correctness: every request's tokens are identical to the
       Phase 3 naive runner (greedy).
    5. An out-of-memory demo: a request whose prompt can't fit in a
       tiny pool is aborted with out_of_kv_blocks while the next
       request still runs normally.

The lifecycle is driven step by step here (scheduler.next_batch ->
engine.prefill_batch -> engine.decode_batch ...) only so block tables
can be printed mid-flight; engine.step_batch does exactly the same.

Usage:
    PYTHONPATH="$(pwd)" python scripts/run_kv_cache_manager.py
    PYTHONPATH="$(pwd)" python scripts/run_kv_cache_manager.py --num-blocks 32 --block-size 8
"""

import argparse
import time

import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.inference.inference_engine import InferenceEngine
from src.inference.kv_cache_manager import KVCacheManager
from src.inference.model_runner import ModelRunner
from src.inference.request import FinishReason, RequestStatus, SamplingParams
from src.tokenizer.tokenizer import BPETokenizer

DEFAULT_PROMPTS = [
    "Hello, my name is",
    "Once upon a time",
    "The little dog",
    "One day, a girl named Sue went to the park and",
    "Tom had a red ball",
    "The sun was shining and the birds were singing. Lily wanted to",
    "Mom said",
    "There was a big tree in the garden",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--prompts", type=str, nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def fmt_blocks(block_ids, limit=8):
    shown = ", ".join(str(b) for b in block_ids[:limit])
    return f"[{shown}{', ...' if len(block_ids) > limit else ''}]"


def run_batches(engine, manager, batch_size):
    """Drive FIFO batches manually, printing the block lifecycle."""

    scheduler = engine.scheduler
    used_by_earlier_batches: set[int] = set()
    reused_any = False
    batch_no = 0
    finished = []

    while scheduler.has_waiting():
        requests = scheduler.next_batch(batch_size, can_schedule=engine.admission_check())

        if not requests:
            requests = scheduler.next_batch(1)

        batch_no += 1
        print(f"\n  Batch {batch_no}: {len(requests)} request(s), "
              f"free blocks before = {manager.num_free_blocks}/{manager.num_blocks}")

        engine.prefill_batch(requests)

        # Every block each request held at any point in this batch.
        held = {r.request_id: list(r.block_ids) for r in requests}

        for request in requests:
            blocks = request.block_ids
            print(f"    {request.request_id}  prompt={request.prompt_len:>2} tok  "
                  f"ALLOCATE {len(blocks)} block(s) {fmt_blocks(blocks)}")

        while True:
            active = [r for r in requests if r.status == RequestStatus.DECODING]

            if not active:
                break

            engine.decode_batch(active)

            for request in active:
                for block in request.block_ids:
                    if block not in held[request.request_id]:
                        held[request.request_id].append(block)

        batch_blocks = set()

        for request in requests:
            scheduler.complete(request)
            finished.append(request)

            blocks = held[request.request_id]
            reused = sorted(set(blocks) & used_by_earlier_batches)
            reused_any |= bool(reused)
            batch_blocks |= set(blocks)

            print(f"    {request.request_id}  {request.finish_reason.value:<15} "
                  f"kv_len={request.num_cached_tokens:>3}  "
                  f"GROW -> {len(blocks)} block(s) {fmt_blocks(blocks)}"
                  f"{'  reused ' + str(len(reused)) if reused else ''}  "
                  f"RELEASE -> {request.block_ids}")

        used_by_earlier_batches |= batch_blocks

        print(f"    free blocks after  = {manager.num_free_blocks}/{manager.num_blocks}")

    return finished, reused_any


def main():
    args = parse_args()

    print("=" * 60)
    print("V1 KV-CACHE MANAGER (Phase 10)")
    print("=" * 60)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)

    print(f"\nCheckpoint:\n    {checkpoint_path}")

    model, metadata = load_inference_checkpoint(checkpoint_path, device=args.device)
    tokenizer = BPETokenizer(args.tokenizer_path)

    sampler = SamplingParams(greedy=True).to_sampler()
    runner = ModelRunner(model=model, tokenizer=tokenizer, sampler=sampler, device=args.device)

    manager = KVCacheManager.for_model(
        model,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        device=args.device,
    )
    engine = InferenceEngine(
        runner=runner,
        tokenizer=tokenizer,
        sampler=sampler,
        device=args.device,
        kv_cache_manager=manager,
    )

    config = model.config
    print(f"\nKV block pool:")
    print(f"    blocks x block_size = {manager.num_blocks} x {manager.block_size} "
          f"= {manager.num_blocks * manager.block_size} token slots")
    print(f"    per layer tensor    = {tuple(manager.pool.k[0].shape)}  "
          f"(blocks, kv_heads, block_size, head_dim)")
    print(f"    layers              = {config.num_layers}  (K and V each)")
    print(f"    pool memory         = {manager.pool.num_bytes / 1024 ** 2:.2f} MB")

    worst_case = sum(
        manager.blocks_needed(len(tokenizer.encode(p)) + args.max_new_tokens)
        for p in args.prompts
    )
    print(f"    blocks to hold ALL requests at once = {worst_case}  "
          f"({'must reuse blocks' if worst_case > manager.num_blocks else 'fits without reuse'})")

    # Warm-up so CUDA init isn't part of the timing.
    engine.generate(args.prompts[0], max_new_tokens=2)

    for prompt in args.prompts:
        engine.submit(engine.create_request(prompt, max_new_tokens=args.max_new_tokens))

    print(f"\nLifecycle (batch_size={args.batch_size}):")

    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()

    finished, reused_any = run_batches(engine, manager, args.batch_size)

    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = sum(r.num_generated for r in finished)

    print(f"\nGeneration:")
    print(f"    requests            = {len(finished)}")
    print(f"    generated tokens    = {generated}")
    print(f"    elapsed             = {elapsed:.3f} s")
    print(f"    tokens/sec          = {generated / elapsed:.1f}")

    print(f"\nCorrectness:")

    all_match = True

    for request in finished:
        naive = runner.generate(request.prompt, max_new_tokens=args.max_new_tokens)
        match = naive["token_ids"] == request.all_tokens
        all_match &= match

        if not match:
            print(f"    MISMATCH {request.request_id}")

    pool_empty = manager.num_free_blocks == manager.num_blocks and not manager.request_ids

    print(f"    tokens match naive  = {all_match}")
    print(f"    all blocks released = {pool_empty}")
    print(f"    blocks reused       = {reused_any}")

    # --------------------------------------------------
    # Out-of-memory demo
    # --------------------------------------------------

    print(f"\nOut-of-memory demo (pool of 2 blocks x {args.block_size}):")

    tiny = KVCacheManager.for_model(model, num_blocks=2, block_size=args.block_size, device=args.device)
    tiny_engine = InferenceEngine(
        runner=runner, tokenizer=tokenizer, sampler=sampler,
        device=args.device, kv_cache_manager=tiny,
    )

    long_prompt = " ".join(args.prompts)
    too_big = tiny_engine.submit(tiny_engine.create_request(long_prompt, max_new_tokens=5))
    small = tiny_engine.submit(tiny_engine.create_request(args.prompts[1], max_new_tokens=5))

    tiny_engine.run_until_complete(max_batch_size=2)

    small_ok = (
        small.all_tokens
        == runner.generate(args.prompts[1], max_new_tokens=5)["token_ids"]
    )
    oom_ok = (
        too_big.finish_reason == FinishReason.OUT_OF_KV_BLOCKS
        and small.finish_reason == FinishReason.MAX_NEW_TOKENS
        and small_ok
        and tiny.num_free_blocks == 2
    )

    print(f"    {too_big.request_id}  prompt={too_big.prompt_len} tok  -> {too_big.finish_reason.value}")
    print(f"    {small.request_id}  prompt={small.prompt_len} tok  -> {small.finish_reason.value}  "
          f"(tokens match naive = {small_ok})")
    print(f"    handled cleanly     = {oom_ok}")

    # Reuse is only guaranteed when the pool can't hold everything.
    reuse_ok = reused_any or worst_case <= manager.num_blocks

    passed = all_match and pool_empty and reuse_ok and oom_ok

    print("\n" + "=" * 60)

    if not passed:
        print("PHASE 10 FAILED")
        print("=" * 60)
        raise RuntimeError("KV-cache manager checks failed. See report above.")

    print("PHASE 10 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
