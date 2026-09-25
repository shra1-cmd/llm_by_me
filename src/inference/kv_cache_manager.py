"""
Phase 10: KV-cache manager — block-based ownership of KV memory.

Until Phase 9 every request owned a private, contiguous KVCache that
grew with torch.cat. Phase 10 moves ownership of KV memory into one
manager that hands out fixed-size blocks:

    KV memory pool  (one K and one V tensor per layer)

    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │ B0  │ B1  │ B2  │ B3  │ B4  │ B5  │ B6  │ B7  │   each block =
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘   block_size tokens

    request A (40 tokens, block_size 16) -> [B2, B7, B1]  (3 blocks)
    request B (10 tokens)                -> [B0]

Blocks do not have to be contiguous: a request grows by acquiring
another free block, and a finished request's blocks go back to the
free pool for the next request to reuse.

The file has three layers, deliberately separated:

    LOGICAL   KVCacheManager bookkeeping (no torch involved)
              free pool, request -> block ids, block -> owner,
              token counts; allocate / grow / release

    PHYSICAL  KVBlockPool
              k[layer], v[layer]: [num_blocks, num_kv_heads, block_size, head_dim]
              a block id is simply an index into dim 0

    HANDLE    PagedKVCache — what a request holds instead of KVCache
              same get_seq_length / get / update interface the model
              already uses, so the model does not change. update()
              scatters new K/V into the request's blocks and returns
              the full K/V; append() only scatters (used by the batched
              runner); get() gathers the blocks back into a contiguous
              [1, H, T, D] tensor.

Token position p of a request lives at:

    block  = block_ids[p // block_size]
    offset = p % block_size

Out of memory is handled up front: allocate()/grow() check that
enough free blocks exist *before* taking any, and raise
KVCacheOutOfMemory otherwise — so a failed allocation never leaves
a half-grown request or touches another request's blocks.

Cost of this design: without a PagedAttention kernel, attention still
needs contiguous K/V, so every step gathers each request's blocks
back into one tensor (and batched decode then pads + stacks those).
That extra copying makes batched decode slower than Phase 9's plain
torch.cat caches; removing it is exactly what a paged attention
kernel is for.

Not implemented yet (later phases): a PagedAttention kernel, prefix
caching, copy-on-write / block sharing, quantization, compaction.

Phase 13: pool writes run in profiler region "kv_cache/paged_write",
block gathers in "kv_cache/paged_read" (no-ops unless profiling is
enabled, see src/model/profiling.py).
"""

import math
from collections import deque

import torch

from src.model.profiling import region


class KVCacheOutOfMemory(RuntimeError):
    """Not enough free KV blocks for an allocation. Nothing was changed."""


# ======================================================================
# Physical memory
# ======================================================================


class KVBlockPool:
    """
    The actual K/V tensors. Knows nothing about requests; a block id
    is just a row index into every layer's tensor.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        shape = (num_blocks, num_kv_heads, block_size, head_dim)

        self.num_layers = num_layers
        self.block_size = block_size

        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]

    @property
    def num_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def _slots(self, table: torch.Tensor, start: int, end: int):
        positions = torch.arange(start, end, device=table.device)

        return table[positions // self.block_size], positions % self.block_size

    def write(self, layer_idx: int, table: torch.Tensor, start: int, key, value):
        """
        Store key/value [1, H, T_new, D] at token positions
        [start, start + T_new) of the sequence whose block table is
        `table` (a long tensor of block ids on the pool's device).
        """

        end = start + key.shape[2]
        blocks, offsets = self._slots(table, start, end)

        # Indexing [blocks, :, offsets, :] addresses [T_new, H, D].
        with region("kv_cache/paged_write"):
            self.k[layer_idx][blocks, :, offsets, :] = key[0].transpose(0, 1)
            self.v[layer_idx][blocks, :, offsets, :] = value[0].transpose(0, 1)

    def read(self, layer_idx: int, table: torch.Tensor, length: int):
        """
        Gather the first `length` tokens of the sequence back into
        contiguous tensors, [1, H, length, D].
        """

        num_used = math.ceil(length / self.block_size)
        table = table[:num_used]

        def gather(pool):
            blocks = pool[table]                                  # [nb, H, bs, D]
            H, D = blocks.shape[1], blocks.shape[3]
            flat = blocks.permute(1, 0, 2, 3).reshape(H, -1, D)  # [H, nb*bs, D]
            return flat[:, :length].unsqueeze(0)

        with region("kv_cache/paged_read"):
            return gather(self.k[layer_idx]), gather(self.v[layer_idx])


# ======================================================================
# Logical bookkeeping + ownership
# ======================================================================


class KVCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_layers: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        With only num_blocks/block_size the manager does pure
        bookkeeping (handy for unit tests). Give it the model's
        layer/head dims to also allocate the physical pool that
        PagedKVCache handles read and write.
        """

        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        if block_size < 1:
            raise ValueError("block_size must be >= 1")

        self.num_blocks = num_blocks
        self.block_size = block_size

        self._free: deque[int] = deque(range(num_blocks))
        self._blocks: dict[str, list[int]] = {}
        self._num_tokens: dict[str, int] = {}
        self._owner: dict[int, str] = {}

        self.pool = None

        if num_layers is not None:
            self.pool = KVBlockPool(
                num_blocks=num_blocks,
                block_size=block_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )

    @classmethod
    def for_model(
        cls,
        model,
        num_blocks: int | None = None,
        block_size: int = 16,
        device: str = "cpu",
        max_full_sequences: int = 16,
    ) -> "KVCacheManager":
        """
        Size a pool for `model`. Default capacity: enough blocks for
        `max_full_sequences` sequences of max_seq_len tokens.
        """

        config = model.config

        if num_blocks is None:
            num_blocks = math.ceil(config.max_seq_len / block_size) * max_full_sequences

        dtype = next(model.parameters()).dtype

        return cls(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.hidden_dim // config.num_q_heads,
            device=device,
            dtype=dtype,
        )

    # --------------------------------------------------
    # Queries
    # --------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_allocated_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def free_block_ids(self) -> list[int]:
        return list(self._free)

    @property
    def request_ids(self) -> list[str]:
        return list(self._blocks)

    def blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def has(self, request_id: str) -> bool:
        return request_id in self._blocks

    def get_block_ids(self, request_id: str) -> list[int]:
        return list(self._blocks[request_id])

    def get_num_tokens(self, request_id: str) -> int:
        return self._num_tokens[request_id]

    def owner_of(self, block_id: int) -> str | None:
        return self._owner.get(block_id)

    def can_allocate(self, num_tokens: int, request_id: str | None = None) -> bool:
        """
        Would reserving `num_tokens` total tokens for `request_id`
        (new, or growing an existing one) fit in the free pool?
        """

        have = len(self._blocks.get(request_id, [])) if request_id else 0
        return self.blocks_needed(num_tokens) - have <= len(self._free)

    # --------------------------------------------------
    # Allocate / grow / release
    # --------------------------------------------------

    def _take(self, request_id: str, count: int) -> list[int]:
        if count > len(self._free):
            raise KVCacheOutOfMemory(
                f"{request_id}: needs {count} more KV block(s), "
                f"only {len(self._free)} of {self.num_blocks} free"
            )

        taken = [self._free.popleft() for _ in range(count)]

        for block_id in taken:
            self._owner[block_id] = request_id

        return taken

    def allocate(self, request_id: str, num_tokens: int) -> list[int]:
        """Reserve blocks for a new request's first `num_tokens` tokens."""

        if request_id in self._blocks:
            raise ValueError(f"{request_id} already has KV blocks; use grow()")

        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")

        blocks = self._take(request_id, self.blocks_needed(num_tokens))

        self._blocks[request_id] = blocks
        self._num_tokens[request_id] = num_tokens

        return list(blocks)

    def grow(self, request_id: str, num_tokens: int) -> list[int]:
        """
        Make sure `request_id` can hold `num_tokens` tokens in total.
        Returns only the newly added block ids (empty if it already
        had room). Never shrinks.
        """

        if request_id not in self._blocks:
            raise KeyError(f"{request_id} has no KV blocks; allocate() first")

        blocks = self._blocks[request_id]
        missing = self.blocks_needed(num_tokens) - len(blocks)

        added = self._take(request_id, missing) if missing > 0 else []

        blocks.extend(added)
        self._num_tokens[request_id] = max(self._num_tokens[request_id], num_tokens)

        return added

    def release(self, request_id: str) -> list[int]:
        """Return all of a request's blocks to the free pool."""

        if request_id not in self._blocks:
            raise KeyError(f"{request_id} has no KV blocks")

        blocks = self._blocks.pop(request_id)
        del self._num_tokens[request_id]

        for block_id in blocks:
            del self._owner[block_id]
            self._free.append(block_id)

        return blocks

    # --------------------------------------------------
    # Handles
    # --------------------------------------------------

    def create_cache(self, request_id: str, num_tokens: int = 0) -> "PagedKVCache":
        """
        Allocate blocks for `num_tokens` and return the handle the
        request uses as its kv_cache.
        """

        if self.pool is None:
            raise RuntimeError("manager has no physical pool; build it with model dims")

        self.allocate(request_id, num_tokens)

        return PagedKVCache(self, request_id)


# ======================================================================
# Request-side handle
# ======================================================================


class PagedKVCache:
    """
    A request's view of its KV state, backed by manager blocks.

    Drop-in for KVCache: the model calls get_seq_length / update per
    layer exactly as before. Lengths are tracked per layer because the
    model updates layer 0, then layer 1, ... within one forward, and
    every layer must see the pre-forward length.

    After the manager releases the request, the handle keeps its
    final length (for stats) but refuses to read or write.

    The block table is kept as a device tensor and rebuilt only when
    the request gains a block, so a normal decode step does no
    host->device copies.
    """

    def __init__(self, manager: KVCacheManager, request_id: str):
        self.manager = manager
        self.request_id = request_id
        self.num_layers = manager.pool.num_layers

        self._lens = [0] * self.num_layers
        self.peak_num_blocks = len(manager.get_block_ids(request_id))

        self._table: torch.Tensor | None = None
        self._table_len = -1

    def _block_table(self) -> torch.Tensor:
        block_ids = self.manager.get_block_ids(self.request_id)

        if len(block_ids) != self._table_len:
            self._table = torch.tensor(
                block_ids,
                dtype=torch.long,
                device=self.manager.pool.k[0].device,
            )
            self._table_len = len(block_ids)
            self.peak_num_blocks = max(self.peak_num_blocks, len(block_ids))

        return self._table

    @property
    def is_released(self) -> bool:
        return not self.manager.has(self.request_id)

    @property
    def block_ids(self) -> list[int]:
        """Blocks currently held ([] once released)."""

        if self.is_released:
            return []

        return self.manager.get_block_ids(self.request_id)

    def _check_live(self):
        if self.is_released:
            raise RuntimeError(f"{self.request_id}: KV cache already released")

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._lens[layer_idx]

    def get(self, layer_idx: int):
        self._check_live()

        n = self._lens[layer_idx]

        if n == 0:
            return None

        return self.manager.pool.read(layer_idx, self._block_table(), n)

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """
        key/value: [1, num_kv_heads, T_new, head_dim]

        Grow (acquiring blocks if needed) and scatter the new K/V into
        this request's blocks. Returns nothing — used when the caller
        doesn't need the full K/V back (batched prefill/decode scatter).
        """

        self._check_live()

        if key.shape[0] != 1:
            raise ValueError("PagedKVCache holds one sequence; batch dim must be 1")

        start = self._lens[layer_idx]
        end = start + key.shape[2]

        self.manager.grow(self.request_id, end)

        self.manager.pool.write(layer_idx, self._block_table(), start, key, value)
        self._lens[layer_idx] = end

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """
        The model-facing call: append, then return the full K/V so far
        ([1, H, T, D]) for attention.
        """

        self.append(layer_idx, key, value)

        return self.manager.pool.read(layer_idx, self._block_table(), self._lens[layer_idx])
