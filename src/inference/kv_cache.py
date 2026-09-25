"""
Phase 4: per-layer key/value cache for single-sequence decoding.

Holds one (key, value) tensor pair per transformer layer and grows
them along the sequence dimension as new tokens are processed. It is
mutated in place by the model during `forward`, so callers never
need to thread cache state through return values.

One sequence per KVCache. No eviction. BatchedKVCache (below) only
stacks several of them temporarily for one batched forward pass.

Phase 13: the torch.cat growth runs in profiler region "kv_cache/cat",
BatchedKVCache's pad+stack in "kv_cache/batch_gather" (no-ops unless
profiling is enabled, see src/model/profiling.py).

Phase 15: StaticKVCache is a drop-in replacement for KVCache that
preallocates [B, H, capacity, D] per layer on first use and writes
each step's K/V in place at the current length, instead of
torch.cat-ing a new, one-token-longer tensor every step:

    KVCache.update        allocate T+1 rows, copy T old + 1 new  (every step)
    StaticKVCache.update  copy 1 new row into buffer[:, :, T]    (no allocation)

Region "kv_cache/write" marks the in-place write.
"""

import torch

from src.model.profiling import region


class KVCache:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers

        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Number of tokens already cached for `layer_idx`.

        All layers are updated together on every forward call, so any
        layer index gives the same answer; layer 0 is used as the
        default.
        """

        if self.key_cache[layer_idx] is None:
            return 0

        return self.key_cache[layer_idx].shape[2]

    def get(self, layer_idx: int):
        """
        Existing (key, value) for this layer, or None if nothing has
        been cached yet.

        key/value shape: [B, num_kv_heads, T_past, head_dim]
        """

        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]

        if k is None:
            return None

        return k, v

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Append the newly computed key/value for this layer's new
        tokens to whatever is already cached, store the result, and
        return the full (key, value) so far.

        key/value shape: [B, num_kv_heads, T_new, head_dim]
        """

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
        else:
            with region("kv_cache/cat"):
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key],
                    dim=2,
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value],
                    dim=2,
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """update() without needing the result (same interface as PagedKVCache)."""

        self.update(layer_idx, key, value)

    @classmethod
    def from_tensors(
        cls,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> "KVCache":
        """
        Build a cache that already holds `keys[i]` / `values[i]` for
        layer i. Used to hand each request its own slice of a batched
        prefill (Phase 9).
        """

        cache = cls(num_layers=len(keys))
        cache.key_cache = list(keys)
        cache.value_cache = list(values)

        return cache


class StaticKVCache:
    """
    Phase 15: preallocated per-layer K/V storage with a write cursor.

        buffer [B, H, capacity, D]   allocated on the first update
        ┌──────────────────────┬───────────────────┐
        │ used: [0, length)    │ free              │
        └──────────────────────┴───────────────────┘
        update(k_new [B, H, t, D]) -> buffer[:, :, length:length+t] = k_new
                                      length += t
                                      return buffer[:, :, :length] (a view)

    Same interface as KVCache (get_seq_length / get / update / append /
    from_tensors), so the model, ModelRunner and BatchedKVCache work
    with either. Buffers take shape, dtype and device from the first
    K/V they see, so the cache needs only the layer count and capacity.

    The views returned by get/update alias the buffer: a later update
    writes past their end, never inside them, so earlier results stay
    valid.
    """

    def __init__(self, num_layers: int, capacity: int):
        if capacity < 1:
            raise ValueError("StaticKVCache capacity must be >= 1")

        self.num_layers = num_layers
        self.capacity = capacity

        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers
        self.lengths = [0] * num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.lengths[layer_idx]

    def get(self, layer_idx: int):
        n = self.lengths[layer_idx]

        if n == 0:
            return None

        return (
            self.key_cache[layer_idx][:, :, :n],
            self.value_cache[layer_idx][:, :, :n],
        )

    def _allocate(self, layer_idx: int, key: torch.Tensor):
        B, H, _, D = key.shape
        shape = (B, H, self.capacity, D)

        self.key_cache[layer_idx] = torch.empty(shape, dtype=key.dtype, device=key.device)
        self.value_cache[layer_idx] = torch.empty(shape, dtype=key.dtype, device=key.device)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Write the new tokens' K/V at the current length and return the
        full (key, value) so far as views of the buffer.

        key/value shape: [B, num_kv_heads, T_new, head_dim]
        """

        start = self.lengths[layer_idx]
        end = start + key.shape[2]

        if end > self.capacity:
            raise ValueError(
                f"StaticKVCache overflow: {end} tokens > capacity {self.capacity}"
            )

        if self.key_cache[layer_idx] is None:
            self._allocate(layer_idx, key)

        with region("kv_cache/write"):
            self.key_cache[layer_idx][:, :, start:end].copy_(key)
            self.value_cache[layer_idx][:, :, start:end].copy_(value)

        self.lengths[layer_idx] = end

        return (
            self.key_cache[layer_idx][:, :, :end],
            self.value_cache[layer_idx][:, :, :end],
        )

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """update() without needing the result (same interface as PagedKVCache)."""

        self.update(layer_idx, key, value)

    @classmethod
    def from_tensors(
        cls,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        capacity: int,
    ) -> "StaticKVCache":
        """Build a cache holding keys[i] / values[i] for layer i (copied in)."""

        cache = cls(num_layers=len(keys), capacity=capacity)

        for layer_idx, (k, v) in enumerate(zip(keys, values)):
            cache.update(layer_idx, k, v)

        return cache


class BatchedKVCache:
    """
    Phase 9: temporary, batch-scoped view of several requests' caches.

    Built fresh for one batched decode step from each request's own
    KVCache, then thrown away. Each request's cache stays the source
    of truth; this object never outlives the forward call.

        request caches (lengths L_0, L_1, ...)
            -> right-pad each to L_max, stack on batch dim
            -> model forward appends the new token at index L_max
            -> new_kv(layer) returns [B, H, 1, D] to scatter back

    Padded slots hold zeros and must be masked out by the caller's
    attention_mask. Same `get_seq_length` / `update` interface as
    KVCache, so the model can't tell the difference.
    """

    def __init__(self, caches: list[KVCache]):
        if not caches:
            raise ValueError("BatchedKVCache needs at least one cache")

        self.num_layers = caches[0].num_layers
        self.seq_lens = [cache.get_seq_length() for cache in caches]
        self.max_len = max(self.seq_lens)

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        with region("kv_cache/batch_gather"):
            for layer_idx in range(self.num_layers):
                self.key_cache.append(self._pad_and_stack(caches, layer_idx, keys=True))
                self.value_cache.append(self._pad_and_stack(caches, layer_idx, keys=False))

    def _pad_and_stack(self, caches, layer_idx, keys):
        rows = []

        for cache in caches:
            k, v = cache.get(layer_idx)
            t = k if keys else v

            pad = self.max_len - t.shape[2]

            if pad > 0:
                t = torch.nn.functional.pad(t, (0, 0, 0, pad))

            rows.append(t)

        return torch.cat(rows, dim=0)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.key_cache[layer_idx].shape[2]

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        with region("kv_cache/cat"):
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def new_kv(self, layer_idx: int, num_new: int = 1):
        """K/V appended by the last forward, [B, H, num_new, D]."""

        return (
            self.key_cache[layer_idx][:, :, -num_new:],
            self.value_cache[layer_idx][:, :, -num_new:],
        )
