"""
Phase 4: per-layer key/value cache for single-sequence decoding.

Holds one (key, value) tensor pair per transformer layer and grows
them along the sequence dimension as new tokens are processed. It is
mutated in place by the model during `forward`, so callers never
need to thread cache state through return values.

One sequence per KVCache. No eviction. BatchedKVCache (below) only
stacks several of them temporarily for one batched forward pass.
"""

import torch


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
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def new_kv(self, layer_idx: int, num_new: int = 1):
        """K/V appended by the last forward, [B, H, num_new, D]."""

        return (
            self.key_cache[layer_idx][:, :, -num_new:],
            self.value_cache[layer_idx][:, :, -num_new:],
        )
