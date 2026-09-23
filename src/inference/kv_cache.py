"""
Phase 4: per-layer key/value cache for single-sequence decoding.

Holds one (key, value) tensor pair per transformer layer and grows
them along the sequence dimension as new tokens are processed. It is
mutated in place by the model during `forward`, so callers never
need to thread cache state through return values.

One sequence only. No batching across requests, no eviction.
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
