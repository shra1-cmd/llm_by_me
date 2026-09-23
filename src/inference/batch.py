"""
Phase 9: Batch — a group of requests executed in one forward pass.

A Batch is built fresh for every model call and holds only what that
call needs. All durable state (tokens, KV cache, sampling params,
status) stays on each InferenceRequest.

    Batch
    ├── phase            "prefill" | "decode"
    ├── requests         row b  ->  requests[b]   (the mapping)
    ├── input_ids        [B, T]
    ├── position_ids     [B, T]    absolute position of every input token
    ├── attention_mask   [B, T_k]  True = real key, False = padding
    ├── seq_lens         prefill: prompt length per row
    │                    decode:  cached length per row (before this step)
    └── kv_caches        decode only: each request's own KVCache

Prefill — right padding, T = longest prompt:

    A = [10 20 30 40 50]     mask [1 1 1 1 1]
    B = [10 20 30  P  P]     mask [1 1 1 0 0]
    C = [10 20 30 40  P]     mask [1 1 1 1 0]

    Logits for row b are read at index seq_lens[b] - 1, and row b's
    K/V is sliced to [:seq_lens[b]]; padded positions are discarded.

Decode — one new token per row, rows at different positions:

    past K/V right-padded to L_max, new token appended at L_max:

    A (L=5)  keys [a a a a a | new]     mask [1 1 1 1 1 1]
    B (L=3)  keys [b b b 0 0 | new]     mask [1 1 1 0 0 1]
    C (L=4)  keys [c c c c 0 | new]     mask [1 1 1 1 0 1]

    position_ids = [[5], [3], [4]] — each row's true position.
"""

from dataclasses import dataclass

import torch

from src.inference.kv_cache import KVCache
from src.inference.request import InferenceRequest

PREFILL = "prefill"
DECODE = "decode"


@dataclass
class Batch:
    phase: str
    requests: list[InferenceRequest]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    seq_lens: list[int]
    kv_caches: list[KVCache] | None = None

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def request_ids(self) -> list[str]:
        return [r.request_id for r in self.requests]

    def row_of(self, request_id: str) -> int:
        return self.request_ids.index(request_id)

    def model_attention_mask(self) -> torch.Tensor:
        """
        The full bool mask the model expects, [B, 1, T_q, T_k]:
        key padding combined with causality.
        """

        if self.phase == PREFILL:
            T = self.input_ids.shape[1]

            causal = torch.ones(
                (T, T),
                dtype=torch.bool,
                device=self.input_ids.device,
            ).tril()

            return causal[None, None, :, :] & self.attention_mask[:, None, None, :]

        # Decode: one query per row, which may see every real past key
        # and itself (the last key). Causality is implied.
        return self.attention_mask[:, None, None, :]


def build_prefill_batch(
    requests: list[InferenceRequest],
    pad_token_id: int = 0,
    device: str = "cpu",
) -> Batch:
    if not requests:
        raise ValueError("cannot build an empty batch")

    seq_lens = [r.prompt_len for r in requests]
    T = max(seq_lens)
    B = len(requests)

    input_ids = torch.full((B, T), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.bool)

    for row, request in enumerate(requests):
        n = seq_lens[row]
        input_ids[row, :n] = torch.tensor(request.input_tokens, dtype=torch.long)
        attention_mask[row, :n] = True

    position_ids = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, T)

    return Batch(
        phase=PREFILL,
        requests=list(requests),
        input_ids=input_ids.to(device),
        position_ids=position_ids.to(device),
        attention_mask=attention_mask.to(device),
        seq_lens=seq_lens,
    )


def build_decode_batch(
    requests: list[InferenceRequest],
    device: str = "cpu",
) -> Batch:
    if not requests:
        raise ValueError("cannot build an empty batch")

    for request in requests:
        if request.kv_cache is None:
            raise ValueError(f"{request.request_id}: decode batch needs a KV cache")

    seq_lens = [r.num_cached_tokens for r in requests]
    L_max = max(seq_lens)
    B = len(requests)

    input_ids = torch.tensor(
        [[r.last_token] for r in requests],
        dtype=torch.long,
    )

    position_ids = torch.tensor([[n] for n in seq_lens], dtype=torch.long)

    # Keys seen by the new token: L_max (padded) past slots + itself.
    key_positions = torch.arange(L_max + 1).unsqueeze(0)
    lens = torch.tensor(seq_lens).unsqueeze(1)

    attention_mask = (key_positions < lens) | (key_positions == L_max)

    return Batch(
        phase=DECODE,
        requests=list(requests),
        input_ids=input_ids.to(device),
        position_ids=position_ids.to(device),
        attention_mask=attention_mask.to(device),
        seq_lens=seq_lens,
        kv_caches=[r.kv_cache for r in requests],
    )
