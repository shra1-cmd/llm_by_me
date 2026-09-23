"""
Phase 3: naive autoregressive inference loop.

    prompt
        -> tokenizer.encode
        -> input_ids [1, T]
        -> loop:
               model(generated) -> logits [1, T, vocab]
               logits[:, -1, :] -> [1, vocab]
               sampler.sample(...) -> next_token [1]
               generated = cat(generated, next_token)
               stop at EOS or max_new_tokens
        -> tokenizer.decode
        -> generated text

No KV cache: every step recomputes the forward pass over the full
sequence so far. This is intentionally the naive baseline that
Phase 4 (KV cache) will be measured against.

Phase 6 adds the two low-level execution modes the InferenceEngine
drives ("how do I execute the model?"):

    prefill(input_ids [1, N])          -> logits + fresh KV cache (len N)
    decode(token [1, 1], kv_cache)     -> logits + same cache (len +1)

Phase 9 adds the batched versions of both modes:

    prefill_batch(batch)   -> logits [B, vocab] + one new KVCache per row
    decode_batch(batch)    -> logits [B, vocab]; each row's own KVCache
                              grows by 1

The runner never shares KV state between rows: batched K/V only
exists for the duration of one forward call.

Phase 10: prefill / prefill_batch accept the cache(s) to fill, so the
engine can pass PagedKVCache handles backed by the KVCacheManager's
blocks. Without them the runner falls back to a private KVCache, as
in Phases 6-9. Decode works with either, since both expose the same
get_seq_length / get / update interface.

`generate` below stays the untouched naive reference.
"""

import time
from dataclasses import dataclass

import torch

from src.inference.batch import Batch, DECODE, PREFILL
from src.inference.kv_cache import BatchedKVCache, KVCache
from src.inference.sampler import Sampler


@dataclass
class PrefillOutput:
    # [B, vocab] — logits for the token after the last prompt token.
    logits: torch.Tensor
    kv_cache: KVCache


@dataclass
class DecodeOutput:
    # [B, vocab] — logits for the token after the decoded token.
    logits: torch.Tensor
    kv_cache: KVCache


@dataclass
class BatchPrefillOutput:
    # [B, vocab] — row b: logits after request b's last prompt token.
    logits: torch.Tensor
    # One new, independent KVCache per row, in batch order.
    kv_caches: list[KVCache]


@dataclass
class BatchDecodeOutput:
    # [B, vocab]
    logits: torch.Tensor
    # The requests' own caches, each grown by one.
    kv_caches: list[KVCache]


class ModelRunner:
    def __init__(
        self,
        model,
        tokenizer,
        sampler: Sampler,
        device: str = "cuda",
        eos_token: str = "<eos>",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.device = device

        self.eos_token_id = tokenizer.token_to_id(eos_token)

        max_seq_len = getattr(model, "config", None)
        self.max_seq_len = (
            max_seq_len.max_seq_len if max_seq_len is not None else None
        )

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, kv_cache=None) -> PrefillOutput:
        """
        Process the whole prompt in one forward pass.

        input_ids:
            [B, N], N >= 1

        kv_cache:
            empty cache to fill (e.g. a PagedKVCache from the
            KVCacheManager), or None for a fresh private KVCache.

        After this call the returned cache holds K/V for all N tokens.
        """

        if input_ids.dim() != 2 or input_ids.shape[1] == 0:
            raise ValueError(
                f"prefill expects input_ids [B, N>=1], got {tuple(input_ids.shape)}"
            )

        if kv_cache is None:
            kv_cache = KVCache(num_layers=self.model.config.num_layers)
        elif kv_cache.get_seq_length() != 0:
            raise ValueError("prefill needs an empty KV cache")

        logits, _ = self.model(input_ids, kv_cache=kv_cache)

        return PrefillOutput(logits=logits[:, -1, :], kv_cache=kv_cache)

    @torch.inference_mode()
    def decode(self, input_ids: torch.Tensor, kv_cache: KVCache) -> DecodeOutput:
        """
        Process exactly one new token per sequence against an
        existing cache. The cache is mutated in place (grows by 1)
        and returned for convenience.

        input_ids:
            [B, 1]
        """

        if input_ids.dim() != 2 or input_ids.shape[1] != 1:
            raise ValueError(
                f"decode expects input_ids [B, 1], got {tuple(input_ids.shape)}"
            )

        if kv_cache.get_seq_length() == 0:
            raise ValueError("decode requires a populated KV cache; call prefill first")

        logits, _ = self.model(input_ids, kv_cache=kv_cache)

        return DecodeOutput(logits=logits[:, -1, :], kv_cache=kv_cache)

    @torch.inference_mode()
    def prefill_batch(self, batch: Batch, kv_caches=None) -> BatchPrefillOutput:
        """
        One forward pass over a right-padded [B, T] prompt batch.

        Row b's logits are taken at its last real token, and its K/V
        is sliced to its real length and copied into row b's own
        cache, so no row's cache aliases another's (or the batch's).

        kv_caches:
            one empty cache per row to fill (e.g. PagedKVCache
            handles), or None for brand-new private KVCaches.
        """

        if batch.phase != PREFILL:
            raise ValueError(f"prefill_batch got a {batch.phase} batch")

        if kv_caches is not None:
            if len(kv_caches) != batch.size:
                raise ValueError("prefill_batch needs one KV cache per row")

            if any(c.get_seq_length() != 0 for c in kv_caches):
                raise ValueError("prefill_batch needs empty KV caches")

        batched_cache = KVCache(num_layers=self.model.config.num_layers)

        logits, _ = self.model(
            batch.input_ids,
            kv_cache=batched_cache,
            position_ids=batch.position_ids,
            attention_mask=batch.model_attention_mask(),
        )

        last_index = torch.tensor(batch.seq_lens, device=logits.device) - 1
        rows = torch.arange(batch.size, device=logits.device)

        last_logits = logits[rows, last_index, :]

        if kv_caches is not None:
            # Scatter each row's real tokens into its own cache.
            for layer_idx in range(batched_cache.num_layers):
                k, v = batched_cache.get(layer_idx)

                for row, n in enumerate(batch.seq_lens):
                    kv_caches[row].append(
                        layer_idx,
                        k[row:row + 1, :, :n],
                        v[row:row + 1, :, :n],
                    )

            return BatchPrefillOutput(logits=last_logits, kv_caches=list(kv_caches))

        kv_caches = []

        for row, n in enumerate(batch.seq_lens):
            keys = []
            values = []

            for layer_idx in range(batched_cache.num_layers):
                k, v = batched_cache.get(layer_idx)
                keys.append(k[row:row + 1, :, :n].clone())
                values.append(v[row:row + 1, :, :n].clone())

            kv_caches.append(KVCache.from_tensors(keys, values))

        return BatchPrefillOutput(logits=last_logits, kv_caches=kv_caches)

    @torch.inference_mode()
    def decode_batch(self, batch: Batch) -> BatchDecodeOutput:
        """
        One forward pass over one new token per row.

        Each row's cache is padded into a temporary BatchedKVCache;
        after the forward only that row's new K/V is appended back to
        its own KVCache.
        """

        if batch.phase != DECODE:
            raise ValueError(f"decode_batch got a {batch.phase} batch")

        batched_cache = BatchedKVCache(batch.kv_caches)

        logits, _ = self.model(
            batch.input_ids,
            kv_cache=batched_cache,
            position_ids=batch.position_ids,
            attention_mask=batch.model_attention_mask(),
        )

        for layer_idx in range(batched_cache.num_layers):
            new_k, new_v = batched_cache.new_kv(layer_idx)

            for row, cache in enumerate(batch.kv_caches):
                cache.append(
                    layer_idx,
                    new_k[row:row + 1],
                    new_v[row:row + 1],
                )

        return BatchDecodeOutput(logits=logits[:, -1, :], kv_caches=batch.kv_caches)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> dict:
        """
        prompt -> generated text, plus a small stats dict for
        Phase 3's naive-inference baseline.
        """

        prompt_ids = self.tokenizer.encode(prompt)

        generated = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device,
        )

        prompt_len = generated.shape[1]

        start = time.perf_counter()

        num_generated = 0

        for _ in range(max_new_tokens):

            if (
                self.max_seq_len is not None
                and generated.shape[1] >= self.max_seq_len
            ):
                break

            logits, _ = self.model(generated)

            next_token_logits = logits[:, -1, :]

            next_token = self.sampler.sample(
                next_token_logits,
                generated,
            )

            next_token = next_token.unsqueeze(-1)

            generated = torch.cat([generated, next_token], dim=-1)

            num_generated += 1

            if (
                self.eos_token_id is not None
                and next_token.item() == self.eos_token_id
            ):
                break

        elapsed = time.perf_counter() - start

        generated_ids = generated[0].tolist()

        text = self.tokenizer.decode(generated_ids)

        stats = {
            "prompt_tokens": prompt_len,
            "generated_tokens": num_generated,
            "total_tokens": len(generated_ids),
            "elapsed_seconds": elapsed,
            "tokens_per_second": (
                num_generated / elapsed if elapsed > 0 else float("inf")
            ),
        }

        return {
            "text": text,
            "token_ids": generated_ids,
            "stats": stats,
        }
