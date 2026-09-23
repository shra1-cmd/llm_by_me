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

`generate` below stays the untouched naive reference.
"""

import time
from dataclasses import dataclass

import torch

from src.inference.kv_cache import KVCache
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
    def prefill(self, input_ids: torch.Tensor) -> PrefillOutput:
        """
        Process the whole prompt in one forward pass.

        input_ids:
            [B, N], N >= 1

        After this call the returned cache holds K/V for all N tokens.
        """

        if input_ids.dim() != 2 or input_ids.shape[1] == 0:
            raise ValueError(
                f"prefill expects input_ids [B, N>=1], got {tuple(input_ids.shape)}"
            )

        kv_cache = KVCache(num_layers=self.model.config.num_layers)

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
