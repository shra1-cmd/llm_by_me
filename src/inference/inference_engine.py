"""
Phase 6/7: prefill + decode inference engine, driven by requests.

ModelRunner answers "how do I execute the model?".
InferenceEngine answers "how do I run a generation request?".

Phase 7 moves all per-generation state (tokens, sampling params,
KV cache, position, status) into an InferenceRequest. The engine is
now stateless between requests; it only advances a request through
its lifecycle:

    create_request(prompt)                 WAITING
        -> prefill(request)                PREFILLING
               runner.prefill(input_tokens)       KV cache len = N
               sample -> append_token             -> DECODING / FINISHED
        -> loop decode(request)            DECODING
               runner.decode(last_token, request.kv_cache)
               sample -> append_token             -> DECODING / FINISHED
        -> FINISHED (EOS / max_new_tokens / max_seq_len)

`generate(prompt)` is kept as the convenience wrapper around that
lifecycle and returns the same result shape as Phase 6.

The last sampled token is never fed back through the model (there is
nothing left to predict), so at the end:

    final KV length = prompt_tokens + generated_tokens - 1

One request at a time. No scheduling, batching or cache memory
management yet — those are later phases.
"""

import time

import torch

from src.inference.model_runner import ModelRunner
from src.inference.request import (
    InferenceRequest,
    RequestStatus,
    SamplingParams,
)
from src.inference.sampler import Sampler


class InferenceEngine:
    def __init__(
        self,
        runner: ModelRunner,
        tokenizer,
        sampler: Sampler | None = None,
        device: str = "cuda",
        eos_token: str = "<eos>",
    ):
        self.runner = runner
        self.tokenizer = tokenizer
        self.device = device

        # Engine-level default; each request may carry its own.
        self.default_sampling_params = (
            SamplingParams.from_sampler(sampler)
            if sampler is not None
            else SamplingParams()
        )

        self.eos_token_id = tokenizer.token_to_id(eos_token)
        self.max_seq_len = runner.max_seq_len

        self._samplers: dict[SamplingParams, Sampler] = {}

    def _sync(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _sampler_for(self, request: InferenceRequest) -> Sampler:
        params = request.sampling_params

        if params not in self._samplers:
            self._samplers[params] = params.to_sampler()

        return self._samplers[params]

    def _sample(self, request: InferenceRequest, logits: torch.Tensor) -> int:
        history = torch.tensor(
            [request.all_tokens],
            dtype=torch.long,
            device=self.device,
        )

        return self._sampler_for(request).sample(logits, history).item()

    # --------------------------------------------------
    # Request construction
    # --------------------------------------------------

    def create_request(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> InferenceRequest:
        return self.create_request_from_ids(
            self.tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt=prompt,
        )

    def create_request_from_ids(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        prompt: str | None = None,
    ) -> InferenceRequest:
        kwargs = {}

        if request_id is not None:
            kwargs["request_id"] = request_id

        return InferenceRequest(
            prompt=prompt if prompt is not None else self.tokenizer.decode(prompt_ids),
            input_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params or self.default_sampling_params,
            eos_token_id=self.eos_token_id,
            max_seq_len=self.max_seq_len,
            **kwargs,
        )

    # --------------------------------------------------
    # Lifecycle steps
    # --------------------------------------------------

    @torch.inference_mode()
    def prefill(self, request: InferenceRequest):
        """
        WAITING -> PREFILLING -> DECODING (or FINISHED).

        Runs the whole prompt through the model, attaches a fresh KV
        cache to the request and samples the first new token.
        """

        if request.status != RequestStatus.WAITING:
            raise RuntimeError(
                f"{request.request_id}: prefill requires WAITING, got {request.status.name}"
            )

        # Nothing to generate: finish without touching the model,
        # exactly like the naive loop running zero iterations.
        reason = request.check_stop()

        if reason is not None:
            request.finish(reason)
            return

        request.start_prefill()

        input_ids = torch.tensor(
            [request.input_tokens],
            dtype=torch.long,
            device=self.device,
        )

        output = self.runner.prefill(input_ids)
        request.kv_cache = output.kv_cache

        request.append_token(self._sample(request, output.logits))

    @torch.inference_mode()
    def decode(self, request: InferenceRequest):
        """
        One DECODING step: feed the request's last token through its
        own KV cache and sample the next one.
        """

        if request.status != RequestStatus.DECODING:
            raise RuntimeError(
                f"{request.request_id}: decode requires DECODING, got {request.status.name}"
            )

        # The cache must hold every token except the last sampled one.
        if request.num_cached_tokens != request.position - 1:
            raise RuntimeError(
                f"{request.request_id}: KV cache length {request.num_cached_tokens} "
                f"does not match position {request.position} - 1"
            )

        token = torch.tensor(
            [[request.last_token]],
            dtype=torch.long,
            device=self.device,
        )

        output = self.runner.decode(token, request.kv_cache)

        request.append_token(self._sample(request, output.logits))

    def run(self, request: InferenceRequest) -> dict:
        """
        Drive one request from WAITING to FINISHED. Returns stats.
        """

        self._sync()
        start = time.perf_counter()

        self.prefill(request)

        prefill_kv_len = request.num_cached_tokens

        self._sync()
        prefill_elapsed = time.perf_counter() - start

        start = time.perf_counter()

        while request.status == RequestStatus.DECODING:
            self.decode(request)

        self._sync()
        decode_elapsed = time.perf_counter() - start

        elapsed = prefill_elapsed + decode_elapsed

        return {
            "request_id": request.request_id,
            "finish_reason": request.finish_reason.value,
            "prompt_tokens": request.prompt_len,
            "prefill_kv_length": prefill_kv_len,
            "generated_tokens": request.num_generated,
            "total_tokens": request.position,
            "final_kv_length": request.num_cached_tokens,
            "prefill_seconds": prefill_elapsed,
            "decode_seconds": decode_elapsed,
            "elapsed_seconds": elapsed,
            "tokens_per_second": (
                request.num_generated / elapsed if elapsed > 0 else float("inf")
            ),
        }

    # --------------------------------------------------
    # Convenience wrappers (Phase 6 API)
    # --------------------------------------------------

    def _result(self, request: InferenceRequest, stats: dict) -> dict:
        return {
            "request": request,
            "token_ids": request.all_tokens,
            "kv_cache": request.kv_cache,
            "stats": stats,
        }

    def generate_from_ids(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        sampling_params: SamplingParams | None = None,
    ) -> dict:
        request = self.create_request_from_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )

        return self._result(request, self.run(request))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        sampling_params: SamplingParams | None = None,
    ) -> dict:
        """
        prompt -> generated text, same result shape as
        ModelRunner.generate plus the prefill/decode stats and the
        finished request.
        """

        request = self.create_request(
            prompt,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )

        result = self._result(request, self.run(request))
        result["text"] = self.tokenizer.decode(request.all_tokens)

        return result
