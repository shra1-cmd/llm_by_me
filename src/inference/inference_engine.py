"""
Phase 6-10: prefill + decode inference engine, driven by requests,
a scheduler, batching and a KV-cache manager.

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

Phase 8 puts a Scheduler in front of execution. The scheduler
decides which request runs next; the engine executes it:

    submit(request)                        -> scheduler.add
    step()                                 -> scheduler.next
                                              execute(request)
                                              scheduler.complete
    run_until_complete()                   -> step() until no work

Phase 9 adds static batching: the scheduler hands over a group of
requests and they share every forward pass until each one finishes:

    step_batch(n)          -> scheduler.next_batch(n)
                              prefill_batch(requests)       one [B, T] forward
                              loop decode_batch(active)     one [B', 1] forward
                                  (finished rows drop out; nobody joins)
                              scheduler.complete(each)

Only model execution is shared. Tokens, KV cache, sampling params
and stop conditions stay per request, and every row is sampled with
its own request's sampler.

Phase 10 hands KV memory to a KVCacheManager. The engine owns the
memory lifecycle of every request:

    prefill        manager.create_cache(id, prompt_len)   ALLOCATE blocks
                   runner writes the prompt's K/V into those blocks
    decode         manager.grow(id, position)             GROW (maybe +1 block)
    FINISHED /     manager.release(id)                    RELEASE -> free pool
    ABORTED                                               (REUSE by later requests)

request.kv_cache is now a PagedKVCache handle; request.block_ids
shows which blocks it holds. If the pool can't cover an allocation,
that request alone is aborted with FinishReason.OUT_OF_KV_BLOCKS and
nobody else's memory is touched. step_batch only admits (FIFO) as
many requests as the free pool can hold prompts for.

Continuous batching lives in continuous_batching.py (Phase 11).

Phase 12: `paged_kv=False` switches back to Phase 6-9 private,
contiguous KVCache objects (no KVCacheManager blocks), so benchmarks
can measure the original KV-cache path and the cost of paging
separately. Default stays paged.

Phase 13: sampling runs in profiler region "sampling" and batch
construction in "batch/build" (no-ops unless profiling is enabled,
see src/model/profiling.py).
"""

import time

import torch

from src.inference.batch import build_decode_batch, build_prefill_batch
from src.inference.kv_cache import KVCache
from src.inference.kv_cache_manager import KVCacheManager, KVCacheOutOfMemory
from src.inference.model_runner import ModelRunner
from src.inference.request import (
    FinishReason,
    InferenceRequest,
    RequestStatus,
    SamplingParams,
)
from src.inference.sampler import Sampler
from src.inference.scheduler import Scheduler
from src.model.profiling import region


class InferenceEngine:
    def __init__(
        self,
        runner: ModelRunner,
        tokenizer,
        sampler: Sampler | None = None,
        device: str = "cuda",
        eos_token: str = "<eos>",
        scheduler: Scheduler | None = None,
        kv_cache_manager: KVCacheManager | None = None,
        paged_kv: bool = True,
    ):
        self.runner = runner
        self.scheduler = scheduler if scheduler is not None else Scheduler()
        self.paged_kv = paged_kv

        # Default pool: room for 16 full-length sequences.
        self.kv_cache_manager = (
            kv_cache_manager
            if kv_cache_manager is not None
            else KVCacheManager.for_model(runner.model, device=device)
        )
        self.tokenizer = tokenizer
        self.device = device

        # Engine-level default; each request may carry its own.
        self.default_sampling_params = (
            SamplingParams.from_sampler(sampler)
            if sampler is not None
            else SamplingParams()
        )

        self.eos_token_id = tokenizer.token_to_id(eos_token)

        # Any id works: padded positions are masked out. Use <pad> if
        # the tokenizer has one so padded inputs are recognisable.
        pad_token_id = tokenizer.token_to_id("<pad>")
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.max_seq_len = runner.max_seq_len

        self._samplers: dict[SamplingParams, Sampler] = {}

        # Results of requests finished by step(), keyed by request_id,
        # until the caller collects them with pop_result().
        self._finished: dict[str, dict] = {}

    def _sync(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _sampler_for(self, request: InferenceRequest) -> Sampler:
        params = request.sampling_params

        if params not in self._samplers:
            self._samplers[params] = params.to_sampler()

        return self._samplers[params]

    def _sample(self, request: InferenceRequest, logits: torch.Tensor) -> int:
        with region("sampling"):
            history = torch.tensor(
                [request.all_tokens],
                dtype=torch.long,
                device=self.device,
            )

            return self._sampler_for(request).sample(logits, history).item()

    # --------------------------------------------------
    # KV memory lifecycle (Phase 10)
    # --------------------------------------------------

    def _allocate(self, request: InferenceRequest) -> bool:
        """
        Give a WAITING request blocks for its prompt. On OOM the
        request is aborted (nothing was allocated) and False returned.
        With paged_kv=False it gets a private contiguous KVCache.
        """

        if not self.paged_kv:
            request.kv_cache = self.runner.new_kv_cache()
            return True

        try:
            request.kv_cache = self.kv_cache_manager.create_cache(
                request.request_id,
                request.prompt_len,
            )
        except KVCacheOutOfMemory:
            request.abort(FinishReason.OUT_OF_KV_BLOCKS)
            return False

        return True

    def _reserve_decode_slot(self, request: InferenceRequest) -> bool:
        """
        Make room for the token this decode step will write (the
        cache grows to `position`). On OOM the request is aborted and
        its blocks released.
        """

        if not self.paged_kv:
            return True

        try:
            self.kv_cache_manager.grow(request.request_id, request.position)
        except KVCacheOutOfMemory:
            request.abort(FinishReason.OUT_OF_KV_BLOCKS)
            self._release(request)
            return False

        return True

    def _release(self, request: InferenceRequest):
        if self.kv_cache_manager.has(request.request_id):
            self.kv_cache_manager.release(request.request_id)

    def _release_if_finished(self, request: InferenceRequest):
        if request.is_finished:
            self._release(request)

    def _kv_stats(self, request: InferenceRequest) -> dict:
        return {
            "kv_block_size": self.kv_cache_manager.block_size,
            "kv_blocks_peak": getattr(request.kv_cache, "peak_num_blocks", 0),
        }

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

        if not self._allocate(request):
            return

        request.start_prefill()

        input_ids = torch.tensor(
            [request.input_tokens],
            dtype=torch.long,
            device=self.device,
        )

        output = self.runner.prefill(input_ids, kv_cache=request.kv_cache)

        request.append_token(self._sample(request, output.logits))
        self._release_if_finished(request)

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

        if not self._reserve_decode_slot(request):
            return

        token = torch.tensor(
            [[request.last_token]],
            dtype=torch.long,
            device=self.device,
        )

        output = self.runner.decode(token, request.kv_cache)

        request.append_token(self._sample(request, output.logits))
        self._release_if_finished(request)

    def execute(self, request: InferenceRequest) -> dict:
        """
        Drive one request from WAITING to FINISHED. Returns stats.
        Does not consult the scheduler; step() is what pairs the two.
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
            **self._kv_stats(request),
        }

    # --------------------------------------------------
    # Batched lifecycle steps (Phase 9)
    # --------------------------------------------------

    @torch.inference_mode()
    def prefill_batch(self, requests: list[InferenceRequest]):
        """
        WAITING -> PREFILLING -> DECODING (or FINISHED) for a group of
        requests, sharing one forward pass. Requests with nothing to
        generate finish without joining the batch.
        """

        for request in requests:
            if request.status != RequestStatus.WAITING:
                raise RuntimeError(
                    f"{request.request_id}: prefill requires WAITING, got {request.status.name}"
                )

        active = []

        for request in requests:
            reason = request.check_stop()

            if reason is not None:
                request.finish(reason)
            elif self._allocate(request):
                request.start_prefill()
                active.append(request)

        if not active:
            return

        with region("batch/build"):
            batch = build_prefill_batch(active, pad_token_id=self.pad_token_id, device=self.device)
        if self.paged_kv:
            output = self.runner.prefill_batch(batch, kv_caches=[r.kv_cache for r in active])
        else:
            # Let the runner build private, cloned per-row caches.
            output = self.runner.prefill_batch(batch)

            for row, request in enumerate(batch.requests):
                request.kv_cache = output.kv_caches[row]

        for row, request in enumerate(batch.requests):
            request.append_token(self._sample(request, output.logits[row:row + 1]))
            self._release_if_finished(request)

    @torch.inference_mode()
    def decode_batch(self, requests: list[InferenceRequest]):
        """
        One DECODING step for a group of requests, sharing one forward
        pass; each row is sampled with its own request's params.
        """

        for request in requests:
            if request.status != RequestStatus.DECODING:
                raise RuntimeError(
                    f"{request.request_id}: decode requires DECODING, got {request.status.name}"
                )

            if request.num_cached_tokens != request.position - 1:
                raise RuntimeError(
                    f"{request.request_id}: KV cache length {request.num_cached_tokens} "
                    f"does not match position {request.position} - 1"
                )

        if len({id(r.kv_cache) for r in requests}) != len(requests):
            raise RuntimeError("requests in one batch must not share a KV cache")

        # Requests the pool can't grow are aborted and drop out; the
        # rest still decode together.
        requests = [r for r in requests if self._reserve_decode_slot(r)]

        if not requests:
            return

        with region("batch/build"):
            batch = build_decode_batch(requests, device=self.device)
        output = self.runner.decode_batch(batch)

        for row, request in enumerate(batch.requests):
            request.append_token(self._sample(request, output.logits[row:row + 1]))
            self._release_if_finished(request)

    def execute_batch(self, requests: list[InferenceRequest]) -> list[dict]:
        """
        Drive a group of requests from WAITING to FINISHED together.
        Returns one stats dict per request, in input order. Timings
        are for the whole batch, since the requests share them.
        """

        self._sync()
        start = time.perf_counter()

        self.prefill_batch(requests)

        prefill_kv_lens = [r.num_cached_tokens for r in requests]

        self._sync()
        prefill_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        decode_steps = 0

        while True:
            active = [r for r in requests if r.status == RequestStatus.DECODING]

            if not active:
                break

            self.decode_batch(active)
            decode_steps += 1

        self._sync()
        decode_elapsed = time.perf_counter() - start

        elapsed = prefill_elapsed + decode_elapsed
        total_generated = sum(r.num_generated for r in requests)

        return [
            {
                "request_id": request.request_id,
                "finish_reason": request.finish_reason.value,
                "batch_size": len(requests),
                "decode_steps": decode_steps,
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
                "batch_tokens_per_second": (
                    total_generated / elapsed if elapsed > 0 else float("inf")
                ),
                **self._kv_stats(request),
            }
            for request, prefill_kv_len in zip(requests, prefill_kv_lens)
        ]

    # --------------------------------------------------
    # Scheduling (Phase 8)
    # --------------------------------------------------

    def submit(self, request: InferenceRequest) -> InferenceRequest:
        self.scheduler.add(request)
        return request

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> dict | None:
        """
        Ask the scheduler for the next request and execute it to
        completion. Returns its result, or None if nothing is waiting.
        """

        request = self.scheduler.next()

        if request is None:
            return None

        try:
            stats = self.execute(request)
        except Exception:
            self.scheduler.abort(request)
            self._release(request)
            raise

        self.scheduler.complete(request)

        result = self._result(request, stats)
        self._finished[request.request_id] = result

        return result

    def admission_check(self):
        """
        can_schedule callback for the scheduler: admit requests while
        the free pool can still hold their prompts. With paged_kv=False
        the pool isn't used, so memory never limits admission.
        """

        if not self.paged_kv:
            return None

        budget = [self.kv_cache_manager.num_free_blocks]

        def can_schedule(request: InferenceRequest) -> bool:
            need = self.kv_cache_manager.blocks_needed(request.prompt_len)

            if need > budget[0]:
                return False

            budget[0] -= need
            return True

        return can_schedule

    def step_batch(self, max_batch_size: int) -> list[dict]:
        """
        Ask the scheduler for up to `max_batch_size` requests and
        execute them together to completion. Returns their results in
        batch order (empty if nothing is waiting).
        """

        requests = self.scheduler.next_batch(
            max_batch_size,
            can_schedule=self.admission_check(),
        )

        if not requests:
            # The oldest request doesn't fit in the free pool even on
            # its own. Run it anyway so it goes through the normal
            # allocation path and is aborted cleanly (OUT_OF_KV_BLOCKS)
            # instead of blocking the queue forever.
            requests = self.scheduler.next_batch(1)

        if not requests:
            return []

        try:
            all_stats = self.execute_batch(requests)
        except Exception:
            for request in requests:
                self.scheduler.abort(request)
                self._release(request)
            raise

        results = []

        for request, stats in zip(requests, all_stats):
            self.scheduler.complete(request)

            result = self._result(request, stats)
            self._finished[request.request_id] = result
            results.append(result)

        return results

    def run_until_complete(self, max_batch_size: int = 1) -> list[dict]:
        """
        Execute every submitted request, in scheduler order. Returns
        their results in completion order.

        max_batch_size=1 is the Phase 8 sequential path (step);
        larger values run FIFO groups through step_batch.
        """

        results = []

        while self.scheduler.has_waiting():
            if max_batch_size == 1:
                step_results = [self.step()]
            else:
                step_results = self.step_batch(max_batch_size)

            for result in step_results:
                if result is not None:
                    results.append(self.pop_result(result["request"].request_id))

        return results

    def pop_result(self, request_id: str) -> dict:
        return self._finished.pop(request_id)

    # --------------------------------------------------
    # Convenience wrappers (Phase 6 API)
    # --------------------------------------------------

    def _result(self, request: InferenceRequest, stats: dict) -> dict:
        return {
            "request": request,
            "token_ids": request.all_tokens,
            "text": self.tokenizer.decode(request.all_tokens),
            "kv_cache": request.kv_cache,
            "stats": stats,
        }

    def _run_to_completion(self, request: InferenceRequest) -> dict:
        """
        Submit through the scheduler and step until this request is
        done. Requests submitted earlier run first (FIFO); their
        results stay available via pop_result().
        """

        self.submit(request)

        while request.request_id not in self._finished:
            if self.step() is None:
                raise RuntimeError(
                    f"{request.request_id} was dropped by the scheduler before running"
                )

        return self.pop_result(request.request_id)

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

        return self._run_to_completion(request)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        sampling_params: SamplingParams | None = None,
    ) -> dict:
        """
        prompt -> generated text, same result shape as
        ModelRunner.generate plus the prefill/decode stats and the
        finished request. Goes through the scheduler like any other
        request.
        """

        request = self.create_request(
            prompt,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )

        return self._run_to_completion(request)
