"""
Phase 11: continuous batching.

Phase 9's static batching picks a group, runs it until *every* member
finishes, and only then looks at the queue again. Requests that finish
early leave empty slots that nobody fills:

    static      [A B C] -> [  B C] -> [  B  ] -> [  B  ] -> next batch

Continuous batching re-decides batch membership on every step: a
finished request leaves immediately, its KV blocks go back to the
pool, and a waiting request takes its slot on the very next step:

    continuous  [A B C] -> [B C D] -> [B D E] -> [B E F] -> ...

One call to ContinuousBatchingEngine.step() is one iteration of the
serving loop:

    1. ADMIT     free slots = max_batch_size - |active|
                 take waiting requests FIFO while there are free slots
                 and the KV pool passes a one-step lookahead:
                   reserve what each active request needs for its next
                   decode (blocks for `position` tokens), then admit a
                   newcomer only if its prompt + first generated token
                   fit in what is left
    2. PREFILL   newly admitted requests, in one batched forward
                 -> each gets its KV blocks and its first token
    3. DECODE    requests that were already active, one batched
                 forward, one new token each
    4. RETIRE    finished requests (EOS / max_new_tokens / max_seq_len /
                 out of KV blocks) leave the active set; their blocks
                 were already released by the engine, so the next
                 step's ADMIT can reuse them

Prefill and decode are two separate forward passes inside the same
step (new requests need a [B, T] prompt pass, active ones a [B, 1]
token pass). A request prefilled in step k is decoded from step k+1.

Everything below the loop is reused unchanged from earlier phases:
InferenceRequest (Phase 7), the FIFO Scheduler whose `running` dict
*is* the active set (Phase 8), batched prefill/decode with per-row
sampling (Phase 9) and the KVCacheManager block lifecycle (Phase 10).
Each request's tokens therefore match running it alone.

Every step is recorded as a StepRecord (who was admitted, decoded,
finished, who is active after), so batch membership over time can be
inspected and tested.

Not implemented (later phases): chunked prefill / mixing prefill and
decode rows in one forward, priorities, preemption, a PagedAttention
kernel.
"""

import time
from dataclasses import dataclass

from src.inference.inference_engine import InferenceEngine
from src.inference.request import InferenceRequest, RequestStatus


@dataclass
class StepRecord:
    step: int
    admitted: list[str]          # prefilled this step
    decoded: list[str]           # decoded this step
    finished: list[str]          # left the active set this step
    active_after: list[str]      # active set at the end of the step
    num_waiting_after: int
    free_blocks_after: int


class ContinuousBatchingEngine(InferenceEngine):
    def __init__(self, *args, max_batch_size: int = 4, **kwargs):
        super().__init__(*args, **kwargs)

        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")

        self.max_batch_size = max_batch_size
        self.history: list[StepRecord] = []

        self._step_count = 0
        self._submit_time: dict[str, float] = {}
        self._first_token_time: dict[str, float] = {}
        self._admitted_step: dict[str, int] = {}

    # --------------------------------------------------
    # State
    # --------------------------------------------------

    @property
    def active(self) -> list[InferenceRequest]:
        """Requests currently holding a batch slot, in admission order."""

        return list(self.scheduler.running.values())

    @property
    def waiting(self) -> list[InferenceRequest]:
        return [r for r in self.scheduler.waiting if r.status == RequestStatus.WAITING]

    def submit(self, request: InferenceRequest) -> InferenceRequest:
        super().submit(request)
        self._submit_time[request.request_id] = time.perf_counter()
        return request

    # --------------------------------------------------
    # Admission
    # --------------------------------------------------

    def _continuous_admission_check(self):
        """
        One-step KV lookahead for admission.

        Every active request will write one more token at its next
        decode (cache grows to `position`), which may need a new
        block; that is reserved first. A newcomer then needs blocks
        for its prompt now plus room for its first generated token at
        its first decode: prompt_len + 1 tokens.

        This keeps admission from pushing an already-running request
        into OUT_OF_KV_BLOCKS on the very next step. Longer-horizon
        shortages can still abort a request later; avoiding that
        needs preemption, which is out of scope here.
        """

        manager = self.kv_cache_manager

        reserved = sum(
            max(0, manager.blocks_needed(r.position) - len(r.block_ids))
            for r in self.active
            if r.status == RequestStatus.DECODING
        )

        budget = [manager.num_free_blocks - reserved]

        def can_schedule(request: InferenceRequest) -> bool:
            need = manager.blocks_needed(request.prompt_len + 1)

            if need > budget[0]:
                return False

            budget[0] -= need
            return True

        return can_schedule

    def _admit(self) -> list[InferenceRequest]:
        num_active = len(self.scheduler.running)
        free_slots = self.max_batch_size - num_active

        if free_slots <= 0:
            return []

        admitted = self.scheduler.next_batch(
            free_slots,
            can_schedule=self._continuous_admission_check(),
        )

        if not admitted and num_active == 0 and self.scheduler.has_waiting():
            # Nothing is running, so nothing will ever free blocks: the
            # oldest request can't fit even in an empty pool. Admit it
            # anyway so prefill aborts it cleanly (OUT_OF_KV_BLOCKS)
            # instead of stalling the queue.
            admitted = self.scheduler.next_batch(1)

        return admitted

    # --------------------------------------------------
    # One serving-loop iteration
    # --------------------------------------------------

    def step(self) -> list[dict]:
        """
        ADMIT -> PREFILL -> DECODE -> RETIRE. Returns the results of
        requests that finished during this step (possibly empty).
        """

        self._step_count += 1

        # Requests that were already active before this step decode;
        # requests admitted now only prefill.
        decoding = [r for r in self.active if r.status == RequestStatus.DECODING]
        admitted = self._admit()

        try:
            if admitted:
                self.prefill_batch(admitted)

                now = time.perf_counter()
                for request in admitted:
                    self._admitted_step[request.request_id] = self._step_count
                    if request.num_generated > 0:
                        self._first_token_time[request.request_id] = now

            if decoding:
                self.decode_batch(decoding)
        except Exception:
            for request in self.active:
                self.scheduler.abort(request)
                self._release(request)
            raise

        finished_results = []
        now = time.perf_counter()

        for request in self.active:
            if request.is_finished:
                self._release(request)
                self.scheduler.complete(request)

                result = self._result(request, self._continuous_stats(request, now))
                self._finished[request.request_id] = result
                finished_results.append(result)

        self.history.append(
            StepRecord(
                step=self._step_count,
                admitted=[r.request_id for r in admitted],
                decoded=[r.request_id for r in decoding],
                finished=[r["request"].request_id for r in finished_results],
                active_after=[r.request_id for r in self.active],
                num_waiting_after=self.scheduler.num_waiting,
                free_blocks_after=self.kv_cache_manager.num_free_blocks,
            )
        )

        return finished_results

    def _continuous_stats(self, request: InferenceRequest, finish_time: float) -> dict:
        submitted = self._submit_time.get(request.request_id, finish_time)
        first_token = self._first_token_time.get(request.request_id)

        return {
            "request_id": request.request_id,
            "finish_reason": request.finish_reason.value,
            "prompt_tokens": request.prompt_len,
            "generated_tokens": request.num_generated,
            "total_tokens": request.position,
            "final_kv_length": request.num_cached_tokens,
            "admitted_step": self._admitted_step.get(request.request_id),
            "finished_step": self._step_count,
            "latency_seconds": finish_time - submitted,
            "ttft_seconds": (first_token - submitted) if first_token is not None else None,
            **self._kv_stats(request),
        }

    # --------------------------------------------------
    # Drivers
    # --------------------------------------------------

    def run_until_complete(self, max_batch_size: int | None = None) -> list[dict]:
        """
        Step until nothing is waiting or active. Returns results in
        completion order. `max_batch_size` here only overrides the
        engine's slot count for this call.
        """

        previous = self.max_batch_size

        if max_batch_size is not None:
            self.max_batch_size = max_batch_size

        try:
            results = []

            while self.scheduler.has_work():
                for result in self.step():
                    results.append(self.pop_result(result["request"].request_id))

            return results
        finally:
            self.max_batch_size = previous

    def _run_to_completion(self, request: InferenceRequest) -> dict:
        """generate() support: step the whole loop until `request` is done."""

        self.submit(request)

        while request.request_id not in self._finished:
            if not self.scheduler.has_work():
                raise RuntimeError(f"{request.request_id} was dropped before finishing")

            self.step()

        return self.pop_result(request.request_id)

    def step_batch(self, max_batch_size: int) -> list[dict]:
        raise NotImplementedError(
            "ContinuousBatchingEngine replaces static step_batch; use step()"
        )

    def execute(self, request: InferenceRequest) -> dict:
        raise NotImplementedError(
            "ContinuousBatchingEngine runs requests through step(); use submit() + step()"
        )
