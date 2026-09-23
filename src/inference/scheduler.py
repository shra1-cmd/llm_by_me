"""
Phase 8: FIFO scheduler.

The scheduler decides *which* request runs next. It never executes
the model, touches tensors or the KV cache — that is the engine's
and ModelRunner's job.

    add(A), add(B), add(C)

    waiting  = [A, B, C]      running = {}
    next()   -> A
    waiting  = [B, C]         running = {A}
    complete(A)
    waiting  = [B, C]         running = {}
    next()   -> B
    ...

Rules:
    - Only WAITING requests can be added.
    - A request can only be in the scheduler once (waiting or running);
      adding it again raises.
    - Requests that became FINISHED / ABORTED while still waiting are
      dropped by next(), never selected.
    - complete() requires a finished request and removes it from
      running.

next() selects one request (Phase 8). next_batch(n) selects up to n
of the oldest waiting requests to run together (Phase 9) — still
FIFO, still a fixed group: nothing joins a batch once it has
started. No priorities, token budgets or preemption yet.

Phase 10: next_batch takes an optional `can_schedule(request)` check,
so the engine can stop admitting requests once the KVCacheManager has
no room for their prompts. The scheduler just asks; it still knows
nothing about memory. Admission stays FIFO — if the oldest waiting
request doesn't fit, nothing behind it jumps the queue.
"""

from collections import deque
from collections.abc import Callable

from src.inference.request import FinishReason, InferenceRequest, RequestStatus


class Scheduler:
    def __init__(self):
        self.waiting: deque[InferenceRequest] = deque()
        self.running: dict[str, InferenceRequest] = {}

        # Every request_id currently waiting or running.
        self._active_ids: set[str] = set()

    # --------------------------------------------------
    # Queue state
    # --------------------------------------------------

    @property
    def num_waiting(self) -> int:
        return sum(1 for r in self.waiting if r.status == RequestStatus.WAITING)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def has_waiting(self) -> bool:
        return self.num_waiting > 0

    def has_work(self) -> bool:
        return self.has_waiting() or self.num_running > 0

    def __contains__(self, request: InferenceRequest) -> bool:
        return request.request_id in self._active_ids

    # --------------------------------------------------
    # Operations
    # --------------------------------------------------

    def add(self, request: InferenceRequest):
        if request.request_id in self._active_ids:
            raise ValueError(f"{request.request_id} is already scheduled")

        if request.status != RequestStatus.WAITING:
            raise ValueError(
                f"{request.request_id}: only WAITING requests can be added, "
                f"got {request.status.name}"
            )

        self.waiting.append(request)
        self._active_ids.add(request.request_id)

    def next(self) -> InferenceRequest | None:
        """
        Pop the oldest WAITING request and mark it running, or return
        None if nothing is waiting. Finished/aborted requests found at
        the head of the queue are discarded along the way.
        """

        while self.waiting:
            request = self.waiting.popleft()

            if request.status != RequestStatus.WAITING:
                self._active_ids.discard(request.request_id)
                continue

            self.running[request.request_id] = request
            return request

        return None

    def peek(self) -> InferenceRequest | None:
        """The request next() would return, without selecting it."""

        for request in self.waiting:
            if request.status == RequestStatus.WAITING:
                return request

        return None

    def next_batch(
        self,
        max_batch_size: int,
        can_schedule: Callable[[InferenceRequest], bool] | None = None,
    ) -> list[InferenceRequest]:
        """
        Up to `max_batch_size` WAITING requests, oldest first, all
        marked running. Empty list if nothing is waiting.

        can_schedule: called on the next candidate before selecting
        it; the batch stops at the first False (FIFO, no skipping).
        """

        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")

        batch = []

        while len(batch) < max_batch_size:
            candidate = self.peek()

            if candidate is None:
                break

            if can_schedule is not None and not can_schedule(candidate):
                break

            request = self.next()

            if request is None:
                break

            batch.append(request)

        return batch

    def complete(self, request: InferenceRequest):
        if request.request_id not in self.running:
            raise ValueError(f"{request.request_id} is not running")

        if not request.is_finished:
            raise ValueError(
                f"{request.request_id}: cannot complete a request in state "
                f"{request.status.name}"
            )

        del self.running[request.request_id]
        self._active_ids.discard(request.request_id)

    def abort(self, request: InferenceRequest, reason: FinishReason = FinishReason.ABORTED):
        """
        Abort a waiting or running request and drop it from the
        scheduler.
        """

        if request.request_id not in self._active_ids:
            raise ValueError(f"{request.request_id} is not scheduled")

        if not request.is_finished:
            request.abort(reason)

        if request.request_id in self.running:
            del self.running[request.request_id]
        else:
            self.waiting.remove(request)

        self._active_ids.discard(request.request_id)
