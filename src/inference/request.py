"""
Phase 7: request abstraction.

One InferenceRequest holds all the state needed to run (and later
pause / resume / schedule) a single generation:

    InferenceRequest
    ├── request_id
    ├── prompt
    ├── input_tokens          prompt token ids
    ├── generated_tokens      grows by one per sampled token
    ├── sampling_params       SamplingParams (owned by this request)
    ├── max_new_tokens
    ├── kv_cache              this request's KV handle (Phase 10: a
    │                         PagedKVCache backed by KVCacheManager
    │                         blocks; block_ids shows which ones)
    ├── status                RequestStatus
    └── finish_reason         FinishReason once FINISHED / ABORTED

Lifecycle:

    WAITING ──► PREFILLING ──► DECODING ──► FINISHED
       │             │             │
       └─────────────┴─────────────┴──────► FINISHED / ABORTED

WAITING can jump straight to FINISHED when there is nothing to
generate (max_new_tokens == 0, or the prompt already fills
max_seq_len). PREFILLING goes straight to FINISHED when the very
first sampled token already completes the request.

Phase 10 adds FinishReason.OUT_OF_KV_BLOCKS: a request is aborted
with that reason when the KVCacheManager cannot give it (more)
blocks, instead of corrupting another request's memory.
"""

import itertools
from dataclasses import dataclass, field
from enum import Enum

from src.inference.kv_cache import KVCache
from src.inference.sampler import Sampler


@dataclass(frozen=True)
class SamplingParams:
    greedy: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    repeat_ngram_size: int = 0

    def __post_init__(self):
        # Reuse the Sampler's validation so bad params fail at request
        # creation, not halfway through generation.
        self.to_sampler()

    def to_sampler(self) -> Sampler:
        return Sampler(
            greedy=self.greedy,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            repeat_ngram_size=self.repeat_ngram_size,
        )

    @classmethod
    def from_sampler(cls, sampler: Sampler) -> "SamplingParams":
        return cls(
            greedy=sampler.greedy,
            temperature=sampler.temperature,
            top_k=sampler.top_k,
            top_p=sampler.top_p,
            repetition_penalty=sampler.repetition_penalty,
            repeat_ngram_size=sampler.repeat_ngram_size,
        )


class RequestStatus(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    ABORTED = "aborted"


class FinishReason(Enum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"
    MAX_SEQ_LEN = "max_seq_len"
    ABORTED = "aborted"
    OUT_OF_KV_BLOCKS = "out_of_kv_blocks"


_ALLOWED_TRANSITIONS = {
    RequestStatus.WAITING: {
        RequestStatus.PREFILLING,
        RequestStatus.FINISHED,
        RequestStatus.ABORTED,
    },
    RequestStatus.PREFILLING: {
        RequestStatus.DECODING,
        RequestStatus.FINISHED,
        RequestStatus.ABORTED,
    },
    RequestStatus.DECODING: {
        RequestStatus.FINISHED,
        RequestStatus.ABORTED,
    },
    RequestStatus.FINISHED: set(),
    RequestStatus.ABORTED: set(),
}


_request_counter = itertools.count(1)


def next_request_id() -> str:
    return f"req_{next(_request_counter):03d}"


@dataclass
class InferenceRequest:
    prompt: str
    input_tokens: list[int]
    max_new_tokens: int = 50
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    eos_token_id: int | None = None
    max_seq_len: int | None = None
    request_id: str = field(default_factory=next_request_id)

    generated_tokens: list[int] = field(default_factory=list)
    kv_cache: KVCache | None = None
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: FinishReason | None = None

    def __post_init__(self):
        if len(self.input_tokens) == 0:
            raise ValueError("request must contain at least one input token")

        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

        # Own a copy so the caller's list can't mutate request state.
        self.input_tokens = list(self.input_tokens)

    # --------------------------------------------------
    # Token state
    # --------------------------------------------------

    @property
    def prompt_len(self) -> int:
        return len(self.input_tokens)

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def all_tokens(self) -> list[int]:
        return self.input_tokens + self.generated_tokens

    @property
    def position(self) -> int:
        """
        Total sequence length so far (prompt + generated). The next
        sampled token will sit at this absolute position.
        """

        return self.prompt_len + self.num_generated

    @property
    def last_token(self) -> int:
        return self.all_tokens[-1]

    @property
    def num_cached_tokens(self) -> int:
        return self.kv_cache.get_seq_length() if self.kv_cache is not None else 0

    @property
    def block_ids(self) -> list[int]:
        """KV blocks currently held (Phase 10 paged caches only)."""

        return list(getattr(self.kv_cache, "block_ids", []))

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def _transition(self, new_status: RequestStatus):
        if new_status not in _ALLOWED_TRANSITIONS[self.status]:
            raise RuntimeError(
                f"{self.request_id}: invalid transition "
                f"{self.status.name} -> {new_status.name}"
            )

        self.status = new_status

    def start_prefill(self):
        self._transition(RequestStatus.PREFILLING)

    def start_decode(self):
        self._transition(RequestStatus.DECODING)

    def finish(self, reason: FinishReason):
        self._transition(RequestStatus.FINISHED)
        self.finish_reason = reason

    def abort(self, reason: FinishReason = FinishReason.ABORTED):
        self._transition(RequestStatus.ABORTED)
        self.finish_reason = reason

    def check_stop(self) -> FinishReason | None:
        """
        Why this request should stop now, or None to keep going.
        Same rules (and order) as the Phase 3/6 loops.
        """

        if (
            self.eos_token_id is not None
            and self.num_generated > 0
            and self.generated_tokens[-1] == self.eos_token_id
        ):
            return FinishReason.EOS

        if self.num_generated >= self.max_new_tokens:
            return FinishReason.MAX_NEW_TOKENS

        if self.max_seq_len is not None and self.position >= self.max_seq_len:
            return FinishReason.MAX_SEQ_LEN

        return None

    def append_token(self, token_id: int):
        """
        Record one sampled token, then either finish the request or
        move it into DECODING.
        """

        if self.status not in (RequestStatus.PREFILLING, RequestStatus.DECODING):
            raise RuntimeError(
                f"{self.request_id}: cannot append token in state {self.status.name}"
            )

        self.generated_tokens.append(int(token_id))

        reason = self.check_stop()

        if reason is not None:
            self.finish(reason)
        elif self.status == RequestStatus.PREFILLING:
            self.start_decode()
