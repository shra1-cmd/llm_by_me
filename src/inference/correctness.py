"""
Phase 15: the correctness gate every optimization must pass.

An optimization is only kept if the engine still produces the same
results as the baseline. A Variant is "how to run the model": a
ModelRunner (eager or torch.compile'd model, dynamic or static KV
cache) plus the fast_paths flags active while it runs. The gate
collects the same measurements from the reference and the candidate
variant and compares them:

    1. prefill_logits     last-position logits after each prompt
    2. decode_logits      teacher-forced decode: both variants are fed
                          the reference's greedy tokens, so every step
                          sees identical inputs (errors can't compound)
    3. greedy_tokens      free-running greedy generation, token for token
    4. eos                with EOS set to the token the reference emits at
                          step k, both stop at the same step, EOS included
    5. kv_cache           within the candidate: cached decode logits vs a
                          full no-cache forward over the same tokens
    6. batched            prompts of different lengths through the
                          continuous-batching engine together: each
                          request's tokens equal its single-request run

Floating-point checks use max / mean absolute error with a tolerance
(fused or compiled kernels may reorder fp32 sums); token checks must
match exactly. Phase 5's KV-cache result (max 3.05e-5, mean 1.67e-6)
is the scale of error expected from a legitimate reordering.
"""

from dataclasses import asdict, dataclass, field

import torch

from src.inference.request import SamplingParams
from src.model import fast_paths


@dataclass(frozen=True)
class Tolerance:
    max_abs: float = 1e-3
    mean_abs: float = 1e-4


@dataclass
class Variant:
    name: str
    runner: object                      # ModelRunner
    flags: dict = field(default_factory=dict)

    def activate(self):
        return fast_paths.enabled(**self.flags)


@dataclass
class CheckResult:
    name: str
    passed: bool
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    detail: str = ""


@dataclass
class GateResult:
    reference: str
    candidate: str
    checks: list[CheckResult]
    tolerance: Tolerance

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def max_abs_error(self) -> float:
        return max((c.max_abs_error or 0.0) for c in self.checks)

    @property
    def mean_abs_error(self) -> float:
        return max((c.mean_abs_error or 0.0) for c in self.checks)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "candidate": self.candidate,
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "tolerance": asdict(self.tolerance),
            "checks": [asdict(c) for c in self.checks],
        }


# ======================================================================
# Running one variant
# ======================================================================


def _ids(tokens: list[int], device) -> torch.Tensor:
    return torch.tensor([tokens], dtype=torch.long, device=device)


@torch.inference_mode()
def teacher_forced(runner, prompt: list[int], forced: list[int]) -> torch.Tensor:
    """Logits [1 + len(forced), vocab]: after the prompt, then after each forced token."""

    out = runner.prefill(_ids(prompt, runner.device))
    logits = [out.logits[0].float()]
    cache = out.kv_cache

    for token in forced:
        step = runner.decode(_ids([token], runner.device), cache)
        logits.append(step.logits[0].float())

    return torch.stack(logits)


@torch.inference_mode()
def greedy(runner, prompt: list[int], max_new_tokens: int, eos_id: int | None = None) -> list[int]:
    """Greedy continuation (new tokens only); stops after emitting eos_id."""

    out = runner.prefill(_ids(prompt, runner.device))
    cache = out.kv_cache
    tokens: list[int] = []
    limit = runner.max_seq_len or (len(prompt) + max_new_tokens)

    while True:
        token = int(out.logits[0].argmax())
        tokens.append(token)

        if (
            len(tokens) >= max_new_tokens
            or token == eos_id
            or len(prompt) + len(tokens) >= limit
        ):
            return tokens

        out = runner.decode(_ids([token], runner.device), cache)


@torch.inference_mode()
def full_forward_last(runner, tokens: list[int]) -> torch.Tensor:
    """Last-position logits of one no-cache forward over all tokens."""

    logits, _ = runner.model(_ids(tokens, runner.device))
    return logits[0, -1].float()


def batched_greedy(runner, tokenizer, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """All prompts through the continuous-batching engine at once (greedy)."""

    from src.inference.continuous_batching import ContinuousBatchingEngine
    from src.inference.kv_cache_manager import KVCacheManager

    engine = ContinuousBatchingEngine(
        runner,
        tokenizer,
        SamplingParams(greedy=True).to_sampler(),
        device=runner.device,
        paged_kv=False,
        kv_cache_manager=KVCacheManager.for_model(runner.model, num_blocks=1, device=runner.device),
        max_batch_size=len(prompts),
    )

    for i, prompt in enumerate(prompts):
        engine.submit(engine.create_request_from_ids(
            prompt, max_new_tokens=max_new_tokens, request_id=f"gate-{i}",
        ))

    results = {r["request"].request_id: r for r in engine.run_until_complete()}

    return [
        results[f"gate-{i}"]["token_ids"][len(prompt):]
        for i, prompt in enumerate(prompts)
    ]


# ======================================================================
# The gate
# ======================================================================


def _errors(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.mean().item()


def _logit_check(name, ref, cand, tol: Tolerance, detail="") -> CheckResult:
    max_err, mean_err = _errors(ref, cand)
    return CheckResult(
        name=name,
        passed=max_err <= tol.max_abs and mean_err <= tol.mean_abs,
        max_abs_error=max_err,
        mean_abs_error=mean_err,
        detail=detail,
    )


def _token_check(name, ref: list, cand: list, detail="") -> CheckResult:
    if ref == cand:
        return CheckResult(name=name, passed=True, detail=detail)

    first = next((i for i, (a, b) in enumerate(zip(ref, cand)) if a != b), min(len(ref), len(cand)))
    return CheckResult(
        name=name,
        passed=False,
        detail=f"{detail} first mismatch at index {first}: "
               f"{ref[first:first + 3]} vs {cand[first:first + 3]} (len {len(ref)} vs {len(cand)})",
    )


def run_gate(
    reference: Variant,
    candidate: Variant,
    prompts: list[list[int]],
    tokenizer=None,
    max_new_tokens: int = 24,
    eos_step: int = 5,
    tolerance: Tolerance = Tolerance(),
    batched: bool = True,
    reference_cache: dict | None = None,
) -> GateResult:
    """
    Compare candidate against reference on `prompts` (token-id lists,
    ideally of different lengths). `tokenizer` is needed for the
    batched check. `reference_cache` (a dict) memoises the reference's
    outputs across calls, since the reference never changes.
    """

    ref = reference_cache if reference_cache is not None else {}

    if not ref:
        with reference.activate():
            ref["greedy"] = [greedy(reference.runner, p, max_new_tokens) for p in prompts]
            ref["forced"] = [
                teacher_forced(reference.runner, p, g[:-1])
                for p, g in zip(prompts, ref["greedy"])
            ]
            ref["eos_id"] = [g[min(eos_step, len(g) - 1)] for g in ref["greedy"]]
            ref["eos"] = [
                greedy(reference.runner, p, max_new_tokens, eos_id=e)
                for p, e in zip(prompts, ref["eos_id"])
            ]
            if batched and tokenizer is not None:
                ref["batched"] = batched_greedy(reference.runner, tokenizer, prompts, max_new_tokens)

    checks: list[CheckResult] = []

    with candidate.activate():
        for i, prompt in enumerate(prompts):
            tag = f"prompt {i} (len {len(prompt)})"
            forced = teacher_forced(candidate.runner, prompt, ref["greedy"][i][:-1])

            checks.append(_logit_check("prefill_logits", ref["forced"][i][:1], forced[:1], tolerance, tag))
            checks.append(_logit_check("decode_logits", ref["forced"][i][1:], forced[1:], tolerance,
                                       f"{tag}, {len(forced) - 1} steps"))

            tokens = greedy(candidate.runner, prompt, max_new_tokens)
            checks.append(_token_check("greedy_tokens", ref["greedy"][i], tokens, tag))

            eos = greedy(candidate.runner, prompt, max_new_tokens, eos_id=ref["eos_id"][i])
            checks.append(_token_check("eos", ref["eos"][i], eos,
                                       f"{tag}, eos={ref['eos_id'][i]} stops at {len(ref['eos'][i])}"))

            full = full_forward_last(candidate.runner, prompt + ref["greedy"][i][:-1])
            checks.append(_logit_check("kv_cache", full[None], forced[-1:], tolerance,
                                       f"{tag}: cached decode vs full recompute"))

        if batched and tokenizer is not None:
            together = batched_greedy(candidate.runner, tokenizer, prompts, max_new_tokens)
            for i, tokens in enumerate(together):
                checks.append(_token_check("batched", ref["batched"][i], tokens,
                                           f"prompt {i} of {len(prompts)} batched vs reference"))
                alone = greedy(candidate.runner, prompts[i], max_new_tokens)
                checks.append(_token_check("batched_independent", alone, tokens,
                                           f"prompt {i}: batched vs alone"))

    return GateResult(reference.name, candidate.name, checks, tolerance)

