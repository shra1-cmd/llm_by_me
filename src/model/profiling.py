"""
Phase 13: named profiler regions that cost nothing when switched off.

The model and the inference engine wrap their major sections in

    with region("attention/sdpa"):
        ...

so a torch.profiler trace shows our structure (embedding, layer_3,
attention/qkv_proj, mlp, lm_head, sampling, kv_cache/...) instead of
one enormous forward() full of anonymous aten:: ops.

torch.profiler.record_function is not free — it enters the
dispatcher's RecordFunction machinery even when no profiler is
running. With ~80 regions per forward and decode steps of ~3 ms,
leaving it on permanently would noticeably slow normal inference and
skew the Phase 12 benchmarks. So regions are OFF by default and
region() returns a shared no-op context manager; profiling code turns
them on explicitly:

    from src.model import profiling
    with profiling.enabled():
        with torch.profiler.profile(...) as prof:
            model(...)

Region names are deliberately layer-independent for leaf sections
("attention/sdpa", "mlp") so the profiler's key_averages() sums them
over all layers; each layer is additionally wrapped in "layer_<i>".
"""

from contextlib import contextmanager, nullcontext

import torch

_ENABLED = False
_NULL = nullcontext()


def is_enabled() -> bool:
    return _ENABLED


def set_enabled(value: bool):
    global _ENABLED
    _ENABLED = bool(value)


@contextmanager
def enabled(value: bool = True):
    """Temporarily switch regions on (or off) for a block."""

    previous = _ENABLED
    set_enabled(value)

    try:
        yield
    finally:
        set_enabled(previous)


def region(name: str):
    """A torch.profiler.record_function(name) when enabled, else a no-op."""

    if _ENABLED:
        return torch.profiler.record_function(name)

    return _NULL
