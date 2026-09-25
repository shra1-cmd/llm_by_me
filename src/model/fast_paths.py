"""
Phase 15: opt-in fast paths for inference, all OFF by default.

Phase 13/14 showed that decode on this model is launch-bound (~424
tiny kernels per step) and that some work is done that the output
never needs. Each flag below removes kernels or work in one place.
With every flag off, the model runs exactly the Phase 3-14 code path,
so the baseline stays reproducible and every optimization can be
measured, and kept or rejected, on its own (profiles/phase15/).

    rope_cache         precomputed, already-interleaved cos/sin tables:
                       drops 4 repeat_interleave kernels per layer.
                       Bit-exact (same values, same arithmetic).
    fused_rmsnorm      F.rms_norm (one fused kernel) instead of
                       pow / mean / add / rsqrt / mul / mul.
    decode_no_mask     a single new query (T=1, no padding mask) may
                       attend to every cached key, so skip building
                       the arange/compare mask and call SDPA unmasked.
    sdpa_gqa           SDPA's enable_gqa=True instead of materialising
                       repeat_interleave'd K/V copies for GQA.
    fused_qkv          one [hidden, q+k+v] matmul instead of three.
    fused_gate_up      one [hidden, 2*ffn] matmul instead of two.
    last_token_logits  when nothing needs the other positions (no
                       targets, no padding mask), run the final norm
                       and LM head on the last position only.

fused_qkv / fused_gate_up need prepare(model) first: it concatenates
the weights into one buffer and turns the original Linear weights
into views of it, so no memory is duplicated and both paths always
see the same values.

Usage:

    from src.model import fast_paths

    fast_paths.prepare(model)                     # once, for the fused_* flags
    with fast_paths.enabled(rope_cache=True, fused_qkv=True):
        model(...)                                # flags not named are OFF

Flags are process-global, like the Phase 13 profiler regions.
torch.compile guards on them, so changing them triggers a recompile.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace

import torch


@dataclass(frozen=True)
class FastPaths:
    rope_cache: bool = False
    fused_rmsnorm: bool = False
    decode_no_mask: bool = False
    sdpa_gqa: bool = False
    fused_qkv: bool = False
    fused_gate_up: bool = False
    last_token_logits: bool = False

    def active(self) -> list[str]:
        return [name for name, on in asdict(self).items() if on]


NAMES = [f.name for f in fields(FastPaths)]

FLAGS = FastPaths()


def set_flags(**flags) -> FastPaths:
    """Replace the whole flag set: named flags on/off, the rest OFF."""

    global FLAGS

    unknown = set(flags) - set(NAMES)
    if unknown:
        raise ValueError(f"unknown fast paths: {sorted(unknown)}")

    FLAGS = replace(FastPaths(), **flags)
    return FLAGS


@contextmanager
def enabled(**flags):
    """Temporarily run with exactly these flags on (others off)."""

    global FLAGS

    previous = FLAGS
    set_flags(**flags)

    try:
        yield FLAGS
    finally:
        FLAGS = previous


def fuse_linears(owner: torch.nn.Module, buffer_name: str, linears: list[torch.nn.Linear]):
    """
    Concatenate the linears' weights (along out_features) into one
    non-persistent buffer on `owner`, and make each Linear's weight a
    view into it. Idempotent.
    """

    if getattr(owner, buffer_name, None) is not None:
        return

    with torch.no_grad():
        fused = torch.cat([linear.weight.detach() for linear in linears], dim=0).contiguous()

    owner.register_buffer(buffer_name, fused, persistent=False)

    offset = 0
    for linear in linears:
        rows = linear.weight.shape[0]
        linear.weight = torch.nn.Parameter(
            fused[offset:offset + rows],
            requires_grad=linear.weight.requires_grad,
        )
        offset += rows


def prepare(model: torch.nn.Module) -> torch.nn.Module:
    """Build the fused weight buffers used by fused_qkv / fused_gate_up."""

    for module in model.modules():
        hook = getattr(module, "prepare_fast_paths", None)
        if hook is not None:
            hook()

    return model
