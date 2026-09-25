"""
Phase 16 — export the V1 checkpoint as a standard Llama model (for vLLM).

V1 is already a Llama-family decoder: GQA attention, RoPE, pre-RMSNorm,
SwiGLU MLP, no biases, tied input/output embeddings. vLLM (and HF
transformers) can therefore run it with their built-in LlamaForCausalLM,
with no custom model code. The one difference is how RoPE pairs the
head dimensions:

    V1 (src/model/rope.py)    rotates interleaved pairs (2i, 2i+1)
    HF / vLLM Llama           rotates halves (i, i + head_dim/2)

Reordering each head's rows of q_proj and k_proj from
[0, 2, 4, ..., D-2, 1, 3, ..., D-1] turns interleaved pairs into halves.
q·k is unchanged because both get the same permutation, and V /
out_proj are untouched. So the exported model computes the same
function (up to fp32 kernel noise).

    V1 checkpoint  ->  LlamaConfig + remapped/permuted state_dict
                   ->  verify with transformers: prefill logits and
                       greedy tokens vs our model
                   ->  save_pretrained(models/v1-llama)   (safetensors)

The tokenizer is our own BPE, which vLLM cannot load, so experiments
send token ids directly (vLLM: skip_tokenizer_init=True + TokensPrompt).

Usage:
    python profiles/phase16/export_llama.py
    python profiles/phase16/export_llama.py --out models/v1-llama --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.checkpoint_loader import (  # noqa: E402
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.tokenizer.tokenizer import BPETokenizer  # noqa: E402


def rope_permutation(head_dim: int) -> torch.Tensor:
    """Interleaved-pair order -> half-split order: [0, 2, ..., D-2, 1, 3, ..., D-1]."""

    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


def permute_heads(weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder the output rows of a [num_heads*head_dim, in] projection within each head."""

    perm = rope_permutation(head_dim)
    w = weight.view(num_heads, head_dim, -1)[:, perm, :]
    return w.reshape(num_heads * head_dim, -1).contiguous()


def llama_config(cfg, tokenizer: BPETokenizer):
    from transformers import LlamaConfig

    head_dim = cfg.hidden_dim // cfg.num_q_heads
    eos = tokenizer.token_to_id("<eos>")
    pad = tokenizer.token_to_id("<pad>")
    bos = tokenizer.token_to_id("<bos>")

    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_dim,
        intermediate_size=cfg.ffn_dim,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_q_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        torch_dtype="float32",
        eos_token_id=eos,
        pad_token_id=pad,
        bos_token_id=bos,
    )


def llama_state_dict(model) -> dict[str, torch.Tensor]:
    cfg = model.config
    head_dim = cfg.hidden_dim // cfg.num_q_heads
    sd = {"model.embed_tokens.weight": model.token_embedding.weight,
          "model.norm.weight": model.final_norm.weight}

    for i, block in enumerate(model.blocks):
        p = f"model.layers.{i}."
        attn, ffn = block.attention, block.ffn
        sd[p + "input_layernorm.weight"] = block.attn_norm.weight
        sd[p + "post_attention_layernorm.weight"] = block.ffn_norm.weight
        sd[p + "self_attn.q_proj.weight"] = permute_heads(attn.q_proj.weight, cfg.num_q_heads, head_dim)
        sd[p + "self_attn.k_proj.weight"] = permute_heads(attn.k_proj.weight, cfg.num_kv_heads, head_dim)
        sd[p + "self_attn.v_proj.weight"] = attn.v_proj.weight
        sd[p + "self_attn.o_proj.weight"] = attn.out_proj.weight
        sd[p + "mlp.gate_proj.weight"] = ffn.gate_proj.weight
        sd[p + "mlp.up_proj.weight"] = ffn.up_proj.weight
        sd[p + "mlp.down_proj.weight"] = ffn.down_proj.weight

    return {k: v.detach().clone().contiguous() for k, v in sd.items()}


def build_llama(model, tokenizer):
    from transformers import LlamaForCausalLM

    llama = LlamaForCausalLM(llama_config(model.config, tokenizer))
    missing, unexpected = llama.load_state_dict(llama_state_dict(model), strict=False)
    # lm_head.weight is tied to embed_tokens, so it may be reported missing
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    llama.tie_weights()
    return llama.to(next(model.parameters()).device).eval()


@torch.inference_mode()
def verify(model, llama, prompt_ids: list[int], new_tokens: int = 32) -> dict:
    """Prefill logits and greedy continuation: our model vs the exported Llama."""

    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=device)

    ours, _ = model(ids)
    theirs = llama(ids).logits
    diff = (ours - theirs).abs()

    def greedy(fn):
        seq = ids
        for _ in range(new_tokens):
            nxt = fn(seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[0, len(prompt_ids):].tolist()

    ours_tokens = greedy(lambda s: model(s)[0])
    llama_tokens = greedy(lambda s: llama(s).logits)

    return {
        "prompt_len": len(prompt_ids),
        "max_abs_logit_error": diff.max().item(),
        "mean_abs_logit_error": diff.mean().item(),
        "greedy_tokens_match": ours_tokens == llama_tokens,
        "greedy_tokens": ours_tokens,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/v1")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--out", type=str, default="models/v1-llama")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint or find_latest_checkpoint(ROOT / args.checkpoint_dir)
    model, _ = load_inference_checkpoint(checkpoint, device=args.device)
    tokenizer = BPETokenizer(str(ROOT / args.tokenizer_path))

    llama = build_llama(model, tokenizer)
    prompt = tokenizer.encode("Once upon a time, there was a little girl named Lily. She loved to")
    report = verify(model, llama, prompt)
    report["checkpoint"] = str(checkpoint)

    print(f"max |Δlogit| {report['max_abs_logit_error']:.2e}, mean {report['mean_abs_logit_error']:.2e}, "
          f"greedy tokens match: {report['greedy_tokens_match']}")
    if report["max_abs_logit_error"] > args.tolerance or not report["greedy_tokens_match"]:
        sys.exit("export does not reproduce the V1 model; not saving")

    out = ROOT / args.out
    llama.save_pretrained(out, safe_serialization=True)
    (out / "v1_export.json").write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
