from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 16_384
    max_seq_len: int = 512
    hidden_dim: int = 512
    num_layers: int = 8
    num_q_heads: int = 8
    num_kv_heads: int = 2
    ffn_dim: int = 1408
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6