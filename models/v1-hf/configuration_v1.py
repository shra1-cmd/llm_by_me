"""
Phase 2: Hugging Face-compatible architecture description for V1.

configuration_v1.py describes WHAT the model is (shapes, sizes).
It holds no learned weights.
"""

from transformers import PretrainedConfig

from configs.v1 import ModelConfig


class V1Config(PretrainedConfig):

    model_type = "v1"

    def __init__(
        self,
        vocab_size: int = 16_384,
        max_seq_len: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        ffn_dim: int = 1408,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps

        super().__init__(**kwargs)

    def to_model_config(self) -> ModelConfig:
        """
        V1Config (HF-facing)
            ->
        ModelConfig (our native dataclass, understood by V1LanguageModel)
        """

        return ModelConfig(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            rms_norm_eps=self.rms_norm_eps,
        )

    @classmethod
    def from_model_config(
        cls,
        config: ModelConfig,
        **kwargs,
    ) -> "V1Config":
        """
        ModelConfig (our native dataclass)
            ->
        V1Config (HF-facing)
        """

        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_q_heads=config.num_q_heads,
            num_kv_heads=config.num_kv_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )
