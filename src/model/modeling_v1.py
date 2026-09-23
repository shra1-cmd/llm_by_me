"""
Phase 2: Hugging Face-compatible causal-LM wrapper around V1LanguageModel.

    HF interface (PreTrainedModel / CausalLMOutput)
            |
     V1ForCausalLM
            |
     V1LanguageModel  (unchanged, from src/model/model.py)

No new computation is introduced here. This wraps the existing
forward pass so it can be used through the standard
`model(input_ids).logits` convention.

Limitation (unchanged from V1LanguageModel): attention is always
causal with no padding mask, so `attention_mask` is accepted for
interface compatibility but not applied. Batches must be built
without padding for now.
"""

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from src.model.configuration_v1 import V1Config
from src.model.model import V1LanguageModel


class V1ForCausalLM(PreTrainedModel):

    config_class = V1Config
    base_model_prefix = "model"

    def __init__(self, config: V1Config):
        super().__init__(config)

        self.model = V1LanguageModel(config.to_model_config())

        self.post_init()

    def _init_weights(self, module):
        # Weights are always loaded from a trained checkpoint via
        # export_hf.py; random init here is never actually used.
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        """
        input_ids:
            [B, T]

        returns CausalLMOutput with:
            logits [B, T, vocab_size]
            loss   scalar, only if labels is given
        """

        logits, loss = self.model(
            input_ids,
            targets=labels,
        )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    def get_input_embeddings(self):
        return self.model.token_embedding

    def set_input_embeddings(self, value):
        self.model.token_embedding = value

    def get_output_embeddings(self):
        # Tied LM head: logits = hidden @ token_embedding.weight.T
        return None
