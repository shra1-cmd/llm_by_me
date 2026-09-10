import torch

from configs.v1 import ModelConfig
from src.model.model import (
    V1LanguageModel,
    count_parameters,
)
from src.tokenizer.tokenizer import BPETokenizer


def get_model():

    tokenizer = BPETokenizer()

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
    )

    model = V1LanguageModel(config)

    return model, config


def test_model_forward():

    model, config = get_model()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 32),
    )

    logits, loss = model(
        input_ids,
        input_ids,
    )

    assert logits.shape == (
        2,
        32,
        config.vocab_size,
    )

    assert loss.ndim == 0


def test_model_without_targets():

    model, config = get_model()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 32),
    )

    logits, loss = model(input_ids)

    assert logits.shape == (
        2,
        32,
        config.vocab_size,
    )

    assert loss is None


def test_parameter_count():

    model, _ = get_model()

    num_params = count_parameters(model)

    print(
        f"\nModel parameters: "
        f"{num_params / 1e6:.2f}M"
    )

    # V1 is intended to be approximately 30M.
    assert 25_000_000 < num_params < 35_000_000


def test_gqa_configuration():

    model, config = get_model()

    attention = model.blocks[0].attention

    assert attention.num_q_heads == 8
    assert attention.num_kv_heads == 2

    assert (
        attention.head_dim
        == 64
    )

    assert (
        attention.num_groups
        == 4
    )