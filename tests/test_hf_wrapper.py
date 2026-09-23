import pytest
import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.model.configuration_v1 import V1Config
from src.model.modeling_v1 import V1ForCausalLM
from src.tokenizer.tokenizer import BPETokenizer

CHECKPOINT_DIR = "checkpoints/v1"
TOKENIZER_PATH = "tokenizer/tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def original_model_and_metadata():
    checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
    return load_inference_checkpoint(checkpoint_path, device=DEVICE)


@pytest.fixture(scope="module")
def hf_model(original_model_and_metadata):
    original_model, metadata = original_model_and_metadata

    hf_config = V1Config.from_model_config(metadata["model_config"])
    model = V1ForCausalLM(hf_config)

    remapped_state_dict = {
        f"model.{key}": value
        for key, value in original_model.state_dict().items()
    }

    load_result = model.load_state_dict(remapped_state_dict, strict=False)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []

    model.to(DEVICE)
    model.eval()

    return model


def test_config_round_trip(original_model_and_metadata):
    _, metadata = original_model_and_metadata
    model_config = metadata["model_config"]

    hf_config = V1Config.from_model_config(model_config)
    round_tripped = hf_config.to_model_config()

    assert round_tripped == model_config


def test_hf_model_output_has_logits_attribute(hf_model):
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=DEVICE)

    with torch.inference_mode():
        outputs = hf_model(input_ids)

    assert hasattr(outputs, "logits")
    assert outputs.logits.shape == (1, 3, hf_model.config.vocab_size)


def test_hf_wrapper_matches_original_logits(original_model_and_metadata, hf_model):
    original_model, _ = original_model_and_metadata
    tokenizer = BPETokenizer(TOKENIZER_PATH)

    for prompt in ["Hello, my name is", "The quick brown fox"]:
        token_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

        with torch.inference_mode():
            logits_original, _ = original_model(input_ids)
            logits_hf = hf_model(input_ids).logits

        assert torch.allclose(logits_original, logits_hf, atol=1e-5)


def test_hf_wrapper_no_nan_or_inf(hf_model):
    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long, device=DEVICE)

    with torch.inference_mode():
        logits = hf_model(input_ids).logits

    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
