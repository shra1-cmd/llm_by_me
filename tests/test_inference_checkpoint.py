import pytest
import torch

from src.inference.checkpoint_loader import (
    find_latest_checkpoint,
    load_inference_checkpoint,
)
from src.tokenizer.tokenizer import BPETokenizer

CHECKPOINT_DIR = "checkpoints/v1"
TOKENIZER_PATH = "tokenizer/tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def latest_checkpoint_path():
    return find_latest_checkpoint(CHECKPOINT_DIR)


@pytest.fixture(scope="module")
def loaded_model(latest_checkpoint_path):
    return load_inference_checkpoint(latest_checkpoint_path, device=DEVICE)


def test_checkpoint_loads(loaded_model):
    model, metadata = loaded_model
    assert model is not None
    assert metadata["step"] is not None


def test_state_dict_matches_exactly(loaded_model):
    _, metadata = loaded_model
    assert metadata["missing_keys"] == []
    assert metadata["unexpected_keys"] == []


def test_tokenizer_vocab_matches_model(loaded_model):
    model, metadata = loaded_model
    tokenizer = BPETokenizer(TOKENIZER_PATH)
    assert tokenizer.vocab_size == metadata["model_config"].vocab_size


def test_forward_pass_produces_valid_logits(loaded_model):
    model, metadata = loaded_model
    tokenizer = BPETokenizer(TOKENIZER_PATH)

    prompt = "Hello, my name is"
    token_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    with torch.inference_mode():
        logits, _ = model(input_ids)

    vocab_size = metadata["model_config"].vocab_size

    assert tuple(logits.shape) == (1, len(token_ids), vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
