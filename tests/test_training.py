import torch

from configs.v1 import ModelConfig
from configs.train import TrainingConfig

from src.model.model import V1LanguageModel
from src.training.optimizer import build_optimizer
from src.training.scheduler import (
    get_learning_rate,
)

from pathlib import Path

# import torch

# from configs.v1 import ModelConfig
# from configs.train import TrainingConfig

# from src.model.model import V1LanguageModel

# from src.training.optimizer import build_optimizer

from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def test_checkpoint_round_trip(
    tmp_path: Path,
):
    config = ModelConfig(
        vocab_size=1000,
    )

    training_config = TrainingConfig()

    model = V1LanguageModel(
        config
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    _, loss = model(
        input_ids,
        targets,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=10,
        loss=loss.item(),
        config=config,
        training_config=training_config,
    )

    new_model = V1LanguageModel(
        config
    )

    new_optimizer = build_optimizer(
        model=new_model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
    )

    assert checkpoint["step"] == 10

    assert (
        checkpoint["loss"]
        == loss.item()
    )

    for p1, p2 in zip(
        model.parameters(),
        new_model.parameters(),
    ):
        assert torch.equal(
            p1,
            p2,
        )

def test_optimizer():

    config = ModelConfig(
        vocab_size=1000,
    )

    model = V1LanguageModel(config)

    optimizer = build_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    assert len(optimizer.param_groups) == 2

    assert (
        optimizer.param_groups[0]["weight_decay"]
        == 0.1
    )

    assert (
        optimizer.param_groups[1]["weight_decay"]
        == 0.0
    )


def test_learning_rate_warmup():

    lr = get_learning_rate(
        step=0,
        max_steps=1000,
        warmup_steps=100,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
    )

    assert lr > 0
    assert lr < 3e-4


def test_learning_rate_reaches_max():

    lr = get_learning_rate(
        step=99,
        max_steps=1000,
        warmup_steps=100,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
    )

    assert lr <= 3e-4


def test_learning_rate_decay():

    lr = get_learning_rate(
        step=1000,
        max_steps=1000,
        warmup_steps=100,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
    )

    assert abs(lr - 3e-5) < 1e-8


def test_training_step():

    config = ModelConfig(
        vocab_size=1000,
    )

    model = V1LanguageModel(config)

    optimizer = build_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    _, loss = model(
        input_ids,
        targets,
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    assert torch.isfinite(loss)