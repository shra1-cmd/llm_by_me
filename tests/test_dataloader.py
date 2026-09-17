import torch

from src.data.dataloader import (
    PackedSequenceDataset,
    create_dataloader,
)


def test_packed_sequence_dataset():
    tokens = list(range(100))

    dataset = PackedSequenceDataset(
        token_ids=tokens,
        seq_len=10,
    )

    x, y = dataset[0]

    assert x.shape == (10,)
    assert y.shape == (10,)

    assert torch.equal(
        x,
        torch.tensor(range(10)),
    )

    assert torch.equal(
        y,
        torch.tensor(range(1, 11)),
    )


def test_dataloader_shapes():
    loader = create_dataloader(
        split="train",
        seq_len=512,
        batch_size=2,
    )

    x, y = next(iter(loader))

    assert x.shape == (2, 512)
    assert y.shape == (2, 512)

    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_next_token_alignment():
    loader = create_dataloader(
        split="train",
        seq_len=512,
        batch_size=1,
    )

    x, y = next(iter(loader))

    assert torch.equal(
        x[:, 1:],
        y[:, :-1],
    )