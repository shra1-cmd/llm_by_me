from pathlib import Path

import numpy as np
import torch

from src.data.bin_dataset import (
    MemmapTokenDataset,
)


def test_memmap_dataset(tmp_path: Path):

    path = tmp_path / "tokens.bin"

    tokens = np.arange(
        1000,
        dtype=np.uint16,
    )

    tokens.tofile(path)

    dataset = MemmapTokenDataset(
        path=path,
        seq_len=10,
    )

    assert len(dataset) > 0

    x, y = dataset[0]

    assert x.shape == (10,)
    assert y.shape == (10,)

    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    assert torch.equal(
        x,
        torch.arange(10),
    )

    assert torch.equal(
        y,
        torch.arange(1, 11),
    )