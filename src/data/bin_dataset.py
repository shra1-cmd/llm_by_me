from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapTokenDataset(Dataset):

    def __init__(
        self,
        path: str | Path,
        seq_len: int = 512,
    ):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Token file not found: {self.path}"
            )

        self.seq_len = seq_len

        # Each token is uint16 = 2 bytes.
        self.tokens = np.memmap(
            self.path,
            dtype=np.uint16,
            mode="r",
        )

        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                "Token file does not contain "
                "enough tokens."
            )

        self.num_sequences = (
            (len(self.tokens) - 1)
            // seq_len
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):

        start = index * self.seq_len

        end = (
            start
            + self.seq_len
            + 1
        )

        sequence = self.tokens[
            start:end
        ]

        # Copy is important because the underlying
        # memmap is read-only.
        sequence = np.asarray(
            sequence,
            dtype=np.int64,
        )

        sequence = torch.from_numpy(
            sequence
        )

        input_ids = sequence[:-1]
        target_ids = sequence[1:]

        return input_ids, target_ids