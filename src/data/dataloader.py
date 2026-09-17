import torch
from torch.utils.data import Dataset, DataLoader

from src.data.bin_dataset import MemmapTokenDataset


TRAIN_BIN = "data/tokens/train.bin"
VAL_BIN = "data/tokens/validation.bin"


class PackedSequenceDataset(Dataset):
    """
    Original in-memory packed sequence dataset.

    Kept here because it is still useful for unit testing
    and small experiments.
    """

    def __init__(
        self,
        token_ids: list[int],
        seq_len: int = 512,
    ):
        if len(token_ids) < seq_len + 1:
            raise ValueError(
                f"Need at least {seq_len + 1} tokens, "
                f"got {len(token_ids)}"
            )

        self.seq_len = seq_len

        self.num_sequences = (
            (len(token_ids) - 1)
            // seq_len
        )

        usable_tokens = (
            self.num_sequences * seq_len + 1
        )

        self.tokens = torch.tensor(
            token_ids[:usable_tokens],
            dtype=torch.long,
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

        input_ids = sequence[:-1]
        target_ids = sequence[1:]

        return input_ids, target_ids


def create_dataloader(
    split: str = "train",
    seq_len: int = 512,
    batch_size: int = 8,
    shuffle: bool = True,
):

    if split == "train":
        path = TRAIN_BIN

    elif split in (
        "validation",
        "val",
    ):
        path = VAL_BIN

    else:
        raise ValueError(
            f"Unknown split: {split}"
        )

    dataset = MemmapTokenDataset(
        path=path,
        seq_len=seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True,
    )