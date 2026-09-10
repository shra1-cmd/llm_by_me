from datasets import load_dataset


DATASET_NAME = "roneneldan/TinyStories"


def load_tinystories(
    train_split: str = "train[:1%]",
    validation_split: str = "validation[:1%]",
):
    """
    Load a small subset of TinyStories for development.

    We deliberately use a small subset during development.
    The full dataset will be used later for the real training run.
    """

    train_dataset = load_dataset(
        DATASET_NAME,
        split=train_split,
    )

    validation_dataset = load_dataset(
        DATASET_NAME,
        split=validation_split,
    )

    return train_dataset, validation_dataset


if __name__ == "__main__":
    train_dataset, validation_dataset = load_tinystories()

    print("TinyStories")
    print("=" * 40)

    print(f"Train examples      : {len(train_dataset):,}")
    print(f"Validation examples : {len(validation_dataset):,}")

    print("\nExample:")
    print(train_dataset[0]["text"])