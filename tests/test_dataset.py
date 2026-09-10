from src.data.dataset import load_tinystories

def test_tinystories_loading():

    train_dataset, validation_dataset = load_tinystories(
        train_split="train[:1%]",
        validation_split="validation[:1%]",
    )

    assert len(train_dataset) > 0
    assert len(validation_dataset) > 0

    assert "text" in train_dataset.column_names

    assert isinstance(
        train_dataset[0]["text"],
        str,
    )