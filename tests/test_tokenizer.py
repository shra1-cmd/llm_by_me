from src.tokenizer.tokenizer import BPETokenizer


def test_encode_decode():

    tokenizer = BPETokenizer()

    text = "The little cat sat on the mat."

    token_ids = tokenizer.encode(text)

    assert isinstance(token_ids, list)
    assert len(token_ids) > 0

    assert all(
        isinstance(token_id, int)
        for token_id in token_ids
    )

    decoded = tokenizer.decode(token_ids)

    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_vocab_size():

    tokenizer = BPETokenizer()

    assert tokenizer.vocab_size <= 16_384
    assert tokenizer.vocab_size > 1