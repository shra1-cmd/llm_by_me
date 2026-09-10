from pathlib import Path

from tokenizers import Tokenizer


TOKENIZER_PATH = Path(
    "data/tokenizer/tokenizer.json"
)


class BPETokenizer:

    def __init__(
        self,
        tokenizer_path: str | Path = TOKENIZER_PATH,
    ):
        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}\n"
                "Run train_tokenizer.py first."
            )

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """
        Convert text → token IDs.
        """

        encoded = self.tokenizer.encode(text)

        return encoded.ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs → text.
        """

        return self.tokenizer.decode(token_ids)

    def encode_batch(
        self,
        texts: list[str],
    ) -> list[list[int]]:
        """
        Encode multiple texts.
        """

        encoded = self.tokenizer.encode_batch(texts)

        return [
            item.ids
            for item in encoded
        ]

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self.tokenizer.id_to_token(token_id)


if __name__ == "__main__":

    tokenizer = BPETokenizer()

    text = "The little cat sat on the mat."

    token_ids = tokenizer.encode(text)

    decoded_text = tokenizer.decode(token_ids)

    print("BPE Tokenizer")
    print("=" * 40)

    print(f"Vocabulary size: {tokenizer.vocab_size:,}")

    print(f"\nOriginal:")
    print(text)

    print("\nToken IDs:")
    print(token_ids)

    print("\nDecoded:")
    print(decoded_text)