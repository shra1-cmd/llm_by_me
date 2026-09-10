from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


DATASET_NAME = "roneneldan/TinyStories"

VOCAB_SIZE = 16_384

OUTPUT_DIR = Path("data/tokenizer")
OUTPUT_FILE = OUTPUT_DIR / "tokenizer.json"


def train_tokenizer():

    print("Loading TinyStories...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train[:1%]",
    )

    print(f"Training examples: {len(dataset):,}")

    # ---------------------------------------------------------
    # 1. Create an empty BPE tokenizer
    # ---------------------------------------------------------

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>"
        )
    )

    # ---------------------------------------------------------
    # 2. Pre-tokenization
    # ---------------------------------------------------------

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    # ---------------------------------------------------------
    # 3. Decoder
    # ---------------------------------------------------------

    tokenizer.decoder = ByteLevelDecoder()

    # ---------------------------------------------------------
    # 4. BPE trainer
    # ---------------------------------------------------------

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,

        special_tokens=[
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ],
    )

    # ---------------------------------------------------------
    # 5. Train on the text column
    # ---------------------------------------------------------

    def text_iterator():
        for example in dataset:
            yield example["text"]

    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer,
    )

    # ---------------------------------------------------------
    # 6. Save tokenizer
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(OUTPUT_FILE))

    print("\nTokenizer trained successfully.")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    train_tokenizer()