from pathlib import Path
import argparse

import numpy as np
from datasets import load_dataset

from src.tokenizer.tokenizer import BPETokenizer


DATASET_NAME = "roneneldan/TinyStories"

OUTPUT_DIR = Path("data/tokens")

TRAIN_OUTPUT = OUTPUT_DIR / "train.bin"
VAL_OUTPUT = OUTPUT_DIR / "validation.bin"

CHUNK_SIZE = 1_000_000


def tokenize_split(
    split: str,
    output_path: Path,
    target_tokens: int | None = None,
):
    """
    Tokenize a TinyStories split and write token IDs to a uint16
    binary file.

    If target_tokens is specified, stop once approximately that
    many tokens have been written.

    Each story receives an EOS token before the next story begins.
    """

    print("=" * 60)
    print(f"Preparing split: {split}")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
    )

    tokenizer = BPETokenizer()

    eos_id = tokenizer.token_to_id("<eos>")

    if eos_id is None:
        raise RuntimeError("Tokenizer does not contain <eos> token.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    token_count = 0
    story_count = 0
    chunks: list[int] = []

    with open(output_path, "wb") as file:

        for example in dataset:

            tokens = tokenizer.encode(
                example["text"]
            )

            # Explicit story boundary.
            tokens.append(eos_id)

            # Don't exceed requested target by too much.
            if target_tokens is not None:

                remaining = target_tokens - token_count

                if remaining <= 0:
                    break

                tokens = tokens[:remaining]

            chunks.extend(tokens)

            token_count += len(tokens)
            story_count += 1

            # Write large chunks instead of performing a disk write
            # for every individual story.
            if len(chunks) >= CHUNK_SIZE:

                array = np.asarray(
                    chunks,
                    dtype=np.uint16,
                )

                file.write(
                    array.tobytes()
                )

                chunks.clear()

            if story_count % 5_000 == 0:

                if target_tokens is not None:

                    percentage = (
                        token_count / target_tokens
                    ) * 100

                    print(
                        f"Stories: {story_count:,} | "
                        f"Tokens: {token_count:,} / "
                        f"{target_tokens:,} "
                        f"({percentage:.2f}%)"
                    )

                else:

                    print(
                        f"Stories: {story_count:,} | "
                        f"Tokens: {token_count:,}"
                    )

            if (
                target_tokens is not None
                and token_count >= target_tokens
            ):
                break

        # Flush remaining tokens.
        if chunks:

            array = np.asarray(
                chunks,
                dtype=np.uint16,
            )

            file.write(
                array.tobytes()
            )

    file_size_mb = (
        output_path.stat().st_size
        / (1024 ** 2)
    )

    print()
    print(f"Finished: {output_path}")
    print(f"Stories processed : {story_count:,}")
    print(f"Tokens written    : {token_count:,}")
    print(f"File size         : {file_size_mb:.2f} MB")

    return token_count


def parse_args():

    parser = argparse.ArgumentParser(
        description="Tokenize TinyStories into binary training data."
    )

    parser.add_argument(
        "--train-tokens",
        type=int,
        default=300_000_000,
        help="Number of training tokens to generate.",
    )

    parser.add_argument(
        "--validation-tokens",
        type=int,
        default=5_000_000,
        help="Number of validation tokens to generate.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print()
    print("=" * 60)
    print("V1 DATA PREPARATION")
    print("=" * 60)

    print(f"Training tokens   : {args.train_tokens:,}")
    print(f"Validation tokens : {args.validation_tokens:,}")
    print(f"Tokenizer         : data/tokenizer/tokenizer.json")
    print()

    train_tokens = tokenize_split(
        split="train",
        output_path=TRAIN_OUTPUT,
        target_tokens=args.train_tokens,
    )

    print()

    validation_tokens = tokenize_split(
        split="validation",
        output_path=VAL_OUTPUT,
        target_tokens=args.validation_tokens,
    )

    print()
    print("=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Training tokens   : {train_tokens:,}"
    )

    print(
        f"Validation tokens : {validation_tokens:,}"
    )

    print(
        f"Training file     : {TRAIN_OUTPUT}"
    )

    print(
        f"Validation file   : {VAL_OUTPUT}"
    )


if __name__ == "__main__":
    main()