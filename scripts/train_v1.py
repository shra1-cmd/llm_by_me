import random

import torch

from configs.v1 import ModelConfig
from configs.train import TrainingConfig

from src.data.dataloader import create_dataloader
from src.model.model import V1LanguageModel

from src.tokenizer.tokenizer import BPETokenizer

from src.training.optimizer import build_optimizer
from src.training.trainer import Trainer


def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    model_config = ModelConfig()

    training_config = TrainingConfig()

    tokenizer = BPETokenizer()

    # Keep model and tokenizer synchronized.
    model_config.vocab_size = (
        tokenizer.vocab_size
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("V1 TRAINING")
    print("=" * 60)

    print(f"Device          : {device}")
    
    print(
    f"BF16 available : "
    f"{torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}"
   )

    print(
        f"Batch size      : "
        f"{training_config.batch_size}"
    )

    print(
        f"Grad accumulation: "
        f"{training_config.gradient_accumulation_steps}"
    )

    effective_batch_size = (
        training_config.batch_size
        * training_config.gradient_accumulation_steps
    )

    print(
        f"Effective batch : "
        f"{effective_batch_size}"
    )

    # --------------------------------------------------
    # Seed
    # --------------------------------------------------

    set_seed(
        training_config.seed
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = V1LanguageModel(
        model_config
    )

    model.to(device)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    train_loader = create_dataloader(
        split="train",
        seq_len=training_config.seq_len,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    val_loader = create_dataloader(
        split="validation",
        seq_len=training_config.seq_len,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = build_optimizer(
        model=model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        beta1=training_config.beta1,
        beta2=training_config.beta2,
        eps=training_config.eps,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        training_config=training_config,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
