from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    seq_len: int = 512

    # Number of sequences processed by the DataLoader
    # at once.
    batch_size: int = 8

    # Number of forward/backward passes before
    # optimizer.step().
    gradient_accumulation_steps: int = 8

    # --------------------------------------------------
    # Optimization
    # --------------------------------------------------

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5

    weight_decay: float = 0.1

    beta1: float = 0.9
    beta2: float = 0.95

    eps: float = 1e-8

    gradient_clip: float = 1.0

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    max_steps: int = 10_000

    warmup_steps: int = 500

    # --------------------------------------------------
    # Precision
    # --------------------------------------------------

    use_bf16: bool = True

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    eval_interval: int = 500
    eval_steps: int = 50

    # --------------------------------------------------
    # Checkpointing
    # --------------------------------------------------

    checkpoint_interval: int = 50
    checkpoint_dir: str = "/content/drive/MyDrive/llm_by_me_data/checkpoints/v1"

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    log_interval: int = 10

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    seed: int = 42
