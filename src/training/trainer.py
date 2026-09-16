import math
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

from src.training.scheduler import update_learning_rate


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        model_config,
        training_config,
        device,
    ):
        self.model = model
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model_config = model_config
        self.training_config = training_config

        self.device = device

        self.use_bf16 = (
            training_config.use_bf16
            and device.startswith("cuda")
            and torch.cuda.is_bf16_supported()
        )

    # ==========================================================
    # FIND LATEST CHECKPOINT
    # ==========================================================

    def find_latest_checkpoint(self):

        checkpoint_dir = Path(
            self.training_config.checkpoint_dir
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoints = list(
            checkpoint_dir.glob("step_*.pt")
        )

        if not checkpoints:
            return None

        def get_step(path):

            return int(
                path.stem.split("_")[1]
            )

        checkpoints.sort(
            key=get_step
        )

        return checkpoints[-1]

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        total_loss = 0.0

        iterator = iter(
            self.val_loader
        )

        for _ in range(
            self.training_config.eval_steps
        ):

            try:

                input_ids, targets = next(
                    iterator
                )

            except StopIteration:

                iterator = iter(
                    self.val_loader
                )

                input_ids, targets = next(
                    iterator
                )

            input_ids = input_ids.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda"
                if self.device.startswith("cuda")
                else "cpu",
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            ):

                _, loss = self.model(
                    input_ids,
                    targets,
                )

            total_loss += loss.item()

        average_loss = (
            total_loss
            / self.training_config.eval_steps
        )

        self.model.train()

        return average_loss

    # ==========================================================
    # TRAIN
    # ==========================================================

    def train(self):

        self.model.train()

        # ------------------------------------------------------
        # Find latest checkpoint
        # ------------------------------------------------------

        latest_checkpoint = (
            self.find_latest_checkpoint()
        )

        start_step = 0

        if latest_checkpoint is not None:

            print()
            print("=" * 70)
            print("🔄 CHECKPOINT FOUND")
            print("=" * 70)

            print(
                f"Loading checkpoint: "
                f"{latest_checkpoint}"
            )

            checkpoint = load_checkpoint(
                path=latest_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
            )

            start_step = (
                checkpoint["step"] + 1
            )

            print(
                f"Previous step : "
                f"{checkpoint['step']}"
            )

            print(
                f"Previous loss : "
                f"{checkpoint['loss']:.6f}"
            )

            print(
                f"Resuming from : "
                f"{start_step}"
            )

            print("=" * 70)

        else:

            print()
            print("=" * 70)
            print("🆕 NO CHECKPOINT FOUND")
            print("=" * 70)

            print(
                "Starting training from step 0."
            )

            print("=" * 70)

        # ------------------------------------------------------
        # Data iterator
        # ------------------------------------------------------

        train_iterator = iter(
            self.train_loader
        )

        # ------------------------------------------------------
        # Restore approximate data position
        #
        # One optimizer step consumes:
        #
        # gradient_accumulation_steps
        #
        # micro-batches.
        # ------------------------------------------------------

        micro_batches_to_skip = (
            start_step
            * self.training_config
            .gradient_accumulation_steps
        )

        if start_step > 0:

            print(
                f"Restoring data position..."
            )

            print(
                f"Skipping "
                f"{micro_batches_to_skip:,} "
                f"micro-batches..."
            )

            for _ in range(
                micro_batches_to_skip
            ):

                try:

                    next(train_iterator)

                except StopIteration:

                    train_iterator = iter(
                        self.train_loader
                    )

                    next(train_iterator)

            print(
                "✓ Data position restored."
            )

        # ------------------------------------------------------
        # Training statistics
        # ------------------------------------------------------

        running_loss = 0.0

        previous_logged_loss = None

        total_start_time = time.time()

        start_time = time.time()

        # ------------------------------------------------------
        # Tokens processed per optimizer step
        # ------------------------------------------------------

        tokens_per_step = (
            self.training_config.batch_size
            * self.training_config.gradient_accumulation_steps
            * self.training_config.seq_len
        )

        total_tokens = (
            self.training_config.max_steps
            * tokens_per_step
        )

        # ======================================================
        # MAIN TRAINING LOOP
        # ======================================================

        for step in range(
            start_step,
            self.training_config.max_steps
        ):

            self.optimizer.zero_grad(
                set_to_none=True
            )

            accumulated_loss = 0.0

            # --------------------------------------------------
            # Gradient accumulation
            # --------------------------------------------------

            for micro_step in range(
                self.training_config
                .gradient_accumulation_steps
            ):

                try:

                    input_ids, targets = next(
                        train_iterator
                    )

                except StopIteration:

                    train_iterator = iter(
                        self.train_loader
                    )

                    input_ids, targets = next(
                        train_iterator
                    )

                input_ids = input_ids.to(
                    self.device,
                    non_blocking=True,
                )

                targets = targets.to(
                    self.device,
                    non_blocking=True,
                )

                # ------------------------------------------------
                # Forward pass
                # ------------------------------------------------

                with torch.autocast(
                    device_type="cuda"
                    if self.device.startswith("cuda")
                    else "cpu",
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):

                    _, loss = self.model(
                        input_ids,
                        targets,
                    )

                    # Divide because gradients are accumulated
                    # over multiple micro-batches.
                    loss = (
                        loss
                        / self.training_config
                        .gradient_accumulation_steps
                    )

                # ------------------------------------------------
                # Backward
                # ------------------------------------------------

                loss.backward()

                accumulated_loss += (
                    loss.item()
                )

            # --------------------------------------------------
            # Gradient clipping
            # --------------------------------------------------

            grad_norm = clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip,
            )

            # --------------------------------------------------
            # Learning rate
            # --------------------------------------------------

            lr = update_learning_rate(
                optimizer=self.optimizer,
                step=step,
                max_steps=self.training_config.max_steps,
                warmup_steps=self.training_config.warmup_steps,
                learning_rate=self.training_config.learning_rate,
                min_learning_rate=self.training_config.min_learning_rate,
            )

            # --------------------------------------------------
            # Optimizer update
            # --------------------------------------------------

            self.optimizer.step()

            # --------------------------------------------------
            # Running loss
            # --------------------------------------------------

            running_loss += (
                accumulated_loss
            )

            # ==================================================
            # LOGGING
            # ==================================================

            if (
                step % self.training_config.log_interval
                == 0
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                # ----------------------------------------------
                # Average loss
                # ----------------------------------------------

                if step > 0:

                    average_loss = (
                        running_loss
                        / self.training_config
                        .log_interval
                    )

                else:

                    average_loss = (
                        accumulated_loss
                    )

                # ----------------------------------------------
                # Perplexity
                # ----------------------------------------------

                perplexity = math.exp(
                    min(
                        average_loss,
                        20,
                    )
                )

                # ----------------------------------------------
                # Loss improvement
                # ----------------------------------------------

                if previous_logged_loss is None:

                    loss_change = 0.0

                else:

                    loss_change = (
                        previous_logged_loss
                        - average_loss
                    )

                # ----------------------------------------------
                # Throughput
                # ----------------------------------------------

                if step > 0:

                    tokens_processed_interval = (
                        tokens_per_step
                        * self.training_config
                        .log_interval
                    )

                else:

                    tokens_processed_interval = (
                        tokens_per_step
                    )

                tokens_per_second = (
                    tokens_processed_interval
                    / max(elapsed, 1e-6)
                )

                # ----------------------------------------------
                # Total progress
                # ----------------------------------------------

                completed_tokens = (
                    (step + 1)
                    * tokens_per_step
                )

                progress = (
                    completed_tokens
                    / total_tokens
                    * 100
                )

                # ----------------------------------------------
                # ETA
                # ----------------------------------------------

                steps_remaining = (
                    self.training_config.max_steps
                    - step
                    - 1
                )

                if step > 0:

                    seconds_per_step = (
                        elapsed
                        / self.training_config
                        .log_interval
                    )

                else:

                    seconds_per_step = elapsed

                eta_seconds = (
                    steps_remaining
                    * seconds_per_step
                )

                # ----------------------------------------------
                # Total elapsed time
                # ----------------------------------------------

                total_elapsed = (
                    time.time()
                    - total_start_time
                )

                # ----------------------------------------------
                # PRINT
                # ----------------------------------------------

                print(
                    f"\n"
                    f"step={step:6d} | "
                    f"loss={average_loss:.4f} | "
                    f"Δloss={loss_change:+.4f} | "
                    f"ppl={perplexity:.2f} | "
                    f"lr={lr:.2e} | "
                    f"grad={grad_norm:.3f}"
                )

                print(
                    f"tokens={completed_tokens:,} "
                    f"/ {total_tokens:,} | "
                    f"progress={progress:.2f}%"
                )

                print(
                    f"throughput={tokens_per_second:,.0f} tok/s | "
                    f"elapsed={total_elapsed / 60:.2f} min | "
                    f"ETA={eta_seconds / 3600:.2f} hr"
                )

                # ----------------------------------------------
                # Update statistics
                # ----------------------------------------------

                previous_logged_loss = (
                    average_loss
                )

                running_loss = 0.0

                start_time = time.time()

            # ==================================================
            # VALIDATION
            # ==================================================

            if (
                step > 0
                and step
                % self.training_config.eval_interval
                == 0
            ):

                val_loss = self.evaluate()

                val_perplexity = math.exp(
                    min(
                        val_loss,
                        20,
                    )
                )

                print()
                print(
                    f"[VALIDATION] "
                    f"step={step} "
                    f"loss={val_loss:.4f} "
                    f"ppl={val_perplexity:.2f}"
                )

            # ==================================================
            # CHECKPOINT
            # ==================================================

            if (
                step > 0
                and step
                % self.training_config.checkpoint_interval
                == 0
            ):

                checkpoint_path = (
                    Path(
                        self.training_config
                        .checkpoint_dir
                    )
                    / f"step_{step}.pt"
                )

                save_checkpoint(
                    path=checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    step=step,
                    loss=(
                        accumulated_loss
                    ),
                    config=self.model_config,
                    training_config=(
                        self.training_config
                    ),
                )

                print()
                print("=" * 70)
                print("💾 CHECKPOINT SAVED")
                print("=" * 70)

                print(
                    f"Step   : {step}"
                )

                print(
                    f"Tokens : {completed_tokens:,}"
                )

                print(
                    f"Loss   : {average_loss:.6f}"
                )

                print(
                    f"Path   : {checkpoint_path}"
                )

                print("=" * 70)

        # ======================================================
        # TRAINING COMPLETE
        # ======================================================

        total_elapsed = (
            time.time()
            - total_start_time
        )

        print()
        print("=" * 70)
        print("🎉 TRAINING COMPLETE")
        print("=" * 70)

        print(
            f"Final step       : "
            f"{self.training_config.max_steps - 1}"
        )

        print(
            f"Total tokens     : "
            f"{total_tokens:,}"
        )

        print(
            f"Total time       : "
            f"{total_elapsed / 3600:.2f} hours"
        )

        print("=" * 70)
