import math
import time

import torch
from torch.nn.utils import clip_grad_norm_

from src.training.checkpoint import save_checkpoint
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

    def train(self):

        self.model.train()

        train_iterator = iter(
            self.train_loader
        )

        running_loss = 0.0

        start_time = time.time()

        for step in range(
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

                    # Important:
                    # Divide loss because we're accumulating
                    # gradients across multiple micro-batches.
                    loss = (
                        loss
                        / self.training_config
                        .gradient_accumulation_steps
                    )

                loss.backward()

                accumulated_loss += loss.item()

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

            running_loss += (
                accumulated_loss
            )

            # --------------------------------------------------
            # Logging
            # --------------------------------------------------

            if (
                step % self.training_config.log_interval
                == 0
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                average_loss = (
                    running_loss
                    / self.training_config.log_interval
                    if step > 0
                    else accumulated_loss
                )

                perplexity = math.exp(
                    min(average_loss, 20)
                )

                print(
                    f"step={step:6d} "
                    f"loss={average_loss:.4f} "
                    f"ppl={perplexity:.2f} "
                    f"lr={lr:.2e} "
                    f"grad_norm={grad_norm:.3f} "
                    f"time={elapsed:.1f}s"
                )

                running_loss = 0.0
                start_time = time.time()

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            if (
                step > 0
                and step
                % self.training_config.eval_interval
                == 0
            ):

                val_loss = self.evaluate()

                print(
                    f"[validation] "
                    f"step={step} "
                    f"loss={val_loss:.4f}"
                )

            # --------------------------------------------------
            # Checkpoint
            # --------------------------------------------------

            if (
                step > 0
                and step
                % self.training_config.checkpoint_interval
                == 0
            ):

                path = (
                    f"{self.training_config.checkpoint_dir}"
                    f"/step_{step}.pt"
                )

                save_checkpoint(
                    path=path,
                    model=self.model,
                    optimizer=self.optimizer,
                    step=step,
                    loss=loss.item(),
                    config=self.model_config,
                    training_config=self.training_config,
                )