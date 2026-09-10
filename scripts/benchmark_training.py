import time

import torch

from configs.v1 import ModelConfig
from configs.train import TrainingConfig
from src.data.dataloader import create_dataloader
from src.model.model import V1LanguageModel
from src.training.optimizer import build_optimizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def benchmark():
    device = get_device()

    model_config = ModelConfig()
    train_config = TrainingConfig()

    # Use the actual tokenizer vocabulary size.
    model_config.vocab_size = 15_485

    # Benchmark configuration.
    seq_len = 512
    batch_size = 8
    gradient_accumulation_steps = 8
    warmup_steps = 3
    benchmark_steps = 20

    print("=" * 60)
    print("V1 TRAINING THROUGHPUT BENCHMARK")
    print("=" * 60)

    print("\nHardware")
    print("-" * 60)
    print(f"Device          : {device}")

    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version    : {torch.version.cuda}")
        print(
            f"BF16 supported  : "
            f"{torch.cuda.is_bf16_supported()}"
        )

    print("\nConfiguration")
    print("-" * 60)
    print(f"Parameters      : {model_config.vocab_size:,} vocab")
    print(f"Sequence length : {seq_len}")
    print(f"Micro batch     : {batch_size}")
    print(f"Grad accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch : {batch_size * gradient_accumulation_steps}")
    print(
        f"Tokens/optimizer step: "
        f"{batch_size * gradient_accumulation_steps * seq_len:,}"
    )
    print(f"Benchmark steps : {benchmark_steps}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = V1LanguageModel(model_config).to(device)
    model.train()

    optimizer = build_optimizer(
    model=model,
    learning_rate=train_config.learning_rate,
    weight_decay=train_config.weight_decay,
    beta1=train_config.beta1,
    beta2=train_config.beta2,
    eps=train_config.eps,
)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    loader = create_dataloader(
        split="train",
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
    )

    data_iter = iter(loader)

    # ------------------------------------------------------------------
    # BF16
    # ------------------------------------------------------------------

    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    print(f"Using BF16      : {use_bf16}")

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    print("\nWarmup")
    print("-" * 60)

    for step in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)

        for _ in range(gradient_accumulation_steps):
            try:
                input_ids, target_ids = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                input_ids, target_ids = next(data_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                _, loss = model(input_ids, target_ids)
                loss = loss / gradient_accumulation_steps

            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.gradient_clip,
        )

        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Reset memory statistics
    # ------------------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    print("\nBenchmark")
    print("-" * 60)

    times = []

    total_tokens = (
        benchmark_steps
        * batch_size
        * gradient_accumulation_steps
        * seq_len
    )

    for step in range(benchmark_steps):
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for _ in range(gradient_accumulation_steps):
            try:
                input_ids, target_ids = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                input_ids, target_ids = next(data_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                _, loss = model(input_ids, target_ids)
                loss = loss / gradient_accumulation_steps

            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.gradient_clip,
        )

        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        times.append(elapsed)

        step_tokens = (
            batch_size
            * gradient_accumulation_steps
            * seq_len
        )

        tokens_per_second = step_tokens / elapsed

        print(
            f"step {step + 1:02d}/{benchmark_steps} | "
            f"time: {elapsed:.3f}s | "
            f"tokens/sec: {tokens_per_second:,.0f}"
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    # Ignore the first benchmark iteration as an additional stabilization
    # point. The explicit warmup above handles most startup effects.
    measured_times = times[1:] if len(times) > 1 else times

    average_step_time = sum(measured_times) / len(measured_times)
    tokens_per_step = (
        batch_size
        * gradient_accumulation_steps
        * seq_len
    )
    tokens_per_second = tokens_per_step / average_step_time

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    else:
        peak_memory = None
        peak_reserved = None

    target_tokens = 300_000_000

    optimizer_steps_required = target_tokens / tokens_per_step
    estimated_seconds = target_tokens / tokens_per_second
    estimated_hours = estimated_seconds / 3600

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"Average step time       : {average_step_time:.3f}s")
    print(f"Tokens per optimizer step: {tokens_per_step:,}")
    print(f"Tokens per second       : {tokens_per_second:,.0f}")

    if peak_memory is not None:
        print(f"Peak allocated memory   : {peak_memory:.2f} GB")
        print(f"Peak reserved memory    : {peak_reserved:.2f} GB")

    print("\n300M TOKEN ESTIMATE")
    print("-" * 60)
    print(f"Target tokens           : {target_tokens:,}")
    print(f"Optimizer steps         : {optimizer_steps_required:,.0f}")
    print(f"Estimated training time : {estimated_hours:.2f} hours")

    print("=" * 60)


if __name__ == "__main__":
    benchmark()