# llm_by_me
Building a llm with ongoing functionalities on my PC


wed sep 16 "A RUNNABLE PERSISTANT COLLAB TRAINING SETUP"
>>Briefing by GPT  to line 2378

# `llm_by_me` — V1 LLM From Scratch Project Documentation

Below is a consolidated technical record of what we have done in this project so far: the objective, architecture, data pipeline, Colab setup, training configuration, checkpointing, problems encountered, debugging process, fatigue testing, what was verified, and what remains to be improved.

I am deliberately separating **things we have actually verified by running them** from **things we discussed/planned but have not yet fully validated**.

---

# 1. Project Overview

## Project

**`llm_by_me` — V1**

The objective of the project is to build and train a small language model from scratch rather than simply using an existing pretrained LLM.

The project is being developed as a proper ML/LLM systems project, covering:

```text
Dataset
   ↓
Tokenization
   ↓
Binary token storage
   ↓
DataLoader
   ↓
Language Model
   ↓
Forward pass
   ↓
Loss
   ↓
Backward pass
   ↓
Gradient accumulation
   ↓
Gradient clipping
   ↓
Optimizer
   ↓
Learning-rate scheduler
   ↓
Checkpoint
   ↓
Evaluation / monitoring
```

The training environment is Google Colab using an NVIDIA T4 GPU, with Google Drive used as persistent storage.

The important systems problem we specifically addressed was:

> **How can a long-running LLM training job survive Colab runtime termination without restarting training from step 0?**

The solution we implemented and tested is **persistent checkpointing to Google Drive + automatic checkpoint discovery and resume**.

---

# 2. Overall System Architecture

The project currently has three important environments:

```text
                    GitHub
                      │
                      │ V1 source code
                      ▼
              ┌─────────────────┐
              │     Colab       │
              │                 │
              │ Python runtime  │
              │ NVIDIA T4       │
              │ PyTorch         │
              │ Training        │
              └────────┬────────┘
                       │
              temporary filesystem
                       │
                       │
                       ▼
              /content/llm_by_me
                       │
             ┌─────────┴─────────┐
             │                   │
          source code          data
             │                   │
             │                   │
             ▼                   ▼
        model/trainer       train.bin
                            validation.bin
                            tokenizer.json

                       │
                       │ persistent storage
                       ▼

                Google Drive
                       │
             llm_by_me_data/
                       │
          ┌────────────┼─────────────┐
          │            │             │
        tokens      tokenizer    checkpoints
          │            │             │
      train.bin   tokenizer.json    v1/
      validation.bin                 │
                                     ├── step_50.pt
                                     ├── step_100.pt
                                     ├── step_150.pt
                                     └── ...
```

The key design principle is:

> **Colab provides compute; Google Drive provides persistence.**

If Colab dies, the compute environment disappears, but the training data and checkpoints remain on Drive.

---

# 3. Git Repository

The repository is:

```text
llm_by_me
```

and we have been working on the:

```text
V1
```

branch.

The Colab notebook clones that branch:

```python
git clone -b V1 https://github.com/shra1-cmd/llm_by_me.git
```

The notebook then enters the repository and runs the training code.

The important source files we have examined/modified during the project are:

```text
llm_by_me/
│
├── configs/
│   └── train.py
│
├── scripts/
│   └── train_v1.py
│
├── src/
│   ├── data/
│   │   ├── dataloader.py
│   │   └── bin_dataset.py
│   │
│   └── training/
│       ├── trainer.py
│       ├── checkpoint.py
│       ├── optimizer.py
│       └── scheduler.py
│
└── tests/
```

The original training script constructs the model, tokenizer, dataloaders, optimizer and trainer, and then starts `Trainer.train()`. The repository's training configuration provides the main hyperparameters such as sequence length, batch size, gradient accumulation, learning rate, maximum steps, evaluation interval and checkpoint interval.

---

# 4. Google Drive Persistent Storage

One of the first important decisions was to avoid keeping the dataset and checkpoints only inside Colab.

Colab's:

```text
/content
```

filesystem is temporary.

Therefore, we created:

```text
My Drive/
└── llm_by_me_data/
    ├── tokens/
    │   ├── train.bin
    │   └── validation.bin
    │
    ├── tokenizer/
    │   └── tokenizer.json
    │
    └── checkpoints/
        └── v1/
```

This is now the persistent storage layer.

The notebook defines:

```python
drive_base = "/content/drive/MyDrive/llm_by_me_data"
```

and then:

```python
drive_tokens = os.path.join(
    drive_base,
    "tokens"
)

drive_tokenizer = os.path.join(
    drive_base,
    "tokenizer"
)
```

The current notebook copies:

```text
Drive:
llm_by_me_data/tokens/train.bin
        ↓
Local:
llm_by_me/data/tokens/train.bin
```

and:

```text
Drive:
llm_by_me_data/tokens/validation.bin
        ↓
Local:
llm_by_me/data/tokens/validation.bin
```

and:

```text
Drive:
llm_by_me_data/tokenizer/tokenizer.json
        ↓
Local:
llm_by_me/data/tokenizer/tokenizer.json
```

This mapping is important because we previously had a path mistake.

The current notebook explicitly performs these copies and verifies the resulting local directories. 

---

# 5. Why We Don't Regenerate the Dataset

An important discussion we had was whether the dataset needed to be generated again.

The answer was:

**No.**

The tokenized dataset already exists.

We have:

```text
train.bin
validation.bin
tokenizer.json
```

Therefore, training starts from the existing tokenized representation.

There is no need to:

```text
download raw data again
        ↓
tokenize again
        ↓
generate train.bin again
```

on every Colab session.

Instead:

```text
Google Drive
     ↓
existing token files
     ↓
copy to temporary Colab filesystem
     ↓
train
```

This makes restarting a runtime much faster and, more importantly, keeps the dataset consistent across training sessions.

---

# 6. Colab Notebook Responsibilities

The notebook is intentionally being used as the **execution/orchestration layer**, not as the place where the model's training logic lives.

The current notebook performs approximately:

```text
1. Check GPU
2. Clone V1 repository
3. Install dependencies
4. Mount Google Drive
5. Copy training data
6. Copy tokenizer
7. Verify files
8. Run tests
9. Run training benchmark
10. Launch train_v1.py
11. Stream training output
```

The notebook's training launcher uses:

```python
subprocess.Popen(
    [sys.executable, "-u", "scripts/train_v1.py"],
    ...
)
```

and streams the process output live. 

This is preferable to hiding the training process because we can observe:

* step
* loss
* learning rate
* gradient norm
* throughput
* token progress
* ETA
* checkpoint creation
* errors

during the run.

---

# 7. Initial Dataset Path Problem

The first major error we encountered was:

```text
FileNotFoundError:
Token file not found: data/tokens/train.bin
```

At first this looked like the dataset was missing.

But the actual issue was the relationship between:

```text
Google Drive location
```

and:

```text
local Colab location
```

We corrected the data structure to:

```text
Drive
llm_by_me_data/
└── tokens/
    ├── train.bin
    └── validation.bin
```

and the notebook copies them to:

```text
/content/llm_by_me/data/tokens/
```

The notebook now explicitly checks that the source files exist before copying them. 

This eliminated the original dataset-location problem during the successful training run.

---

# 8. Training Configuration

The important training parameters observed during the successful run were:

```text
Device                : cuda
GPU                   : NVIDIA T4
BF16 available        : True

Batch size            : 8
Gradient accumulation : 8
Effective batch       : 64
Sequence length       : 512
```

The effective batch calculation is:

```text
micro-batch = 8 sequences
gradient accumulation = 8

8 × 8 = 64 sequences per optimizer step
```

Because every sequence contains 512 tokens:

```text
8 × 8 × 512
=
32,768 tokens / optimizer step
```

Therefore:

> **One optimizer step processes approximately 32,768 tokens.**

This is one of the most important quantities for understanding the training run.

---

# 9. Token Throughput

During the actual training run we observed roughly:

```text
9,000–10,000 tokens/sec
```

For example:

```text
step 50:
~10,209 tok/s

step 100:
~10,112 tok/s

step 110:
~9,793 tok/s
```

After restarting the Colab runtime:

```text
step 210:
~8,763 tok/s

step 220:
~9,117 tok/s
```

So the new runtime was operating in roughly the same order of magnitude.

The temporary difference is expected because GPU/runtime conditions can vary somewhat between Colab sessions.

---

# 10. Training Progress Observed

One of the most useful things we observed was the loss trajectory.

The initial run showed:

```text
Step      Loss
----------------
0         473.3734
10        470.0282
20        460.0577
30        434.7178
40        358.2504
50        226.9308
60        124.4289
70         75.3719
80         55.5094
90         43.3828
100        37.3223
110        32.9633
```

This shows a strong reduction in training loss during the initial phase.

Later, the resumed run found:

```text
checkpoint step = 200
checkpoint loss = 18.321518
```

and continued:

```text
step 210 → loss 17.6731
step 220 → loss 17.2429
```

Therefore, the resumed model continued training from the existing model state rather than initializing a new model.

---

# 11. Logging System

We added/used a fairly detailed training logger.

The logger reports:

```text
step
loss
loss improvement
perplexity
learning rate
gradient norm
tokens processed
progress
throughput
elapsed time
ETA
```

For example:

```text
step=   220
loss=17.2429
Δloss=+0.4302
ppl=30796559.72
lr=1.33e-04
grad=9.496

tokens=7,241,728 / 327,680,000
progress=2.21%

throughput=9,117 tok/s
elapsed=1.22 min
ETA=9.76 hr
```

The logging code lives in the training implementation, not in the Colab notebook. 

This was an important clarification during our discussion.

### Notebook vs training code

We established:

```text
Colab notebook
    ↓
starts training

trainer.py
    ↓
actually performs training
    ↓
logs metrics
    ↓
saves checkpoints
    ↓
loads checkpoints
```

Therefore, a line such as:

```python
step % self.training_config.log_interval
```

belongs to:

```text
src/training/trainer.py
```

and is not something we should randomly modify in the notebook.

---

# 12. Perplexity Logging Issue

We identified one logging issue.

The code calculates:

```python
math.exp(
    min(
        average_loss,
        20,
    )
)
```

This effectively means:

```text
if loss > 20:
    perplexity = exp(20)
```

Therefore, during the early part of training, perplexity appeared as:

```text
485165195.41
```

even while the loss was changing:

```text
473
470
460
434
358
...
```

This is because the value is being clipped at 20 before exponentiation.

Once the loss went below 20, perplexity started changing:

```text
loss = 18.3215
loss = 17.6731
loss = 17.2429
```

and the displayed perplexity correspondingly changed.

### Interpretation

This is a **logging/metric presentation issue**, not evidence that the model stopped learning.

It is something we should clean up later.

---

# 13. Checkpointing

The central systems feature of the project is persistent checkpointing.

The checkpoint directory is:

```text
/content/drive/MyDrive/llm_by_me_data/checkpoints/v1
```

The training run produced:

```text
step_50.pt
step_100.pt
...
```

and later:

```text
step_200.pt
```

The checkpoint output explicitly showed:

```text
Checkpoint saved:
/content/drive/MyDrive/llm_by_me_data/checkpoints/v1/step_100.pt
```

The checkpoint mechanism uses the project's existing:

```python
save_checkpoint()
load_checkpoint()
```

rather than creating a separate checkpoint format.

The checkpoint contains training state including:

```text
step
loss
model_state_dict
optimizer_state_dict
model_config
training_config
torch RNG state
CUDA RNG state
```

This is important because saving only the model weights would not be sufficient for a proper training resume.

We want:

```text
Model
+
Optimizer
+
Training position
+
Random state
+
Configuration
```

to be restored.

---

# 14. Why Optimizer State Matters

Suppose we only saved:

```text
model_state_dict
```

Then after a restart:

```text
Model weights
    ↓
restored

Optimizer
    ↓
new optimizer
```

The optimizer would not have its previous internal state.

For optimizers such as Adam-style optimizers, that means losing momentum/variance estimates.

Our checkpoint system instead saves:

```text
optimizer_state_dict
```

as well.

Therefore the restart is much closer to:

```text
training never stopped
```

rather than:

```text
load model weights and start a new optimization process
```

---

# 15. Automatic Latest-Checkpoint Discovery

The trainer does not require us to manually specify:

```text
step_100.pt
```

or:

```text
step_200.pt
```

Instead it searches:

```text
checkpoints/v1/
```

for:

```text
step_*.pt
```

and determines the highest step number.

Conceptually:

```text
checkpoints/v1/

step_50.pt
step_100.pt
step_150.pt
step_200.pt
```

becomes:

```text
latest = step_200.pt
```

Then:

```python
load_checkpoint(...)
```

is called automatically.

This means the workflow after a Colab restart is:

```text
Start notebook
      ↓
Mount Drive
      ↓
Training starts
      ↓
Search checkpoint directory
      ↓
Find highest checkpoint
      ↓
Load it
      ↓
Resume
```

No manual checkpoint selection is required.

---

# 16. The Fatigue Test

This was one of the most important experiments we performed.

The purpose was not simply to train the model.

We wanted to answer:

> **If Colab completely dies, can the next Colab session continue training from the persistent checkpoint?**

We therefore intentionally stopped/terminated the runtime.

Before termination, the model had already created checkpoints.

The key principle was:

```text
Colab runtime
     ↓
temporary

Google Drive
     ↓
persistent
```

After terminating the runtime, we started a fresh session and reran the notebook.

---

# 17. Nested Repository Error After Restart

The first fresh-session attempt produced:

```text
/content/llm_by_me/llm_by_me/scripts/train_v1.py
```

This was a critical clue.

The repository was being referenced as:

```text
/content/llm_by_me/llm_by_me
```

instead of:

```text
/content/llm_by_me
```

The result was:

```text
FileNotFoundError:
Token file not found:
data/tokens/train.bin
```

The dataset itself wasn't the problem.

The problem was the current working directory.

The notebook had:

```python
os.chdir("llm_by_me")
```

If the notebook was already inside:

```text
/content/llm_by_me
```

then that command produced:

```text
/content/llm_by_me/llm_by_me
```

### Fix

We made the repository initialization robust by starting from:

```python
os.chdir("/content")
```

before cloning/entering the repository.

The correct intended structure is:

```text
/content
    │
    └── llm_by_me
```

not:

```text
/content
    │
    └── llm_by_me
          │
          └── llm_by_me
```

This was a **Colab working-directory issue**, not a checkpoint failure.

---

# 18. Second Fatigue Test — Successful Resume

After fixing the working-directory problem, we started a new Colab session.

This time the output showed:

```text
🔄 CHECKPOINT FOUND

Loading checkpoint:
/content/drive/MyDrive/llm_by_me_data/checkpoints/v1/step_200.pt

Previous step : 200
Previous loss : 18.321518
Resuming from : 201
```

This is extremely important.

It proves that:

1. Google Drive survived the previous runtime.
2. The checkpoint file survived.
3. The new Colab session could access Drive.
4. The trainer found the checkpoint automatically.
5. The model checkpoint loaded.
6. The optimizer checkpoint loaded.
7. The training step was restored.
8. Training continued from the checkpoint rather than starting at step 0.

---

# 19. Data-Position Restoration

There is another component in the resume mechanism.

The model knows:

```text
"I was at step 200."
```

But a Python `DataLoader` iterator itself is not automatically preserved just because the model checkpoint was saved.

Therefore, we added logic that attempts to advance the training iterator.

The logic is approximately:

```text
completed optimizer steps
        ×
gradient accumulation
        =
micro-batches to skip
```

For the observed resume:

```text
start_step = 201

gradient accumulation = 8

201 × 8
=
1,608 micro-batches
```

The log showed exactly:

```text
Restoring data position...
Skipping 1,608 micro-batches...
✓ Data position restored.
```

So this component executed successfully.

---

# 20. Important Data-Position Caveat

Although the fatigue test proved that the data-position restoration **runs**, we have not yet proven that it is mathematically/exactly equivalent to continuing the original process.

There is a subtle issue.

The current code does:

```python
start_step = checkpoint["step"] + 1
```

and then:

```python
micro_batches_to_skip = (
    start_step
    * gradient_accumulation_steps
)
```

Therefore:

```text
checkpoint step 200
        ↓
start_step = 201
        ↓
201 × 8
        ↓
1,608 skipped micro-batches
```

But if `step_200.pt` represents the state **after optimizer update 200**, the exact number of previously consumed micro-batches depends on how the step numbering convention is defined.

This is an **off-by-one/data-position concern**.

We deliberately did not change it immediately because the goal of the fatigue test was first to establish whether the basic persistence architecture worked.

It does.

The exact deterministic data-stream resume should be the next engineering refinement.

---

# 21. Shuffle and Exact Resume

There is another related issue.

The training dataloader originally used:

```python
shuffle=True
```

while validation uses:

```python
shuffle=False
```

If the training data is shuffled, simply saying:

```text
skip N batches
```

does not necessarily reproduce the exact same sequence of training batches after a restart.

For exact deterministic training continuation, ideally we would preserve:

```text
sampler state
dataloader state
random generator state
worker state
```

Our current approach is simpler:

```text
restore checkpoint
+
advance data iterator
```

This is acceptable for the current single-GPU V1 experiment as a practical resume mechanism, but it should not be described as perfect distributed deterministic recovery.

---

# 22. What the Current Checkpoint System Actually Guarantees

At this point we can confidently say:

### Guaranteed/verified

```text
Model weights persist       ✅
Optimizer state persists    ✅
Training step persists      ✅
CPU RNG state persists      ✅
CUDA RNG state persists     ✅
Checkpoint persists on Drive ✅
Fresh Colab finds checkpoint ✅
Fresh Colab loads checkpoint ✅
Training continues           ✅
```

### Not yet fully guaranteed

```text
Exact same DataLoader sequence after restart
        ⚠️

Bit-for-bit identical continuation
        ⚠️

Exact step/data-position semantics
        ⚠️
```

That distinction is important for good engineering documentation.

---

# 23. Checkpoint Frequency

The successful run saved checkpoints at:

```text
step 50
step 100
```

and subsequently a:

```text
step 200
```

checkpoint was available.

The intended checkpoint frequency is:

```text
every 50 optimizer steps
```

This means the maximum amount of recent training potentially lost by a sudden runtime death is approximately one checkpoint interval, assuming the checkpoint itself has successfully finished writing.

For example:

```text
step 200 → checkpoint
step 201
step 202
...
step 229
runtime dies
```

The recovery point would be approximately:

```text
step 200
```

So the most recent ~29 steps would have to be recomputed.

This is the tradeoff of checkpoint frequency:

```text
More frequent
    ↓
less lost computation
    ↓
more Drive I/O

Less frequent
    ↓
more lost computation
    ↓
less checkpoint overhead
```

For this project, 50 steps is a reasonable experimental checkpoint interval.

---

# 24. Training Target

The current training output reported:

```text
327,680,000 tokens
```

This comes from:

```text
10,000 optimizer steps
×
32,768 tokens/step
=
327,680,000 tokens
```

Therefore the current configuration is targeting approximately:

> **327.68 million tokens**

Earlier in our design discussion, we considered a target of approximately **300 million tokens**.

For exactly/approximately 300M tokens:

```text
300,000,000
----------------
32,768 tokens/step
≈ 9,155.27
```

Therefore:

```text
ceil(...)
=
9,156 optimizer steps
```

which gives:

```text
9,156 × 32,768
=
300,187,648 tokens
```

So we currently have two possible definitions:

### Current observed configuration

```text
10,000 steps
327,680,000 tokens
```

### 300M-target configuration

```text
9,156 steps
300,187,648 tokens
```

This should be explicitly decided before the final production training run.

---

# 25. Training Time Estimate

At roughly:

```text
~10,000 tokens/sec
```

and approximately:

```text
327.68M tokens
```

the theoretical raw compute time is:

```text
327,680,000 / 10,000
≈ 32,768 sec
≈ 9.10 hours
```

The actual logs showed:

```text
ETA ≈ 8.9–10.2 hours
```

depending on the point in the run.

This is consistent with the measured throughput.

The ETA calculation therefore gives us a useful operational estimate for Colab session planning.

---

# 26. Why Persistent Checkpoints Matter in Colab

Without Drive persistence:

```text
Colab dies
   ↓
RAM/GPU state disappears
   ↓
model disappears
   ↓
optimizer disappears
   ↓
training restarts
```

With our current architecture:

```text
Colab dies
   ↓
temporary compute disappears
   ↓
Google Drive remains
   ↓
checkpoint remains
   ↓
new Colab session
   ↓
load checkpoint
   ↓
resume training
```

This changes the training job from:

> one fragile multi-hour Colab process

into:

> a resumable training process distributed across multiple Colab sessions.

That is the main systems achievement of this phase of the project.

---

# 27. Tests and Benchmarking

The notebook also runs:

```text
pytest tests/ -v
```

and:

```text
scripts/benchmark_training.py
```

before starting the long training process.

This gives us two separate levels of validation:

### Software correctness

```text
pytest
```

checks project functionality.

### Hardware/training performance

```text
benchmark_training.py
```

checks training performance on the available GPU.

Then:

```text
actual training
```

checks the complete pipeline.

This separation is useful because a failure can be classified as:

```text
unit/software issue
        OR
performance issue
        OR
training issue
        OR
environment/Colab issue
```

---

# 28. Why We Did Not Immediately Rewrite the Repository

During the debugging process, we deliberately followed this principle:

> **Don't modify working code until we know the actual failure.**

Initially, we discussed changing:

```text
trainer.py
train_v1.py
train.py
```

to improve checkpoint/resume behavior.

But instead of immediately rewriting everything, we ran the existing setup.

That produced useful evidence.

We discovered:

```text
checkpoint saving already works
```

and later:

```text
checkpoint loading already works
```

and finally:

```text
fresh-session resume works
```

Therefore, a large rewrite was not immediately justified.

This is a much safer engineering workflow.

---

# 29. Current State of the Project

At the point of this documentation, the experimentally verified state is approximately:

```text
                    V1 TRAINING
                         │
                         ▼
                 NVIDIA T4 / CUDA
                         │
                         ▼
                   BF16 enabled
                         │
                         ▼
               Batch size = 8
                         │
                         ▼
          Gradient accumulation = 8
                         │
                         ▼
            Effective batch = 64
                         │
                         ▼
             Sequence length = 512
                         │
                         ▼
          32,768 tokens / optimizer step
                         │
                         ▼
                 Training loss
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Logging              Checkpoint
                                    │
                                    ▼
                             Google Drive
                                    │
                                    ▼
                              step_*.pt
                                    │
                                    ▼
                          Fresh Colab session
                                    │
                                    ▼
                           automatic discovery
                                    │
                                    ▼
                                resume
```

---

# 30. Problems We Encountered and Solutions

## Problem 1 — Training data not found

### Error

```text
FileNotFoundError:
Token file not found: data/tokens/train.bin
```

### Cause

The local Colab data path did not contain the expected token file.

### Solution

Established a clean Drive → local mapping:

```text
Drive:
llm_by_me_data/tokens/train.bin

↓

Local:
llm_by_me/data/tokens/train.bin
```

and similarly for validation and tokenizer.

---

## Problem 2 — Confusion about where logging code belongs

### Question

We discussed whether:

```python
step % self.training_config.log_interval
```

should be changed in the notebook.

### Resolution

No.

That code belongs in:

```text
src/training/trainer.py
```

The notebook is only the execution/orchestration layer.

---

## Problem 3 — Colab runtime is temporary

### Problem

Long training could be lost if Colab terminates.

### Solution

Persistent checkpoints:

```text
/content/drive/MyDrive/
llm_by_me_data/
checkpoints/
v1/
```

---

## Problem 4 — Need automatic resume

### Problem

We did not want to manually select:

```text
step_50.pt
step_100.pt
step_200.pt
```

after every restart.

### Solution

Search:

```text
step_*.pt
```

and select the checkpoint with the largest step number.

---

## Problem 5 — Need optimizer recovery

### Problem

Model weights alone aren't enough for a true training continuation.

### Solution

Checkpoint includes:

```text
model_state_dict
optimizer_state_dict
```

plus training step and RNG state.

---

## Problem 6 — Runtime restart caused nested repository path

### Error

```text
/content/llm_by_me/llm_by_me/scripts/train_v1.py
```

### Cause

The notebook executed:

```python
os.chdir("llm_by_me")
```

when it was already inside the repository directory.

### Solution

Start repository initialization from:

```text
/content
```

and use an explicit repository path.

---

## Problem 7 — Need to know whether fatigue recovery really works

### Solution

We intentionally terminated the runtime.

Then restarted from scratch.

The new session successfully found:

```text
step_200.pt
```

and printed:

```text
Previous step : 200
Resuming from : 201
```

This is our most important validation result so far.

---

## Problem 8 — Perplexity appears absurdly large

### Observation

Perplexity remained:

```text
485165195.41
```

for high losses.

### Cause

Loss is clipped to 20 before exponentiation:

```python
exp(min(loss, 20))
```

### Status

Known logging issue.

It does not invalidate the observed loss reduction.

---

## Problem 9 — Exact data resume semantics

### Observation

The resume logic skips:

```text
start_step × gradient_accumulation_steps
```

micro-batches.

### Concern

There may be a step-indexing off-by-one issue.

Also, shuffled data can prevent exact reproduction of the original batch sequence unless sampler state is persisted.

### Status

**Known technical refinement, not yet fully resolved.**

---

# 31. Current Training Flow

The current intended flow is:

```text
Fresh Colab
    │
    ▼
Force working directory = /content
    │
    ▼
Clone V1 repository
    │
    ▼
Install requirements
    │
    ▼
Mount Google Drive
    │
    ▼
Copy train.bin
Copy validation.bin
Copy tokenizer.json
    │
    ▼
Verify files
    │
    ▼
Run tests
    │
    ▼
Run benchmark
    │
    ▼
Start train_v1.py
    │
    ▼
Trainer initializes
    │
    ▼
Search Drive/checkpoints/v1
    │
    ├───────────────┐
    │               │
No checkpoint    checkpoint found
    │               │
    ▼               ▼
step 0          load checkpoint
                    │
                    ▼
               restore model
                    │
                    ▼
               restore optimizer
                    │
                    ▼
               restore RNG
                    │
                    ▼
             restore step/data
                    │
                    ▼
              continue training
                    │
                    ▼
              periodic logging
                    │
                    ▼
             periodic checkpoint
                    │
                    ▼
              Google Drive
```

---

# 32. What Happens When Colab Dies

Suppose the training reaches:

```text
step 4,720
```

and the latest checkpoint is:

```text
step_4,700.pt
```

Then:

```text
Colab dies
```

The runtime disappears.

But:

```text
Google Drive
└── checkpoints
    └── v1
        └── step_4700.pt
```

remains.

New runtime:

```text
start notebook
      ↓
mount Drive
      ↓
copy data
      ↓
start training
      ↓
find latest checkpoint
      ↓
step_4700.pt
      ↓
load
      ↓
resume
```

The training process does not have to start again at:

```text
step 0
```

---

# 33. What We Have Learned Technically

This project has already touched several important LLM-training systems concepts.

## Gradient accumulation

Instead of requiring a physical batch of 64 sequences, we use:

```text
8 sequences
×
8 accumulation steps
=
64 effective batch
```

This allows a larger effective batch while fitting within the T4's memory constraints.

---

## Mixed precision

The run reports:

```text
BF16 available : True
```

and uses BF16 autocasting where supported.

This reduces memory requirements and can improve GPU throughput.

---

## Gradient clipping

The trainer records:

```text
grad=...
```

and clips gradients before the optimizer update.

This helps control large gradient norms.

---

## Learning-rate scheduling

The logs show the learning rate increasing during the warmup period:

```text
step 0:
6.00e-07

step 50:
3.06e-05

step 100:
6.06e-05

step 220:
1.33e-04
```

So the scheduler is actively controlling the learning rate rather than simply keeping it constant.

---

## Checkpointing

We are storing the complete training state rather than only model weights.

---

## Fault tolerance

The training job can now survive a complete Colab runtime replacement, subject to the caveats around exact data-loader state.

This is an important practical ML systems concept.

---

# 34. Current Architecture vs Ideal Future Architecture

## Current practical architecture

```text
Checkpoint
    │
    ├── model state
    ├── optimizer state
    ├── RNG
    └── step
          │
          ▼
skip N data batches
          │
          ▼
continue
```

## More rigorous future architecture

For a production-quality trainer, we would ideally save:

```text
model state
optimizer state
scheduler state
gradient scaler state if applicable
global step
epoch
micro-step
random states
dataloader state
sampler state
dataset cursor
configuration
```

Then:

```text
checkpoint
     ↓
exact state reconstruction
     ↓
exact next batch
     ↓
exact continuation
```

That would remove most of the current resume ambiguity.

---

# 35. Recommended Next Engineering Tasks

Now that the fundamental system has been proven, the next improvements should be made in this order.

## Phase 1 — Fix checkpoint step semantics

Define explicitly:

```text
step = number of completed optimizer updates
```

Then checkpoint names should represent completed updates consistently.

For example:

```text
step_50.pt
```

should unambiguously mean:

> 50 optimizer updates have completed.

This removes the current zero-based ambiguity.

---

## Phase 2 — Fix data-position restoration

Instead of relying only on:

```text
step × gradient accumulation
```

we should define the exact number of consumed micro-batches.

Potentially store:

```text
global_step
micro_step
data_position
```

inside the checkpoint.

---

## Phase 3 — Decide shuffle behavior

For this fixed packed V1 dataset, we can determine whether:

```python
shuffle=False
```

is desirable for deterministic sequential training.

If we retain shuffling, we should preserve the relevant sampler RNG/state.

---

## Phase 4 — Fix perplexity logging

Instead of:

```python
exp(min(loss, 20))
```

we could log:

```text
PPL = exp(loss)
```

when numerically safe, or explicitly report:

```text
PPL > X
```

when the value is outside a meaningful display range.

---

## Phase 5 — Decide final token target

Choose one:

```text
300M target
```

or:

```text
327.68M target
```

Currently the observed configuration corresponds to:

```text
10,000 × 32,768
=
327,680,000 tokens
```

If we want approximately 300M:

```text
9,156 steps
=
300,187,648 tokens
```

---

## Phase 6 — Long-duration training

Once the resume semantics are cleaned up:

```text
full training
    ↓
checkpoint every 50 steps
    ↓
validation periodically
    ↓
monitor loss
    ↓
monitor throughput
    ↓
monitor ETA
    ↓
recover automatically after Colab failure
```

Then the model can be trained across multiple Colab sessions.

---

# 36. Important Distinction: What Has Been Proven

It is worth recording this very explicitly.

### Proven experimentally

```text
Dataset loading                         ✅
GPU training                           ✅
BF16                                    ✅
Gradient accumulation                  ✅
Training loss decreases                ✅
~9–10k tok/s throughput                ✅
Checkpoint written to Drive            ✅
Multiple checkpoint files              ✅
Runtime termination                     ✅
Fresh runtime                           ✅
Drive checkpoint discovery             ✅
Model/optimizer checkpoint loading     ✅
Resume from previous step              ✅
Training continues after restart       ✅
```

### Identified but not fully perfected

```text
Exact dataloader continuation           ⚠️
Exact sampler restoration               ⚠️
Step numbering semantics                ⚠️
Perplexity display                      ⚠️
Final 300M vs 327.68M target            ⚠️
```

This is a much more accurate status than simply saying "checkpointing works."

---

# 37. Final Project Mental Model

The most useful way to think about the entire system is:

```text
                         LLM V1
                           │
                           ▼
                     Training State
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
       Model            Optimizer          RNG
       weights           state             state
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                       Checkpoint
                           │
                           ▼
                    Google Drive
                           │
                 persistent storage
                           │
          ┌────────────────┴────────────────┐
          │                                 │
     Colab session #1                 Colab session #2
          │                                 │
     GPU computation                  GPU computation
          │                                 │
          └─────────── checkpoint ──────────┘
```

So the project is no longer just:

> "train an LLM in Colab."

It is becoming:

> **A fault-tolerant, resumable LLM training pipeline where ephemeral GPU compute is separated from persistent training state.**

That is the key systems-engineering result of the work we've done so far.

---

# 38. Current Project Status — One-Page Summary

```text
PROJECT
-------
llm_by_me
V1


COMPUTE
-------
Google Colab
NVIDIA T4
CUDA
BF16


TRAINING
--------
Batch size              = 8
Gradient accumulation   = 8
Effective batch         = 64
Sequence length         = 512
Tokens / optimizer step = 32,768


OBSERVED PERFORMANCE
--------------------
~9,000–10,000 tokens/sec


CURRENT TOKEN TARGET
--------------------
10,000 steps
327,680,000 tokens


PERSISTENT DATA
---------------
Google Drive

llm_by_me_data/
├── tokens/
│   ├── train.bin
│   └── validation.bin
├── tokenizer/
│   └── tokenizer.json
└── checkpoints/
    └── v1/
        └── step_*.pt


CHECKPOINT
----------
Persistent on Drive
Automatic latest-checkpoint discovery
Model state saved
Optimizer state saved
RNG state saved
Training step saved


FATIGUE TEST
------------
Runtime terminated
        ↓
Fresh runtime
        ↓
Checkpoint discovered
        ↓
step_200.pt loaded
        ↓
Resume step = 201
        ↓
Training continued


KNOWN ISSUES
------------
1. Data-position off-by-one semantics
2. Exact DataLoader/sampler state isn't preserved
3. Perplexity clipping during high loss
4. Need final decision: 300M vs 327.68M tokens
5. Checkpoint step semantics should be made explicit


CURRENT CONCLUSION
------------------
Core training works.
Persistent Drive storage works.
Checkpoint saving works.
Automatic resume works.
Fresh-runtime fatigue test passed.

Next work should focus on making resume
semantics exact and production-quality,
rather than redesigning the entire system.
```

The current Colab notebook itself reflects the GPU setup, Drive persistence, data copying, testing/benchmarking, and live training launch described above.  The current trainer implementation contains the checkpoint discovery, resume, data-position advancement, logging, validation, and checkpoint-saving logic we tested.  

**The most important milestone so far is the fatigue test: we intentionally killed the Colab runtime, started a new one, and it successfully discovered `step_200.pt` from Google Drive and continued from step 201.** That is the core functionality we set out to establish.


