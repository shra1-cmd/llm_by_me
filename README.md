


# 🧠 llm_by_me

### Building a Language Model From Scratch — V1

> A hands-on implementation of a small autoregressive language model, built from the ground up to understand the complete LLM stack — from text and tokenization to transformer computation, training, checkpointing, and inference.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/Status-V1_Training-FFD43B?style=for-the-badge)

</p>

---

## ✦ What is `llm_by_me`?

`llm_by_me` is a from-scratch LLM engineering project.

The goal is not simply to call:

```python
model = AutoModelForCausalLM.from_pretrained(...)
````

and generate text.

The goal is to understand what happens **inside** the model.

That means implementing and connecting the major pieces ourselves:

```text
                    RAW TEXT
                       │
                       ▼
              ┌─────────────────┐
              │     Dataset     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    Tokenizer    │
              └────────┬────────┘
                       │
                       ▼
                  Token IDs
                       │
                       ▼
              ┌─────────────────┐
              │   Embeddings    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Transformer   │
              │     Blocks      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     LM Head     │
              └────────┬────────┘
                       │
                       ▼
              Next Token
                       │
                       ▼
                  Generation
```

The project is therefore both a **machine-learning project** and a **systems-engineering exercise**.

---

# 🚀 Project Goals

The V1 project is built around a few core objectives:

* Build a language model rather than only consume one.
* Understand the Transformer computation path.
* Build the data → token → batch → model pipeline.
* Implement causal language-model training.
* Understand optimization and learning-rate scheduling.
* Implement checkpointing and restoration.
* Run training on GPU hardware.
* Make training persistent across temporary Colab runtimes.
* Build a clean separation between model, data, training, and inference.
* Validate individual components instead of treating the model as a black box.

---

# 🧩 V1 Architecture

At a high level:

```text
                 ┌──────────────────────┐
                 │      Raw Dataset     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Tokenizer       │
                 │                      │
                 │      BPE             │
                 └──────────┬───────────┘
                            │
                            ▼
                    Packed Token IDs
                            │
                            ▼
                 ┌──────────────────────┐
                 │     DataLoader       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   V1 Language Model  │
                 │                      │
                 │  Embedding           │
                 │      ↓               │
                 │  Transformer Blocks  │
                 │      ↓               │
                 │  LM Head             │
                 └──────────┬───────────┘
                            │
                            ▼
                      Cross Entropy
                            │
                            ▼
                      Backpropagation
                            │
                            ▼
                      Optimizer Step
                            │
                            ▼
                      Checkpoint
```

---

# 🏗️ Model

V1 implements a causal language-model architecture.

The forward path can be represented as:

```text
Token IDs
   │
   ▼
Token Embedding
   │
   ▼
Transformer Block
   │
   ├── Normalization
   │
   ├── Self Attention
   │
   ├── Residual Connection
   │
   ├── Normalization
   │
   ├── Feed Forward Network
   │
   └── Residual Connection
   │
   ▼
Repeated Transformer Blocks
   │
   ▼
Final Normalization
   │
   ▼
Language Model Head
   │
   ▼
Vocabulary Logits
```

The model is trained using the standard causal language-model objective:

```text
Given:

The cat sat on the

Predict:

mat
```

More generally:

```text
x₀ x₁ x₂ ... xₙ

      ↓

predict x₁
predict x₂
predict x₃
...
predict xₙ₊₁
```

The model therefore learns next-token prediction.

---

# 🔬 Transformer Computation

The attention mechanism follows the familiar:

```text
Q = XWq
K = XWk
V = XWv
```

followed by scaled attention:

```text
Attention(Q,K,V)

       QKᵀ
        │
        ▼
   scale by √dₖ
        │
        ▼
 causal mask
        │
        ▼
      Softmax
        │
        ▼
        × V
```

The resulting representation is passed through the remaining Transformer computation.

This project is intentionally structured so that these operations can be inspected and debugged individually.

---

# 📦 Data Pipeline

The data path is:

```text
Dataset
   │
   ▼
Text
   │
   ▼
Tokenizer
   │
   ▼
Integer Token IDs
   │
   ▼
Packed Binary Data
   │
   ├───────────────┐
   ▼               ▼
train.bin     validation.bin
   │               │
   ▼               ▼
DataLoader      DataLoader
   │               │
   └───────┬───────┘
           ▼
       Training
```

The training setup uses packed token data so the training loop can consume fixed-length sequences efficiently.

---

# 🔤 Tokenization

V1 uses a BPE-style tokenizer.

The tokenizer converts:

```text
text
 ↓
subword tokens
 ↓
integer IDs
```

Example conceptually:

```text
"I love AI"

        ↓

["I", " love", " AI"]

        ↓

[ token_id_1,
  token_id_2,
  token_id_3 ]
```

The model never receives raw text directly.

It operates on integer token IDs.

---

# 🧮 Training Objective

V1 is trained as a causal language model.

For each sequence:

```text
Input:

[t₀, t₁, t₂, t₃, t₄]

Target:

[t₁, t₂, t₃, t₄, t₅]
```

The model produces vocabulary logits:

```text
[B, T, V]
```

where:

```text
B = batch size
T = sequence length
V = vocabulary size
```

The loss is computed using next-token prediction.

---

# ⚙️ Training Pipeline

The training loop follows:

```text
             Batch
               │
               ▼
          Input IDs
               │
               ▼
          Forward Pass
               │
               ▼
        Vocabulary Logits
               │
               ▼
         Cross Entropy
               │
               ▼
          loss.backward()
               │
               ▼
        Gradient Clipping
               │
               ▼
        Learning Rate Update
               │
               ▼
        Optimizer Step
               │
               ▼
         Next Training Step
```

Gradient accumulation is used to increase the effective batch size without requiring the entire effective batch to fit into GPU memory simultaneously.

---

# 📊 V1 Training Configuration

The current Colab training configuration uses:

| Parameter               |               Value |
| ----------------------- | ------------------: |
| Device                  |            CUDA GPU |
| GPU used during testing |           NVIDIA T4 |
| Precision               | BF16 when supported |
| Sequence length         |                 512 |
| Micro batch size        |                   8 |
| Gradient accumulation   |                   8 |
| Effective batch size    |        64 sequences |
| Tokens / optimizer step |              32,768 |
| Target training tokens  |                300M |

Effective token calculation:

```text
8 × 8 × 512

= 32,768 tokens / optimizer step
```

The 300M-token target corresponds to approximately:

```text
300,000,000 / 32,768
≈ 9,155 optimizer steps
```

The resumable runner uses the ceiling of this value.

---

# 🔥 Mixed Precision

Where supported, V1 training uses BF16 autocasting:

```text
FP32
  │
  ├── model parameters / optimizer state
  │
  ▼
BF16 compute
  │
  ▼
GPU acceleration
```

The training runner checks CUDA and BF16 support before enabling BF16.

This allows the training computation to use reduced precision while maintaining the required training state.

---

# 🧠 Optimization

The training system separates optimizer construction from the training loop.

Conceptually:

```text
Model Parameters
       │
       ▼
    Gradients
       │
       ▼
Gradient Clipping
       │
       ▼
Learning Rate Scheduler
       │
       ▼
   Optimizer
       │
       ▼
Updated Parameters
```

This separation keeps the training components independently testable.

---

# 📈 Learning Rate

The learning-rate scheduler is implemented separately from the optimizer.

The training step determines the current learning rate based on:

```text
training step
      │
      ▼
warmup
      │
      ▼
scheduled learning rate
      │
      ▼
optimizer
```

This keeps scheduling logic independent from model architecture.

---

# 💾 Checkpointing

Long-running training cannot rely on the lifetime of a temporary compute environment.

V1 therefore uses persistent checkpoints.

A checkpoint can contain the training state required to continue the run, including:

```text
┌───────────────────────────┐
│       Checkpoint          │
├───────────────────────────┤
│ Model state               │
│ Optimizer state           │
│ Training step             │
│ Loss                      │
│ Configuration             │
│ Training configuration    │
└───────────────────────────┘
```

The project uses dedicated:

```text
save_checkpoint()
load_checkpoint()
```

functions rather than embedding checkpoint logic throughout the training loop.

---

# ☁️ Persistent Colab Training

The V1 training environment uses:

```text
GitHub
   │
   ▼
Colab Runtime
   │
   ├── Model
   ├── Training
   └── GPU
   │
   ▼
Google Drive
   │
   ├── train.bin
   ├── validation.bin
   ├── tokenizer.json
   └── checkpoints/
```

The important distinction is:

```text
Colab = compute
Drive = persistence
GitHub = source code
```

If the Colab runtime disappears, the persistent training data and checkpoints remain available.

---

# 🔄 Automatic Resume

The Colab-specific training runner searches the persistent checkpoint directory for:

```text
step_*.pt
```

It selects the latest checkpoint and restores the model and optimizer.

Conceptually:

```text
Start
  │
  ▼
Checkpoint exists?
  │
 ┌┴──────────────┐
 │               │
NO              YES
 │               │
 ▼               ▼
Step 0       Find latest
                 │
                 ▼
             Load state
                 │
                 ▼
            Resume step
                 │
                 ▼
              Training
```

No manual checkpoint selection is required.

---

# 🧪 Validation

Training and validation are separated.

During validation:

```text
model.eval()
      │
      ▼
no_grad()
      │
      ▼
validation batches
      │
      ▼
validation loss
```

The model is returned to training mode afterwards.

This prevents validation computation from changing training gradients.

---

# 🧰 Repository Structure

```text
V1/
│
├── configs/
│   ├── v1.py
│   └── train.py
│
├── data/
│   └── ...
│
├── scripts/
│   ├── train_v1.py
│   └── colab_train.py
│
├── src/
│   ├── data/
│   ├── inference/
│   ├── model/
│   ├── tokenizer/
│   ├── training/
│   └── utils/
│
├── tests/
│
├── README.md
│
└── docs/
    └── V1_DETAILED_DOCUMENTATION.md
```

---

# 🧱 Code Organization

The project is intentionally divided into logical layers.

### `src/model/`

Contains the language-model implementation.

```text
model
 ├── embeddings
 ├── transformer blocks
 ├── attention
 ├── feed-forward layers
 ├── normalization
 └── LM head
```

### `src/data/`

Responsible for loading and preparing training data.

### `src/tokenizer/`

Contains tokenizer functionality.

### `src/training/`

Contains training infrastructure:

```text
checkpointing
optimizer
scheduler
trainer
```

### `src/inference/`

Contains inference/generation functionality.

### `configs/`

Contains model and training configuration.

### `tests/`

Contains component-level validation.

### `scripts/`

Contains executable entry points.

---

# 🧪 Engineering Validation

The project was developed incrementally rather than writing the entire model and immediately launching a long training run.

Validation has included:

```text
Tokenizer
   ↓
Data
   ↓
Model construction
   ↓
Forward pass
   ↓
Loss
   ↓
Backward pass
   ↓
Optimizer
   ↓
Training loop
   ↓
Checkpoint
   ↓
Checkpoint restoration
```

This makes debugging substantially easier because failures can be isolated to individual layers.

---

# 🐛 Engineering Problems Solved

A major part of this project is the debugging process.

Some of the engineering problems addressed include:

### 1. Data pipeline validation

Ensuring the training system receives the expected fixed-length token sequences.

```text
raw data
   ↓
token IDs
   ↓
packed storage
   ↓
batch
   ↓
model
```

---

### 2. Training-state persistence

A temporary Colab runtime is not reliable persistent infrastructure.

The solution was to separate:

```text
compute environment
        ≠
persistent training state
```

and store checkpoints on Google Drive.

---

### 3. Checkpoint discovery

Instead of hardcoding:

```text
step_500.pt
```

the resumable runner searches:

```text
step_*.pt
```

and selects the latest checkpoint.

---

### 4. Data position during resume

A checkpoint restores model and optimizer state, but the current DataLoader does not directly persist its exact iterator position.

The V1 Colab runner therefore advances the fixed packed dataset by the number of completed micro-batches when resuming.

This is sufficient for the current single-GPU V1 setup, while leaving room for a more sophisticated dataloader-state mechanism in future versions.

---

# 🧪 Experiments

The project follows an experimental workflow:

```text
Implement
   ↓
Run
   ↓
Inspect
   ↓
Measure
   ↓
Debug
   ↓
Modify
   ↓
Validate
```

Rather than assuming that an implementation is correct because it executes successfully, individual components are inspected and tested.

---

# 🔍 Debugging Philosophy

One of the primary goals of `llm_by_me` is to make the internal computation understandable.

For example, instead of treating attention as:

```python
attention(...)
```

as a black box, the implementation can be examined through:

```text
Input
 ↓
Q projection
 ↓
K projection
 ↓
V projection
 ↓
Attention scores
 ↓
Mask
 ↓
Softmax
 ↓
Weighted values
 ↓
Output projection
```

The same principle applies to the training system.

---

# 📚 Detailed Documentation

The README intentionally contains only the high-level technical picture.

The complete engineering record is maintained separately.

It contains:

* Detailed implementation notes
* Debugging sessions
* Experiments
* Training setup
* Checkpoint behavior
* Colab persistence
* Problems encountered
* Solutions
* Design decisions
* Commands
* Validation details
* Training observations

### 👉 [Read the complete V1 engineering documentation](Docs/V1_DETAILED_DOCUMENTATION.md)

---

# ▶️ Quick Start

Clone the repository:

```bash
git clone -b V1 https://github.com/shra1-cmd/llm_by_me.git
cd llm_by_me
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest tests/ -v
```

---

# 🏋️ Training

The normal V1 training entry point is:

```bash
PYTHONPATH="$(pwd)" python scripts/train_v1.py
```

For persistent Colab training, the dedicated runner is:

```bash
PYTHONPATH="$(pwd)" \
COLAB_CHECKPOINT_DIR="/content/drive/MyDrive/llm_by_me_data/checkpoints/v1" \
python scripts/colab_train.py
```

The Colab runner is designed to:

```text
load data
   ↓
construct model
   ↓
find latest checkpoint
   ↓
restore training state
   ↓
continue training
   ↓
periodically save checkpoints
```

---

# ☁️ Colab Workflow

The intended persistent workflow is:

```text
1. Start Colab
       │
       ▼
2. Connect Google Drive
       │
       ▼
3. Clone V1
       │
       ▼
4. Restore dataset
       │
       ▼
5. Restore tokenizer
       │
       ▼
6. Find latest checkpoint
       │
       ▼
7. Resume training
       │
       ▼
8. Save checkpoints to Drive
```

If the runtime terminates:

```text
Colab Runtime
      ✕
      │
      ▼
Google Drive
      │
      ├── dataset
      ├── tokenizer
      └── checkpoints
             │
             ▼
       New Colab Runtime
             │
             ▼
          Resume
```

---

# 📊 Training State

The current V1 target is:

```text
Target:

300,000,000 tokens
```

With:

```text
Batch size              = 8
Gradient accumulation   = 8
Sequence length         = 512
```

the effective token throughput per optimizer step is:

```text
8 × 8 × 512

= 32,768 tokens
```

Therefore the target corresponds to approximately:

```text
9,156 optimizer steps
```

---

# 🗺️ Roadmap

The project is intentionally being developed in versions.

## V1 — Foundation

```text
[x] Dataset pipeline
[x] Tokenization
[x] Model implementation
[x] Training pipeline
[x] Optimizer
[x] Scheduler
[x] Checkpointing
[x] Validation
[x] GPU training
[x] Persistent Colab training
[ ] Complete long training run
[ ] Analyze final model behavior
```

---

## V2 — Training Improvements

Potential future work:

```text
┌────────────────────────────┐
│ Better data pipeline       │
├────────────────────────────┤
│ Better checkpoint recovery │
├────────────────────────────┤
│ Training instrumentation   │
├────────────────────────────┤
│ More extensive evaluation  │
└────────────────────────────┘
```

---

## V3 — Systems

The longer-term direction is toward LLM systems engineering:

```text
Training
   │
   ▼
Inference
   │
   ▼
KV Cache
   │
   ▼
Memory Management
   │
   ▼
Quantization
   │
   ▼
GPU Kernels
   │
   ▼
Inference Optimization
```

This project therefore serves as the foundation for exploring the complete LLM systems stack.

---

# 🎯 Why Build From Scratch?

Modern LLM libraries make it incredibly easy to run a model.

That is useful.

But:

```python
model.generate(...)
```

does not explain:

```text
Where did the tokens come from?
How are embeddings produced?
How is attention computed?
Where does the causal mask act?
How are gradients accumulated?
How does the optimizer update weights?
How does checkpoint restoration work?
What actually happens during generation?
```

`llm_by_me` is an attempt to answer those questions by implementing the pipeline directly.

---

# 🧠 Learning Through Implementation

The project follows a simple principle:

> **Don't just use the abstraction. Understand the abstraction.**

Instead of stopping at:

```text
Transformer
```

the project goes deeper:

```text
Transformer
    ↓
Attention
    ↓
Q / K / V
    ↓
Matrix operations
    ↓
Tensor shapes
    ↓
GPU computation
```

And instead of stopping at:

```text
train()
```

the project investigates:

```text
batch
 ↓
forward
 ↓
loss
 ↓
backward
 ↓
gradients
 ↓
clipping
 ↓
scheduler
 ↓
optimizer
 ↓
checkpoint
```

---

# 🧪 From Model Engineering to Systems Engineering

The ultimate direction of this project is broader than building one small language model.

The progression is:

```text
                 llm_by_me
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Model         Data        Training
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
                 Inference
                     │
                     ▼
                 KV Cache
                     │
                     ▼
               Optimization
                     │
                     ▼
              GPU / CUDA
                     │
                     ▼
              LLM Systems
```

The project is intended to evolve from understanding **how an LLM works** to understanding **how an LLM system works**.

---

# 📌 Current Status

### V1 Foundation

```text
Model              ████████████████████  Implemented
Tokenizer          ████████████████████  Implemented
Dataset            ████████████████████  Implemented
Training           ████████████████████  Implemented
Checkpointing      ████████████████████  Implemented
Validation         ████████████████████  Implemented
GPU Training       ████████████████████  Implemented
Colab Persistence  ████████████████████  Implemented
Long Training      ░░░░░░░░░░░░░░░░░░░░  In progress
```

---

# 📖 Project Philosophy

```text
Understand
    ↓
Implement
    ↓
Measure
    ↓
Debug
    ↓
Validate
    ↓
Optimize
```

The objective is not to reproduce a production-scale LLM.

The objective is to build the mental and technical foundation required to work on one.

---

# ⭐ Project Highlights

```text
✓ Language model implemented from scratch
✓ Custom tokenizer/data pipeline
✓ Transformer-based causal LM
✓ GPU training
✓ BF16 training support
✓ Gradient accumulation
✓ Gradient clipping
✓ Learning-rate scheduling
✓ Validation loop
✓ Persistent checkpointing
✓ Automatic checkpoint resume
✓ Colab + Google Drive training workflow
✓ Component-level testing
✓ Debug-first development
```

---

# 📁 Documentation

| Document                            | Purpose                               |
| ----------------------------------- | ------------------------------------- |
| `README.md`                         | Project overview and quick start      |
| `docs/V1_DETAILED_DOCUMENTATION.md` | Complete technical engineering record |
| `configs/`                          | Model and training configuration      |
| `src/`                              | Core implementation                   |
| `scripts/`                          | Training and execution entry points   |
| `tests/`                            | Validation and tests                  |

---

# 🔗 Project Structure at a Glance

```text
                    llm_by_me
                       │
              ┌────────┴────────┐
              │                 │
             V1               docs
              │                 │
       ┌──────┼──────┐          │
       │      │      │          ▼
     Model   Data  Training   Detailed
       │      │      │       Engineering
       │      │      │       Documentation
       └──────┼──────┘
              │
              ▼
          Inference
```

---

# 🚀 Final Goal

The long-term objective is to move through the entire stack:

```text
              LLM
               │
       ┌───────┴────────┐
       ▼                ▼
    Training         Inference
       │                │
       ▼                ▼
   Optimizer          KV Cache
       │                │
       ▼                ▼
  Distributed       Batching
   Training             │
       │                ▼
       └────────► Optimization
                       │
                       ▼
                    CUDA
                       │
                       ▼
                  GPU Kernels
                       │
                       ▼
                LLM Systems 🚀
```

---

## 🧠 Built to Understand, Not Just Run.

**`llm_by_me` — V1**

From tokens → tensors → transformers → training → inference.

---

```

This version is intentionally **cleaner than the 2,400-line document**: the GitHub visitor gets the architecture, engineering scope, training setup, current status, and commands without having to walk through every debugging session. The detailed document remains the place for the exhaustive Colab/checkpoint and engineering history. 
```
