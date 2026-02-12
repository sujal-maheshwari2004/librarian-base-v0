# 📚 Librarian-Base-v0

*A Lightweight GPT Model Built from Scratch*

Librarian-Base-v0 is a compact GPT-style transformer trained end-to-end in PyTorch.
The goal of this project is to deeply understand language model internals while maintaining a clean, scalable architecture suitable for experimentation on modest hardware.

This is not a wrapper around HuggingFace — it is a full custom implementation.

---

# 🧠 Model Architecture

| Component             | Value                 |
| --------------------- | --------------------- |
| Architecture          | GPT-style Transformer |
| Layers                | 2                     |
| Attention Heads       | 4                     |
| Embedding Dimension   | 128                   |
| Feedforward Dimension | 256                   |
| Context Length        | 128 tokens            |
| Vocabulary Size       | 8000 (Byte-Level BPE) |
| Weight Tying          | Enabled               |
| Dropout               | 0.2                   |
| Total Parameters      | **1,305,600 (~1.3M)** |

### Design Choices

* Pre-LayerNorm Transformer blocks
* Manual causal self-attention with masking
* GELU activation
* AdamW optimizer
* Mixed Precision (AMP)
* Early stopping
* Best-checkpoint saving

The architecture is intentionally compact to allow fast iteration and experimentation on 8GB GPUs.

---

# 📦 Dataset

* Dataset: TinyStories
* Subsampled to 2% of total corpus
* Cleaned and normalized
* Byte-Level BPE tokenizer trained from scratch
* Vocabulary size: 8000
* Special tokens included

The dataset pipeline supports:

* Chunk-based tokenization with progress tracking
* Sliding window autoregressive training
* Automatic train / eval / test splitting

---

# 🏋️ Training Configuration

| Setting               | Value                |
| --------------------- | -------------------- |
| Optimizer             | AdamW                |
| Learning Rate         | 1e-4                 |
| Batch Size            | 32                   |
| Gradient Accumulation | 1                    |
| Mixed Precision       | Enabled              |
| Early Stopping        | Enabled (patience=5) |
| Hardware Target       | 8GB GPU              |

Training includes:

* Automatic checkpointing per epoch
* Best model selection based on validation loss
* Validation monitoring every epoch

---

# 📊 Evaluation Results

Evaluation performed on held-out validation set.

```
Cross-Entropy Loss : 2.5466
Perplexity         : 12.7630
Top-1 Accuracy     : 45.37%
Top-5 Accuracy     : 72.45%
Top-10 Accuracy    : 80.23%
```

### Interpretation

* **Perplexity ≈ 12.76**
  Indicates strong next-token modeling for a 1.3M parameter transformer.

* **Top-1 Accuracy ≈ 45%**
  The exact next token is predicted correctly nearly half the time.

* **Top-5 Accuracy ≈ 72%**
  The correct token appears within the top 5 predictions most of the time.

For a compact transformer trained on a limited data fraction, these metrics demonstrate stable convergence and meaningful language structure learning.

---

# 💬 Inference & Sampling

Generation supports:

* Temperature scaling
* Top-K sampling
* Top-P (nucleus) sampling

Run interactive chat:

```bash
python chat_librarian.py
```

The model autoregressively generates tokens conditioned on:

```
User: ...
Assistant:
```

---

# 🗂 Project Structure

```
model.py              → GPT architecture implementation
dataset.py            → Tokenization + dataset pipeline
train.py              → Training loop
test.py               → Evaluation metrics
chat_librarian.py     → Inference interface
utills/               → Data cleaning & tokenizer training
```

---

# 🔬 Technical Highlights

This project demonstrates:

* Manual implementation of causal attention
* Weight tying between embeddings and output head
* Mixed precision training
* Early stopping logic
* Tokenizer training pipeline
* Evaluation with perplexity and top-k accuracy
* End-to-end LLM workflow (data → tokenizer → model → training → evaluation → inference)

It is designed to be easily extensible for:

* Deeper architectures
* Larger context windows
* Rotary embeddings
* RMSNorm / SwiGLU
* Larger parameter counts (10M+)

---

# 🚀 Future Directions

Potential next steps:

* Increase depth to 4–6 layers
* Expand context length to 256+
* Add cosine learning rate schedule
* Introduce gradient clipping
* Implement rotary embeddings
* Scale to 10M+ parameters
* Add supervised chat fine-tuning

---

# 🎯 Purpose

Librarian-Base-v0 serves as:

* A research playground for transformer experimentation
* A foundation for scaling small LLMs
* A portfolio-grade demonstration of full-stack LLM development
* A minimal but complete GPT training ecosystem
