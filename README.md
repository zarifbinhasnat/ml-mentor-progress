# ML Mentor — Daily Learning Journal

A structured, 112-concept journey through machine learning — from the first artificial neuron to production inference stacks. One lesson per day, automatically generated and committed.

> This repository is **not related** to [class-ai-companion](https://github.com/zarifbinhasnat/class-ai-companion).

---

## How it works

Every day at 8:00 AM, an automated agent:
1. Reads `progress.json` to determine where the journey left off
2. Generates a full lesson for the next concept in the roadmap
3. Commits the lesson as `LESSON_YYYY_MM_DD_DAY_N.md` and pushes it here

Each lesson has six fixed sections:
- **🧠 Concept of the Day** — intuition, math, and an interview question
- **🐍 Pythonic Edge** — one useful PyTorch/NumPy trick
- **📡 Signal Lab** — a DSP-flavored problem tied to the concept
- **🏋️ The Gauntlet** — a coding challenge (C++) with escalating hints
- **🏗️ Blueprint** — system design nugget (~2× per week)
- **🗺️ Marching Orders** — what's next

---

## The Roadmap

### Phase 1 — Foundations (Concepts 1–12)
Core building blocks: neurons, activations, loss functions, backprop, and gradient descent.

| # | Concept |
|---|---------|
| 1 | Artificial neuron & forward pass |
| 2 | Sigmoid & tanh — why nonlinearity |
| 3 | ReLU, LeakyReLU, GELU, SiLU |
| 4 | MSE loss |
| 5 | Softmax + cross-entropy |
| 6 | Computational graph |
| 7 | Chain rule refresher |
| 8 | Backprop I — credit-assignment intuition |
| 9 | Backprop II — full math (one hidden layer) |
| 10 | Gradient descent (batch / SGD / mini-batch) |
| 11 | The learning rate |
| 12 | Epochs, batches, iterations |

### Phase 2 — Optimization (Concepts 13–23)
Making training fast and stable.

| # | Concept |
|---|---------|
| 13 | Momentum |
| 14 | Nesterov |
| 15 | AdaGrad & RMSprop |
| 16 | Adam & AdamW |
| 17 | LR schedules (step / cosine / warmup) |
| 18 | Weight init: zeros fail; Xavier/Glorot |
| 19 | He init (ReLU nets) |
| 20 | Vanishing gradients |
| 21 | Exploding gradients & clipping |
| 22 | Bias–variance tradeoff |
| 23 | Reading loss curves |

### Phase 3 — Regularization (Concepts 24–31)
Keeping models from memorizing noise.

| # | Concept |
|---|---------|
| 24 | L2 weight decay |
| 25 | L1 & sparsity |
| 26 | Dropout |
| 27 | Batch normalization |
| 28 | Layer normalization |
| 29 | Data augmentation |
| 30 | Early stopping |
| 31 | Label smoothing |

### Phase 4 — Convolutional Networks (Concepts 32–44)
Spatial inductive bias and the architectures that defined a decade.

| # | Concept |
|---|---------|
| 32 | The convolution operation |
| 33 | Kernels, stride, padding |
| 34 | Feature maps & channels |
| 35 | Pooling & downsampling |
| 36 | Receptive field |
| 37 | 1×1 conv & bottlenecks |
| 38 | LeNet & AlexNet |
| 39 | VGG |
| 40 | Inception / GoogLeNet |
| 41 | ResNet & residual connections |
| 42 | DenseNet |
| 43 | Transposed conv / upsampling |
| 44 | Dilated / atrous conv |

### Phase 5 — Sequence Models (Concepts 45–52)
Learning over time before attention took over.

| # | Concept |
|---|---------|
| 45 | RNN cell |
| 46 | Backprop through time (BPTT) |
| 47 | Why RNNs forget |
| 48 | LSTM gates & cell state |
| 49 | GRU |
| 50 | Bidirectional RNNs |
| 51 | Seq2seq encoder–decoder |
| 52 | Fixed-context bottleneck |

### Phase 6 — Attention & Transformers (Concepts 53–64)
The mechanism that rewrote everything.

| # | Concept |
|---|---------|
| 53 | Attention intuition (soft lookup) |
| 54 | Query, Key, Value |
| 55 | Scaled dot-product — why √d |
| 56 | Multi-head attention |
| 57 | Self- vs cross-attention |
| 58 | Sinusoidal positional encoding |
| 59 | Learned & rotary (RoPE) positions |
| 60 | Transformer block (residual + LN + FFN) |
| 61 | Encoder-only vs decoder-only vs enc-dec |
| 62 | Causal masking |
| 63 | O(n²) cost & long-context pain |
| 64 | FlashAttention |

### Phase 7 — Language Models & Fine-tuning (Concepts 65–77)
From tokenization to instruction-tuned giants.

| # | Concept |
|---|---------|
| 65 | Tokenization (BPE / WordPiece / SentencePiece) |
| 66 | Embeddings → contextual embeddings |
| 67 | BERT & masked LM |
| 68 | GPT & causal LM |
| 69 | Transfer learning & fine-tuning |
| 70 | In-context learning & prompting |
| 71 | Instruction tuning |
| 72 | RLHF overview |
| 73 | LoRA |
| 74 | Quantization (int8 / int4) & QLoRA |
| 75 | Knowledge distillation |
| 76 | Mixture of Experts |
| 77 | Scaling laws |

### Phase 8 — Generative Models (Concepts 78–88)
VAEs, GANs, and diffusion — how machines learn to create.

| # | Concept |
|---|---------|
| 78 | Autoencoders |
| 79 | VAE & reparameterization trick |
| 80 | VAE decoder & upsampling artifacts |
| 81 | GANs (minimax game) |
| 82 | GAN instability & Wasserstein GAN |
| 83 | Diffusion: forward noising |
| 84 | Diffusion: reverse denoising & objective |
| 85 | Score-based generative models |
| 86 | Classifier-free guidance |
| 87 | Latent diffusion (Stable Diffusion) |
| 88 | Spectral signatures of generated images |

### Phase 9 — Vision & Multimodal (Concepts 89–95)
Self-supervised vision, contrastive learning, and detection.

| # | Concept |
|---|---------|
| 89 | Vision Transformer (ViT) |
| 90 | CLIP |
| 91 | SimCLR |
| 92 | Masked autoencoders (MAE) |
| 93 | DINO & self-distillation |
| 94 | U-Net & skip connections |
| 95 | Object detection (anchors, IoU, NMS, YOLO) |

### Phase 10 — Efficiency & Inference (Concepts 96–103)
Making models fast enough to actually ship.

| # | Concept |
|---|---------|
| 96 | Mixed precision (fp16 / bf16) |
| 97 | Gradient accumulation & checkpointing |
| 98 | Data / model / pipeline parallelism |
| 99 | ZeRO & sharded training |
| 100 | KV cache |
| 101 | Throughput vs latency, batching |
| 102 | ONNX / TorchScript export |
| 103 | Serving (vLLM, continuous batching) |

### Phase 11 — RL, Agents & Frontiers (Concepts 104–112)
Reinforcement learning, RAG, and where the field is heading.

| # | Concept |
|---|---------|
| 104 | RL basics (MDP, reward, policy) |
| 105 | Q-learning & DQN |
| 106 | Policy gradients / REINFORCE |
| 107 | Actor-critic & PPO |
| 108 | RAG |
| 109 | Agents & tool use |
| 110 | Multimodal models |
| 111 | Long-context techniques |
| 112 | Evaluation, benchmarks & pitfalls |

---

## Progress

**Current day:** 52 / 112  
**Last concept:** Fixed-context bottleneck (Concept 52)  
**Next up:** Attention intuition — soft lookup (Concept 53)

**Pending backfill:** Concept 50, Bidirectional RNNs, was intentionally skipped to prioritize the Seq2seq/bottleneck lead-in into Phase 6 — see `progress.json`'s `skipped_concepts`. It still needs its own lesson.

---

## Repository layout

```
LESSON_YYYY_MM_DD_DAY_N.md   — daily lesson files
progress.json                — bot state (day, last concept, mode)
README.md                    — this file
```
