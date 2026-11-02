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

## v2 — হার্ডওয়্যার, ফ্রন্টিয়ার এআই ও CSE 4711

Day 152 থেকে রুটিন v2-তে গেছে। এখন থেকে সব নতুন লেখা বাংলায়; technical term ইংরেজিতেই থাকে। পুরোনো lesson, graph, README অংশ এবং `progress.json`-এর ক্ষেত্র অপরিবর্তিত।

### নতুন lesson format
প্রতিদিনের lesson-এ থাকে: ⚙️ গভীর পাঠ (Deep Dive, back-of-envelope হিসাবসহ) · 🎓 কোর্স কর্নার (CSE 4711) · 🏋️ গন্টলেট (C++) · 📄 পেপার কর্নার (শুক্রবার) · 🔭 রিসার্চ রাডার (বুধবার) · 🏗️ ব্লুপ্রিন্ট (সপ্তাহে ≈ ২ বার) · 🗺️ পরের পথ। **"Pythonic Edge" ও "Signal Lab" Day 152 থেকে বন্ধ**; আগের lesson-এ সেগুলো যেমন ছিল তেমনই আছে।

### Track B — Phase 12–17

**Phase 12 — হার্ডওয়্যারের ভিত্তি**

| # | Concept |
|---|---|
| 115 | Compute বনাম memory: FLOPs, bytes, latency বনাম bandwidth |
| 116 | Memory hierarchy: registers → cache → DRAM → HBM, locality |
| 117 | Arithmetic intensity ও roofline model |
| 118 | CPU parallelism: SIMD, multicore, cache-tiled matmul |
| 119 | Number formats I: IEEE float, fp32/fp16/bf16 |
| 120 | Number formats II: int8, fp8, fp4, block/micro-scaling |
| 121 | GPU বনাম CPU: throughput machine ও latency hiding |
| 122 | GPU architecture: SM, warp, SIMT, divergence |
| 123 | GPU memory: HBM, shared memory, register, coalescing, bank conflict |
| 124 | Tensor core ও shape 8/16-এর গুণিতক কেন |

**Phase 13 — Kernels**

| # | Concept |
|---|---|
| 125 | CUDA programming model: grid/block/thread |
| 126 | Naive matmul → shared-memory tiled matmul |
| 127 | Occupancy ও profiling mindset |
| 128 | Kernel fusion |
| 129 | Reduction ও online softmax |
| 130 | Triton |
| 131 | FlashAttention revisited |
| 132 | FlashAttention-2/3 |
| 133 | torch.compile |
| 134 | CUDA graphs ও launch overhead |

**Phase 14 — Inference engineering**

| # | Concept |
|---|---|
| 135 | Prefill বনাম decode |
| 136 | KV cache math |
| 137 | MQA ও GQA |
| 138 | MLA (DeepSeek-V2) |
| 139 | PagedAttention ও vLLM |
| 140 | Continuous batching |
| 141 | Prefix caching / RadixAttention |
| 142 | Speculative decoding, EAGLE, MTP |
| 143 | Weight-only quantization: GPTQ, AWQ |
| 144 | Activation outlier, SmoothQuant, KV-cache quantization |
| 145 | Prefill/decode disaggregation |
| 146 | Long-context: sparse attention ও NSA |

**Phase 15 — Distributed training**

| # | Concept |
|---|---|
| 147 | Collectives ও ring algorithm |
| 148 | Interconnects: PCIe, NVLink, InfiniBand |
| 149 | DDP, ZeRO, FSDP ও communication cost |
| 150 | Tensor parallelism |
| 151 | Pipeline parallelism |
| 152 | DualPipe |
| 153 | Expert parallelism, DeepEP |
| 154 | DeepSeekMoE |
| 155 | Auxiliary-loss-free load balancing |
| 156 | FP8 training ও DeepGEMM |
| 157 | Activation recompute ও memory-budget |
| 158 | Fault tolerance, checkpointing, 3FS |

**Phase 16 — Reasoning ও RL**

| # | Concept |
|---|---|
| 159 | Q-learning থেকে policy gradient |
| 160 | PPO for LLMs |
| 161 | GRPO |
| 162 | DeepSeek-R1 ও R1-Zero |
| 163 | Verifiable reward ও reward model |
| 164 | Test-time compute: beam, best-of-N, MCTS |
| 165 | Reasoning distillation |
| 166 | RL infrastructure |

**Phase 17 — Architecture frontier**

| # | Concept |
|---|---|
| 167 | Muon |
| 168 | SSM ও Mamba |
| 169 | Linear attention ও hybrid |
| 170 | Capstone: frontier model-এর training + serving খরচ |

১৭০-এর পর "Frontier Review Day": paper বা Research Radar theme ধরে শুরু থেকে শেষ পর্যন্ত আলোচনা।

### 🎓 কোর্স কর্নার (CSE 4711, IUT Winter 2026–27)

| সময় | অবস্থা |
|---|---|
| 26 Oct – 11 Dec 2026 | Class, Week 1–7 |
| 14 – 25 Dec 2026 | Mid-semester exam (Week 1–7), তার আগে drill |
| 28 Dec 2026 – 12 Feb 2027 | Class, Week 8–14 |
| 15 – 26 Feb 2027 | Final exam (Week 8–14), drill |
| 22 Mar 2027 | Summer semester শুরু |

সাপ্তাহিক বিষয়: W1 Intro/agent · W2 Uninformed search · W3 Informed search · W4–5 CSP · W6 Adversarial search · W7 Uncertainty/utility · W8–9 MDP · W10–11 RL · W12 Probability · W13–14 Bayes net। প্রতিদিন একটা হাতে-কলমে trace ও একটা formulate-and-justify প্রশ্ন থাকে।

### 📄 পেপার কর্নার (শুক্রবার)
ক্রম: Roofline → FlashAttention → FlashAttention-2 → FlashAttention-3 → Megatron-LM → ZeRO → PagedAttention → LLM.int8() → GPTQ → AWQ → SmoothQuant → Speculative Decoding → EAGLE → SGLang → DeepSeekMoE → DeepSeek-V2 → DeepSeek-V3 → DeepSeekMath → DeepSeek-R1 → Native Sparse Attention → Mamba।

### 🔭 রিসার্চ রাডার (বুধবার)
১০টা theme পালা করে: reasoning ও test-time compute · agents · long context ও memory · efficient architecture · low-precision · inference systems · multimodal ও world model · diffusion LM · interpretability ও alignment · data ও small model। প্রতিবার সাম্প্রতিক ২–৩টা কাজ, শুধু যা সত্যিই খুলে দেখা হয়েছে সেগুলোর link সহ।

---

## Progress

**Current day:** 52 / 112  
**Last concept:** Fixed-context bottleneck (Concept 52)  
**Next up:** Attention intuition — soft lookup (Concept 53)

**Pending backfill:** Concept 50, Bidirectional RNNs, was intentionally skipped to prioritize the Seq2seq/bottleneck lead-in into Phase 6 — see `progress.json`'s `skipped_concepts`. It still needs its own lesson.

---

## Special sessions

Outside the regular Day N / Concept N cadence, bonus capstones occasionally get added when they're tied to an actual external syllabus (a course lecture, an exam) rather than the next roadmap concept. These don't advance `progress.json`.

| File | Built from |
|---|---|
| `LESSON_2026_07_19_SPECIAL_ATTENTION_TRANSFORMERS.md` | CSE 4621 Lecture 9 ("Language Modeling") + "Maths Behind Transformers" slide decks — a problem-first, numerically worked exam-prep capstone spanning language modeling, word embeddings, RNN/LSTM (cross-referencing Days 46–49), and the full attention/Transformer Q/K/V formalism. |

---

## Repository layout

```
LESSON_YYYY_MM_DD_DAY_N.md              — daily lesson files
LESSON_YYYY_MM_DD_SPECIAL_*.md          — bonus/off-roadmap capstones (see "Special sessions")
progress.json                           — bot state (day, last concept, mode)
README.md                               — this file
```
