"""
Day 56 — Multi-Head Attention: visualize how independent heads
learn different attention patterns over the SAME sequence.

Synthetic setup: 8 tokens, 4 heads, d_k=8 per head. Each head gets its
own random Q/K projection (seed=42 for reproducibility), so the
resulting attention-weight heatmaps differ head-to-head even though
every head is looking at the identical input sequence.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_tokens = 8
n_heads = 4
d_k = 8

def softmax_rows(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

# one shared input embedding sequence, but each head projects it
# through its own random W_q / W_k -> different Q, K -> different pattern
x = np.random.randn(n_tokens, 16)

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for h, ax in enumerate(axes.flat):
    Wq = np.random.randn(16, d_k) * (1.0 / np.sqrt(16))
    Wk = np.random.randn(16, d_k) * (1.0 / np.sqrt(16))
    Q = x @ Wq
    K = x @ Wk
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax_rows(scores)

    im = ax.imshow(weights, cmap="viridis", vmin=0, vmax=weights.max())
    ax.set_title(f"Head {h} (random, untrained W_q/W_k)", fontsize=10)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.set_xticks(range(n_tokens))
    ax.set_yticks(range(n_tokens))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Multi-Head Attention: 4 heads, same input, 4 different (untrained) attention patterns", fontsize=12)
plt.tight_layout()
plt.savefig("graph_DAY_56.png", dpi=120, bbox_inches="tight")
