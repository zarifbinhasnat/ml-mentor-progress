"""
Day 57 — Self- vs Cross-Attention: same QKV mechanism, different
source of Q versus K/V.

Self-attention: one sequence of 9 tokens, Q/K/V all projected from
the SAME embeddings -> attention matrix is square (9x9).

Cross-attention: a short 5-token decoder sequence supplies Q, while
a longer 9-token encoder sequence supplies K/V -> attention matrix
is rectangular (5x9), since query count and key count differ.

Both use identical scaled dot-product machinery (seed=42, same d_k);
the only structural difference is where Q comes from.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

d_model = 16
d_k = 8
n_self = 9      # self-attention sequence length
n_dec = 5       # cross-attention: decoder (query) length
n_enc = 9       # cross-attention: encoder (key/value) length


def softmax_rows(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def attention_weights(q_src, kv_src, Wq, Wk):
    Q = q_src @ Wq
    K = kv_src @ Wk
    scores = (Q @ K.T) / np.sqrt(d_k)
    return softmax_rows(scores)


Wq = np.random.randn(d_model, d_k) * (1.0 / np.sqrt(d_model))
Wk = np.random.randn(d_model, d_k) * (1.0 / np.sqrt(d_model))

# self-attention: one shared sequence, Q and K/V both come from it
x_self = np.random.randn(n_self, d_model)
self_weights = attention_weights(x_self, x_self, Wq, Wk)

# cross-attention: decoder sequence (Q) is different from encoder sequence (K/V)
x_dec = np.random.randn(n_dec, d_model)
x_enc = np.random.randn(n_enc, d_model)
cross_weights = attention_weights(x_dec, x_enc, Wq, Wk)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

im0 = axes[0].imshow(self_weights, cmap="viridis", vmin=0, vmax=self_weights.max())
axes[0].set_title(f"Self-attention: {n_self}x{n_self} (square)\nQ, K, V all from one sequence")
axes[0].set_xlabel("key position (same sequence)")
axes[0].set_ylabel("query position (same sequence)")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(cross_weights, cmap="viridis", vmin=0, vmax=cross_weights.max())
axes[1].set_title(f"Cross-attention: {n_dec}x{n_enc} (rectangular)\nQ from decoder, K/V from encoder")
axes[1].set_xlabel("key position (encoder sequence)")
axes[1].set_ylabel("query position (decoder sequence)")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

fig.suptitle("Same scaled dot-product mechanism, different source for Q vs K/V", fontsize=12)
plt.tight_layout()
plt.savefig("graph_DAY_57.png", dpi=120, bbox_inches="tight")
