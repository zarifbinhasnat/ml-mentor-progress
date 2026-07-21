"""
Day 54 -- Query, Key, Value: visualize the attention weight matrix that
falls out of projecting a tiny synthetic sequence of token embeddings
through three learned (here: randomly initialized, fixed) weight matrices
W_Q, W_K, W_V, then computing softmax(QK^T / sqrt(d_k)).

Self-contained: only numpy + matplotlib, no downloads, no torch needed.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

tokens = ["the", "cat", "sat", "on", "mat"]
n, d_model, d_k = len(tokens), 8, 4

# Synthetic token embeddings: X[i] is token i's d_model-dim vector.
X = np.random.randn(n, d_model)

# Learned projections (random-but-fixed here, since we only care about the
# mechanics of Q, K, V -- not about training).
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_k) * 0.5

Q = X @ W_Q  # (n, d_k) -- each row is token i's "what am I looking for" vector
K = X @ W_K  # (n, d_k) -- each row is token i's "what do I offer as a match" vector
V = X @ W_V  # (n, d_k) -- each row is token i's "what do I actually hand back" vector

scores = (Q @ K.T) / np.sqrt(d_k)  # (n, n) raw similarity, scaled (previews Day 55)


def softmax_rows(m):
    e = np.exp(m - m.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


weights = softmax_rows(scores)  # (n, n) -- row i sums to 1: how much token i attends to every token
output = weights @ V  # (n, d_k) -- each row is a blended value vector

fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(weights, cmap="viridis", vmin=0, vmax=weights.max())

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(tokens)
ax.set_yticklabels(tokens)
ax.set_xlabel("attending TO (Key)")
ax.set_ylabel("attending FROM (Query)")
ax.set_title("Self-attention weights: softmax(QK$^T$/$\\sqrt{d_k}$)V")

for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                color="white" if weights[i, j] < weights.max() * 0.6 else "black",
                fontsize=8)

fig.colorbar(im, ax=ax, label="attention weight")
plt.savefig("graph_DAY_54.png", dpi=120, bbox_inches="tight")
