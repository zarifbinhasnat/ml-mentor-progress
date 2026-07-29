"""
Day 62 -- Causal masking mechanics: raw scores -> masked scores -> softmax weights.

Self-contained, no downloads. Generates a tiny synthetic 8-token attention
example (random Q, K with a fixed seed) and shows the three-stage pipeline
that additive causal masking actually runs: QK^T/sqrt(d) -> (+ mask) -> softmax.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n, d = 8, 4  # 8 tokens, head dim 4 (tiny synthetic example, no real embeddings)

Q = np.random.randn(n, d)
K = np.random.randn(n, d)

raw_scores = (Q @ K.T) / np.sqrt(d)  # scaled dot-product scores, shape (n, n)

# Additive causal mask: 0 where allowed (j <= i), -inf where blocked (j > i)
blocked = np.triu(np.ones((n, n), dtype=bool), k=1)
causal_mask = np.where(blocked, -1e9, 0.0)  # finite sentinel stand-in for -inf, purely for float math below

masked_scores = raw_scores + causal_mask

# Row-wise softmax (numerically stable)
shifted = masked_scores - masked_scores.max(axis=1, keepdims=True)
exp_scores = np.exp(shifted)
attn_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

# For display only: clip the -1e9 sentinel down to a fixed dark value so the
# "blocked" region reads as a clean solid block rather than a diverging color.
masked_scores_display = np.where(causal_mask < -1e8, np.nan, masked_scores)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

titles = ["Raw scores  $QK^T/\\sqrt{d}$", "After additive mask (blank = $-\\infty$)", "After softmax (attention weights)"]
data = [raw_scores, masked_scores_display, attn_weights]
cmaps = ["coolwarm", "coolwarm", "viridis"]

for ax, mat, title, cmap in zip(axes, data, titles, cmaps):
    im = ax.imshow(mat, cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("key position $j$")
    ax.set_ylabel("query position $i$")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # annotate each cell so the "blocked -> zero probability" story is legible
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if np.isnan(val):
                ax.text(j, i, "x", ha="center", va="center", fontsize=8, color="black")
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                         color="white" if cmap == "viridis" else "black")

fig.suptitle("Causal masking pipeline: mask is additive BEFORE softmax, never a post-hoc zeroing", fontsize=12)
plt.tight_layout()
plt.savefig("graph_DAY_62.png", dpi=120, bbox_inches="tight")
