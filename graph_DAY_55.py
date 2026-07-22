"""
Day 55 — Scaled dot-product attention: why divide by sqrt(d_k)?

Two panels:
  (left)  empirical variance of raw QK^T dot products vs. dimension d,
          plotted against the theoretical line Var = d (since each of the
          d independent unit-variance components contributes ~1 to the sum).
  (right) softmax saturation: the max attention weight in a row of scores,
          comparing UNSCALED scores (QK^T) vs SCALED scores (QK^T / sqrt(d)),
          as d grows. Unscaled scores blow up in variance -> softmax
          saturates near one-hot -> vanishing gradients into the other logits.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dims = [4, 8, 16, 32, 64, 128, 256, 512]
n_trials = 500
seq_len = 8  # small "sequence" of keys competing for one query's attention

raw_variances = []
unscaled_max_probs = []
scaled_max_probs = []

for d in dims:
    # --- panel 1: variance of a single q . k dot product ---
    q = np.random.randn(n_trials, d)
    k = np.random.randn(n_trials, d)
    dots = np.sum(q * k, axis=1)  # (n_trials,) raw dot products
    raw_variances.append(np.var(dots))

    # --- panel 2: softmax saturation over a row of seq_len keys ---
    q_row = np.random.randn(n_trials, d)
    k_rows = np.random.randn(n_trials, seq_len, d)
    scores = np.einsum("nd,nsd->ns", q_row, k_rows)  # (n_trials, seq_len) raw QK^T

    def softmax(x):
        x = x - x.max(axis=1, keepdims=True)  # numerical stability, standard trick
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    unscaled_probs = softmax(scores)
    scaled_probs = softmax(scores / np.sqrt(d))

    unscaled_max_probs.append(unscaled_probs.max(axis=1).mean())
    scaled_max_probs.append(scaled_probs.max(axis=1).mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(dims, raw_variances, "o-", color="#d62728", label="empirical Var(q . k)")
ax.plot(dims, dims, "--", color="gray", label="theoretical: Var = d")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("dimension d")
ax.set_ylabel("variance of raw dot product")
ax.set_title("Dot product variance grows linearly with d")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(dims, unscaled_max_probs, "o-", color="#d62728", label="unscaled: softmax(QK^T)")
ax.plot(dims, scaled_max_probs, "o-", color="#2ca02c", label="scaled: softmax(QK^T / sqrt(d))")
ax.axhline(1.0 / seq_len, ls=":", color="gray", label=f"uniform = 1/{seq_len}")
ax.set_xscale("log", base=2)
ax.set_xlabel("dimension d")
ax.set_ylabel("mean max attention weight")
ax.set_title("Unscaled scores saturate softmax as d grows")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("Why scaled dot-product attention divides by sqrt(d_k)", y=1.03)
plt.savefig("graph_DAY_55.png", dpi=120, bbox_inches="tight")
