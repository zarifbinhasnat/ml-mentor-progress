"""
Day 100 — KV cache: compute savings during autoregressive decoding,
and the memory cost that savings is traded for.

Synthetic, no downloads. Run with: python graph_DAY_100.py
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Panel 1: cumulative decode compute, naive re-encode vs KV cache ----
n_tokens = np.arange(1, 51)  # generate up to 50 tokens

# Naive: at step t, re-run the FULL model over all t tokens seen so far.
# Per-step cost ~ t (one full forward pass per token in the sequence).
naive_step_cost = n_tokens.astype(float)
naive_cumulative = np.cumsum(naive_step_cost)

# KV cache: at step t, run the model on only the 1 new token, then attend
# over t cached keys/values. The forward pass is O(1); attention is O(t)
# but with a much smaller constant (attn_coef << full-layer cost).
attn_coef = 0.05
cached_step_cost = 1.0 + attn_coef * n_tokens
cached_cumulative = np.cumsum(cached_step_cost)

# ---- Panel 2: KV cache memory footprint vs sequence length ----
seq_len = np.arange(0, 4097, 64)
bytes_per_fp16 = 2

# (num_layers, num_heads, head_dim) — illustrative, not real checkpoints
configs = {
    "small (12L, 12H, 64d)": (12, 12, 64),
    "medium (24L, 16H, 64d)": (24, 16, 64),
    "large (40L, 32H, 128d)": (40, 32, 128),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(n_tokens, naive_cumulative, color="#d62728", lw=2, label="naive: re-encode full sequence every step")
ax.plot(n_tokens, cached_cumulative, color="#1f77b4", lw=2, label="KV cache: reuse past K/V, encode 1 new token")
ax.set_xlabel("tokens generated (n)")
ax.set_ylabel("cumulative compute (arb. units)")
ax.set_title("Decode compute: O(n²) vs O(n)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

ax2 = axes[1]
colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
for (label, (L, H, d)), c in zip(configs.items(), colors):
    # 2 (K and V) * layers * heads * head_dim * seq_len * bytes, batch=1
    mem_bytes = 2 * L * H * d * seq_len * bytes_per_fp16
    mem_mb = mem_bytes / (1024 ** 2)
    ax2.plot(seq_len, mem_mb, lw=2, label=label, color=c)
ax2.set_xlabel("sequence length (tokens)")
ax2.set_ylabel("KV cache size (MB, batch=1, fp16)")
ax2.set_title("KV cache memory grows linearly in seq_len")
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("graph_DAY_100.png", dpi=120, bbox_inches="tight")
print("saved graph_DAY_100.png")
