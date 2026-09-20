"""
Day 111 - Long-context techniques
Synthetic graph: how per-token memory cost grows with sequence length
under three attention strategies:
  1. Full causal attention (KV cache grows linearly in tokens, but the
     attention COMPUTE at each new step is O(n) -> O(n^2) total)
  2. Sliding-window attention (fixed window W -> O(1) memory once n > W)
  3. Sparse/strided attention with a fixed number of global tokens
     (O(1) memory plus a small constant global-token budget)

No downloads, tiny synthetic arrays, seeded RNG only for cosmetic jitter.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Sequence lengths we simulate (tokens generated so far)
n = np.arange(1, 65)  # 64 decoding steps, small + hardcoded, no downloads

# --- Strategy 1: full causal attention KV cache size (in "slots") ---
full_cache = n.astype(float)  # grows 1:1 with tokens -> unbounded

# --- Strategy 2: sliding window attention, window W ---
W = 16
sliding_cache = np.minimum(n, W).astype(float)  # caps at W once n > W

# --- Strategy 3: sparse/strided attention with G fixed "global" tokens
#     plus a small local window Wl ---
G = 4
Wl = 8
sparse_cache = np.minimum(n, G + Wl).astype(float)  # caps at G+Wl

# Tiny cosmetic jitter so overlapping flat lines are still readable
jitter = np.random.normal(0, 0.15, size=n.shape)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(n, full_cache, label="Full causal attention (KV cache)", color="#d62728", linewidth=2)
ax.plot(n, sliding_cache + jitter, label=f"Sliding window (W={W})", color="#1f77b4", linewidth=2)
ax.plot(n, sparse_cache + jitter, label=f"Sparse: {G} global + {Wl} local", color="#2ca02c", linewidth=2)
ax.axhline(W, color="#1f77b4", linestyle=":", linewidth=1, alpha=0.6)
ax.axhline(G + Wl, color="#2ca02c", linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel("tokens generated so far (n)")
ax.set_ylabel("KV cache size (slots retained)")
ax.set_title("Cache growth: full vs. sliding-window vs. sparse")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

# --- Right panel: cumulative attention FLOPs (relative units) ---
# Full: sum_{i=1}^{n} i  ~ O(n^2)   (each new query attends to all past keys)
full_flops = np.cumsum(n)
# Sliding: each step attends to at most W keys -> O(n*W), linear in n
sliding_flops = np.cumsum(np.minimum(n, W))
# Sparse: each step attends to at most G+Wl keys -> O(n*(G+Wl)), linear in n
sparse_flops = np.cumsum(np.minimum(n, G + Wl))

ax2 = axes[1]
ax2.plot(n, full_flops, label="Full: O(n²) total", color="#d62728", linewidth=2)
ax2.plot(n, sliding_flops, label="Sliding: O(n·W)", color="#1f77b4", linewidth=2)
ax2.plot(n, sparse_flops, label="Sparse: O(n·(G+Wl))", color="#2ca02c", linewidth=2)
ax2.set_xlabel("tokens generated so far (n)")
ax2.set_ylabel("cumulative attention work (relative units)")
ax2.set_title("Cumulative compute: quadratic vs. linear")
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(alpha=0.3)

fig.suptitle("Long-context techniques: capping memory & compute per decode step", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("graph_DAY_111.png", dpi=120, bbox_inches="tight")
print("saved graph_DAY_111.png")
