import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Sequence lengths from 256 to 131072 tokens (24 log-spaced points)
n = np.logspace(np.log2(256), np.log2(131072), num=24, base=2)
d_head = 64
bytes_per_elem = 2  # fp16

# Memory to materialize the full attention score matrix: n x n, one head, one layer
attn_matrix_mem_gb = (n ** 2 * bytes_per_elem) / 1e9

# Memory for the KV cache: 2 (K and V) * n * d_head, one head, one layer -- grows linearly
kv_cache_mem_gb = (2 * n * d_head * bytes_per_elem) / 1e9

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(n, attn_matrix_mem_gb, marker="o", label="Attention score matrix — O(n²), 1 head/1 layer", color="#d62728")
ax.plot(n, kv_cache_mem_gb, marker="s", label="KV cache — O(n), 1 head/1 layer", color="#1f77b4")

for mark_n in [2048, 8192, 32768, 131072]:
    idx = int(np.abs(n - mark_n).argmin())
    ax.annotate(
        f"n={int(round(n[idx]))}\n{attn_matrix_mem_gb[idx]:.2f} GB",
        (n[idx], attn_matrix_mem_gb[idx]),
        textcoords="offset points",
        xytext=(0, 10),
        fontsize=7.5,
        ha="center",
    )

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Sequence length n (tokens)")
ax.set_ylabel("Memory (GB, fp16)")
ax.set_title("O(n²) attention matrix vs. O(n) KV cache: memory blowup with context length")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="both", alpha=0.3)

plt.savefig("graph_DAY_63.png", dpi=120, bbox_inches="tight")
