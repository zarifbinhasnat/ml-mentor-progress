import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Sequence lengths from 256 to 65536 tokens (20 log-spaced points)
n = np.logspace(np.log2(256), np.log2(65536), num=20, base=2)
d = 64          # head dim
M = 25_000      # SRAM capacity in elements (~100KB fp32 on-chip, illustrative)

# Standard attention: read/write Q,K,V (O(n*d)) plus write+re-read the full
# n x n score matrix to/from slow HBM (O(n^2)) for softmax and the second matmul.
standard_hbm = n * d + n ** 2

# FlashAttention: tiles Q,K,V into SRAM-sized blocks and fuses matmul+softmax+matmul
# into one kernel, so the n x n matrix never touches HBM at all -- only O(n^2 d^2 / M)
# block-reload traffic remains (paper's HBM-access counting argument, illustrative constants).
flash_hbm = (n ** 2 * d ** 2) / M

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(n, standard_hbm, marker="o", label="Standard attention HBM traffic — O(nd + n²)", color="#d62728")
ax.plot(n, flash_hbm, marker="s", label="FlashAttention HBM traffic — O(n²d²/M)", color="#2ca02c")

for mark_n in [1024, 4096, 16384, 65536]:
    idx = int(np.abs(n - mark_n).argmin())
    speedup = standard_hbm[idx] / flash_hbm[idx]
    ax.annotate(
        f"n={int(round(n[idx]))}\n{speedup:.1f}x fewer\nHBM reads/writes",
        (n[idx], flash_hbm[idx]),
        textcoords="offset points",
        xytext=(0, -35),
        fontsize=7.5,
        ha="center",
    )

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Sequence length n (tokens)")
ax.set_ylabel("Elements moved to/from HBM (illustrative units)")
ax.set_title("Same O(n²) FLOPs, far less HBM traffic: why FlashAttention is memory-bound-fast")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="both", alpha=0.3)

plt.savefig("graph_DAY_64.png", dpi=120, bbox_inches="tight")
