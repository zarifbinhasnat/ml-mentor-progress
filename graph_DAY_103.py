"""
Day 103 — Serving (vLLM, continuous batching): PagedAttention memory waste.

Compares KV-cache memory reservation under:
  1. Naive contiguous allocation — each request pre-reserves a full
     max-context-length slab up front (classic pre-vLLM serving).
  2. PagedAttention — each request reserves fixed-size blocks on demand,
     so waste is bounded by (block_size - 1) tokens per request instead
     of (max_len - actual_len).

Synthetic data only, no downloads.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

NUM_REQUESTS = 16
MAX_CONTEXT = 2048      # naive scheme reserves this per request, worst case
BLOCK_SIZE = 16         # vLLM-style KV-cache page size (tokens per block)

# Actual generated length (prompt + decoded tokens so far) per request.
actual_lengths = np.random.randint(50, 400, size=NUM_REQUESTS)

# --- Naive contiguous allocation ---
naive_reserved = np.full(NUM_REQUESTS, MAX_CONTEXT)
naive_used = actual_lengths
naive_wasted = naive_reserved - naive_used

# --- PagedAttention: round each request up to the next full block ---
blocks_needed = np.ceil(actual_lengths / BLOCK_SIZE).astype(int)
paged_reserved = blocks_needed * BLOCK_SIZE
paged_used = actual_lengths
paged_wasted = paged_reserved - paged_used

naive_total_reserved, naive_total_used = naive_reserved.sum(), naive_used.sum()
paged_total_reserved, paged_total_used = paged_reserved.sum(), paged_used.sum()

naive_util = 100 * naive_total_used / naive_total_reserved
paged_util = 100 * paged_total_used / paged_total_reserved

fig, ax = plt.subplots(figsize=(6, 5))

labels = ["Naive contiguous\n(reserve max_len)", f"PagedAttention\n(block={BLOCK_SIZE})"]
used_vals = [naive_total_used, paged_total_used]
wasted_vals = [naive_total_reserved - naive_total_used, paged_total_reserved - paged_total_used]

x = np.arange(2)
bars_used = ax.bar(x, used_vals, width=0.5, label="tokens actually used", color="#3b82f6")
bars_wasted = ax.bar(x, wasted_vals, width=0.5, bottom=used_vals, label="reserved but wasted", color="#f97316")

for i, (u, w, util) in enumerate(zip(used_vals, wasted_vals, [naive_util, paged_util])):
    ax.text(x[i], u + w + naive_total_reserved * 0.01, f"{util:.1f}% used",
            ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("KV-cache slots (tokens)")
ax.set_title(f"KV-cache memory: naive vs PagedAttention ({NUM_REQUESTS} concurrent requests)")
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig("graph_DAY_103.png", dpi=120, bbox_inches="tight")
print(f"Naive utilization:  {naive_util:.1f}%  ({naive_total_used}/{naive_total_reserved} tokens)")
print(f"Paged utilization:  {paged_util:.1f}%  ({paged_total_used}/{paged_total_reserved} tokens)")
