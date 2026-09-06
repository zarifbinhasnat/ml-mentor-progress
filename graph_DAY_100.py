"""
Day 100 -- KV cache
Two panels:
  Left:  cumulative non-attention compute (QKV/FFN projections) with vs
         without a KV cache, as a function of tokens generated T.
         No cache reprocesses the whole growing prefix every step -> O(T^2).
         With cache, only the one new token is projected every step -> O(T).
  Right: KV cache memory (GiB) vs. context length T for a concrete
         7B-parameter-class model (L=32 layers, d_model=4096, fp16),
         using memory = 2 * L * T * d_model * bytes_per_element.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- panel 1: cumulative compute, no-cache vs cached ----
T_steps = np.arange(1, 51)
no_cache_cumulative = np.cumsum(T_steps)          # sum_{t=1}^{T} t  -> O(T^2)
cached_cumulative = np.cumsum(np.ones_like(T_steps))  # sum_{t=1}^{T} 1 -> O(T)

# ---- panel 2: KV cache memory vs context length, concrete model ----
L = 32              # transformer layers (Llama-2-7B-class)
d_model = 4096       # hidden size
bytes_per_elem = 2   # fp16 / bf16
T_context = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384])
kv_bytes = 2 * L * T_context * d_model * bytes_per_elem  # 2 for K and V
kv_gib = kv_bytes / (1024 ** 3)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(T_steps, no_cache_cumulative, marker="o", markersize=3, label="No KV cache (reprocess full prefix)", color="#DD8452")
ax.plot(T_steps, cached_cumulative, marker="o", markersize=3, label="With KV cache (only new token)", color="#55A868")
ax.set_xlabel("Tokens generated so far, $T$")
ax.set_ylabel("Cumulative projection/FFN work (arbitrary units)")
ax.set_title("Non-attention compute: cached vs. uncached decoding")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(T_context, kv_gib, marker="o", color="#4C72B0")
ax2.set_xscale("log", base=2)
ax2.set_yscale("log", base=2)
ax2.set_xticks(T_context)
ax2.set_xticklabels(T_context, rotation=45)
ax2.set_xlabel("Context length $T$ (tokens)")
ax2.set_ylabel("KV cache size per sequence (GiB)")
ax2.set_title(r"KV cache memory: $L{=}32$, $d_{model}{=}4096$, fp16")
ax2.grid(True, which="both", alpha=0.3)

fig.suptitle("Day 100 -- KV cache: compute savings and the memory bill", y=1.02)
plt.savefig("graph_DAY_100.png", dpi=120, bbox_inches="tight")
