"""
Day 100 -- KV cache
KV cache memory footprint (GB) vs. sequence length, for a ~7B-scale decoder-only
transformer (L=32 layers, head_dim=128, fp16), comparing three attention
head-sharing schemes that only differ in how many KV heads they keep:
  MHA (Multi-Head Attention):  n_kv_heads = 32  (one KV head per query head)
  GQA (Grouped-Query Attn):    n_kv_heads = 8   (8 query heads share each KV head)
  MQA (Multi-Query Attn):      n_kv_heads = 1   (all 32 query heads share one KV head)

Formula (batch=1):
  bytes = 2 (K and V) * L * n_kv_heads * head_dim * seq_len * bytes_per_elem
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

L = 32          # transformer layers
head_dim = 128  # per-head dimension
bytes_per_elem = 2  # fp16

seq_len = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384])

def kv_cache_gb(n_kv_heads):
    bytes_total = 2 * L * n_kv_heads * head_dim * seq_len * bytes_per_elem
    return bytes_total / 1e9

mha = kv_cache_gb(32)
gqa = kv_cache_gb(8)
mqa = kv_cache_gb(1)

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(seq_len, mha, marker="o", label="MHA (32 KV heads)", color="#C44E52")
ax.plot(seq_len, gqa, marker="o", label="GQA (8 KV heads)", color="#DD8452")
ax.plot(seq_len, mqa, marker="o", label="MQA (1 KV head)", color="#55A868")

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks(seq_len)
ax.set_xticklabels(seq_len)
ax.set_xlabel("Sequence length (tokens)")
ax.set_ylabel("KV cache size (GB), batch=1, fp16")
ax.set_title("KV cache memory vs. sequence length (7B-scale, L=32, head_dim=128)")
ax.legend(frameon=False)
ax.grid(True, which="both", alpha=0.3)

plt.savefig("graph_DAY_100.png", dpi=120, bbox_inches="tight")
