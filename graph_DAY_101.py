"""
Day 101 - Throughput vs. latency, batching.

Synthetic roofline-style model of one LLM decode step as a function of batch
size B:
    mem_time(B)     = base_mem_load + kv_read_slope * B   (memory-bound term:
                       weights are read once per step regardless of B, plus a
                       small per-request KV-cache read cost)
    compute_time(B) = compute_slope * B                    (compute-bound term:
                       FLOPs scale ~linearly with how many sequences you run
                       through the matmuls this step)
    step_latency(B) = max(mem_time(B), compute_time(B))    (GPU work is
                       whichever resource is the bottleneck)
    throughput(B)   = B / step_latency(B)                  (requests served
                       per unit time)

No real GPU or model is involved -- these are hand-picked constants that
reproduce the textbook memory-bound -> compute-bound crossover shape.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- synthetic roofline constants (all in milliseconds) ---
base_mem_load = 15.0     # ms: fixed cost of streaming weights from HBM once per step
kv_read_slope = 0.05      # ms/request: extra memory traffic for each sequence's KV cache
compute_slope = 0.40      # ms/request: extra matmul FLOPs for each sequence in the batch

batch_sizes = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256])

mem_time = base_mem_load + kv_read_slope * batch_sizes
compute_time = compute_slope * batch_sizes
step_latency = np.maximum(mem_time, compute_time)
throughput = batch_sizes / step_latency * 1000.0   # requests/sec (latency is in ms)

# crossover: base_mem_load + kv_read_slope*B = compute_slope*B
b_star = base_mem_load / (compute_slope - kv_read_slope)

fig, ax1 = plt.subplots(figsize=(7, 4.5))

ax1.plot(batch_sizes, step_latency, "o-", color="#1f77b4", label="Step latency (ms)")
ax1.set_xlabel("Batch size (concurrent requests per decode step)")
ax1.set_ylabel("Latency per decode step (ms)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_xscale("log", base=2)

ax2 = ax1.twinx()
ax2.plot(batch_sizes, throughput, "s-", color="#d62728", label="Throughput (req/s)")
ax2.set_ylabel("Throughput (requests/sec)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

ax1.axvline(b_star, color="gray", linestyle="--", linewidth=1)
ax1.text(b_star * 1.05, ax1.get_ylim()[1] * 0.9, f"crossover\nB*≈{b_star:.0f}",
          fontsize=8, color="gray")

ax1.text(b_star * 0.25, ax1.get_ylim()[1] * 0.55, "memory-bound\n(latency ~flat,\nthroughput scales)",
          fontsize=8, ha="center", color="#1f77b4")
ax1.text(b_star * 4.0, ax1.get_ylim()[1] * 0.3, "compute-bound\n(latency grows,\nthroughput saturates)",
          fontsize=8, ha="center", color="#d62728")

plt.title("LLM decode step: latency vs. throughput as batch size grows")
fig.tight_layout()
plt.savefig("graph_DAY_101.png", dpi=120, bbox_inches="tight")
