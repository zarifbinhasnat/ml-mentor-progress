import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# Rough, order-of-magnitude figures for a typical modern CPU system (≈ values)
levels = ["Register", "L1 cache", "L2 cache", "L3 cache", "DRAM", "NVMe SSD"]
latency_ns = np.array([0.3, 1, 4, 15, 100, 20_000])
capacity_bytes = np.array([1e3, 48e3, 2e6, 32e6, 64e9, 2e12])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
y = np.arange(len(levels))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(levels)))

ax1.barh(y, latency_ns, color=colors)
ax1.set_xscale("log")
ax1.set_yticks(y, levels)
ax1.invert_yaxis()
ax1.set_xlabel("Access latency (ns, log scale)")
ax1.set_title("Further from the core = slower")
for i, v in enumerate(latency_ns):
    label = f"~{v:g} ns" if v < 1000 else f"~{v/1000:g} µs"
    ax1.text(v * 1.3, i, label, va="center", fontsize=9)
ax1.set_xlim(0.1, 1e6)

ax2.barh(y, capacity_bytes, color=colors)
ax2.set_xscale("log")
ax2.set_yticks(y, levels)
ax2.invert_yaxis()
ax2.set_xlabel("Typical capacity (bytes, log scale)")
ax2.set_title("...but much bigger")
names = ["~1 KB", "~48 KB", "~2 MB", "~32 MB", "~64 GB", "~2 TB"]
for i, (v, n) in enumerate(zip(capacity_bytes, names)):
    ax2.text(v * 1.5, i, n, va="center", fontsize=9)
ax2.set_xlim(1e2, 1e15)

fig.suptitle("Memory hierarchy (order-of-magnitude, typical CPU system)", fontsize=12)
plt.tight_layout()
plt.savefig("graph_DAY_153.png", dpi=120, bbox_inches="tight")
