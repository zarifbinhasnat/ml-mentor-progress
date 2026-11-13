import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
N = 2**26            # elements
bytes_per = 2        # bf16
BW = 3.0e12          # ~3 TB/s HBM (approximate)
k = np.arange(1, 13) # number of chained elementwise ops
tensor = N * bytes_per
unfused = k * 2 * tensor / BW * 1e6   # each op: read + write
fused = np.full_like(unfused, 2 * tensor / BW * 1e6)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(k, unfused, "o-", color="#c0503a", label="unfused: k kernels, k round trips to HBM")
ax.plot(k, fused, "s-", color="#3b6fb6", label="fused: 1 kernel, 1 round trip")
ax.set_xlabel("number of chained elementwise ops k")
ax.set_ylabel("time lower bound (microseconds)")
ax.set_title("Memory-bound chain: time = bytes / bandwidth (2^26 bf16 elems)")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig("graph_DAY_165.png", dpi=120, bbox_inches="tight")
