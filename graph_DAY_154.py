import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
peak = 989.0      # TFLOP/s, H100 SXM bf16 dense (approx)
bw = 3.35         # TB/s, HBM3 (approx)
ridge = peak / bw # FLOP/byte

ai = np.logspace(-1, 4, 400)
perf = np.minimum(peak, ai * bw)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(ai, perf, lw=2.5, color="#1f77b4", label="Roofline (H100 bf16, approx)")
ax.axvline(ridge, ls="--", color="gray")
ax.text(ridge * 1.05, 2, f"ridge ~ {ridge:.0f} FLOP/B", color="gray")

pts = {
    "elementwise add\n(AI~0.17)": 1 / 6,
    "decode, batch 1\n(AI~1)": 1.0,
    "decode, batch 64\n(AI~64)": 64.0,
    "matmul n=128\n(AI~43)": 128 / 3,
    "matmul n=4096\n(AI~1365)": 4096 / 3,
}
for name, x in pts.items():
    y = min(peak, x * bw)
    ax.plot(x, y, "o", color="#d62728")
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, -22), fontsize=7)

ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
ax.set_ylabel("Attainable performance (TFLOP/s)")
ax.set_title("Roofline model: memory-bound slope vs compute-bound ceiling")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right")
plt.savefig("graph_DAY_154.png", dpi=120, bbox_inches="tight")
