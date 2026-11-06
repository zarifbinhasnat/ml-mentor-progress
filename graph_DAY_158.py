import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
# Little's law: bandwidth achieved = min(peak, bytes_in_flight / latency)
peak = 3.0  # TB/s (approx HBM-class)
inflight_mb = np.linspace(0.01, 3.0, 300)  # MB in flight
fig, ax = plt.subplots(figsize=(6.5, 4))
for lat_ns, c in [(250, "#1f77b4"), (500, "#d62728"), (1000, "#2ca02c")]:
    bw = np.minimum(peak, inflight_mb * 1e6 / (lat_ns * 1e-9) / 1e12)
    ax.plot(inflight_mb, bw, color=c, label=f"latency = {lat_ns} ns")
ax.axhline(peak, color="gray", ls="--", lw=0.8)
ax.set_xlabel("Bytes in flight (MB)")
ax.set_ylabel("Achieved bandwidth (TB/s)")
ax.set_title("Latency hiding: need bytes_in_flight >= bandwidth x latency")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig("graph_DAY_158.png", dpi=120, bbox_inches="tight")
