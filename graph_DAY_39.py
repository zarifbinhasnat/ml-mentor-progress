"""
Day 39 -- LeNet & AlexNet
Left:  scale explosion from LeNet-5 (1998) to AlexNet (2012) across four axes --
       parameter count, depth, training-set size, input resolution -- all log scale.
Right: tanh vs ReLU derivative -- why swapping tanh for ReLU was one of AlexNet's
       biggest speed unlocks (saturating gradients vs a constant-1 gradient).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # deterministic synthetic data

# ---------------------------------------------------------------------------
# Left panel: LeNet-5 vs AlexNet, four scale axes (log)
# ---------------------------------------------------------------------------
metrics = ["params", "depth\n(layers)", "train images", "input pixels"]
lenet = np.array([60_000, 7, 60_000, 32 * 32])
alexnet = np.array([60_000_000, 8, 1_200_000, 227 * 227])

x = np.arange(len(metrics))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.bar(x - width / 2, lenet, width, label="LeNet-5 (1998)", color="#1f77b4")
ax.bar(x + width / 2, alexnet, width, label="AlexNet (2012)", color="#d62728")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=8.5)
ax.set_ylabel("count (log scale)")
ax.set_title("14 years of scale: LeNet-5 vs AlexNet")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")
for xi, (l, a) in enumerate(zip(lenet, alexnet)):
    ax.annotate(f"{a / l:.0f}x", (xi, a), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=7.5, color="#d62728")

# ---------------------------------------------------------------------------
# Right panel: tanh vs ReLU, function + derivative
# ---------------------------------------------------------------------------
z = np.linspace(-4, 4, 400)
tanh = np.tanh(z)
tanh_grad = 1 - tanh ** 2
relu = np.maximum(0, z)
relu_grad = (z > 0).astype(float)

ax = axes[1]
ax.plot(z, tanh_grad, color="#1f77b4", label="d/dz tanh(z)  (saturates -> 0)")
ax.plot(z, relu_grad, color="#d62728", label="d/dz ReLU(z)  (constant 1 for z>0)")
ax.set_xlabel("pre-activation z")
ax.set_ylabel("gradient magnitude")
ax.set_title("Why ReLU trained AlexNet ~6x faster than tanh")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_39.png", dpi=120, bbox_inches="tight")
