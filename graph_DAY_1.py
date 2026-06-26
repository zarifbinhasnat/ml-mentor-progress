import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# Visualise a single neuron: inputs, weights, weighted sum, and sigmoid output
n = 5
x = np.array([0.8, -0.5, 1.2, 0.3, -0.9])
w = np.array([0.6, -0.4, 0.9, 0.2, -0.7])
b = 0.1
z = np.dot(w, x) + b
a = 1 / (1 + np.exp(-z))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Day 1 — Artificial Neuron & Forward Pass", fontsize=13, fontweight="bold")

# --- panel 1: inputs and weights ---
ax = axes[0]
idx = np.arange(n)
bars = ax.bar(idx, w * x, color=["#e06c75" if v < 0 else "#98c379" for v in w * x], edgecolor="k", linewidth=0.7)
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xticks(idx)
ax.set_xticklabels([f"$w_{i+1}x_{i+1}$" for i in range(n)], fontsize=9)
ax.set_ylabel("Weighted input $w_i x_i$")
ax.set_title("Step 1: weighted inputs")
for bar, val in zip(bars, w * x):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01 * np.sign(val),
            f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

# --- panel 2: summation ---
ax = axes[1]
contributions = list(w * x) + [b]
labels = [f"$w_{i+1}x_{i+1}$" for i in range(n)] + ["bias"]
cumsum = 0.0
colors = ["#98c379" if v >= 0 else "#e06c75" for v in contributions]
for i, (val, lbl, col) in enumerate(zip(contributions, labels, colors)):
    ax.barh(i, val, color=col, edgecolor="k", linewidth=0.6, left=cumsum)
    cumsum += val
ax.axvline(0, color="k", linewidth=0.8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Value")
ax.set_title(f"Step 2: summation → $z = {z:.3f}$")
ax.axvline(z, color="navy", linestyle="--", linewidth=1.5, label=f"z = {z:.3f}")
ax.legend(fontsize=8)

# --- panel 3: sigmoid squashing ---
ax = axes[2]
z_range = np.linspace(-5, 5, 300)
sig = 1 / (1 + np.exp(-z_range))
ax.plot(z_range, sig, color="#61afef", linewidth=2, label=r"$\sigma(z)=\frac{1}{1+e^{-z}}$")
ax.axvline(z, color="navy", linestyle="--", linewidth=1.2, label=f"z = {z:.3f}")
ax.plot(z, a, "ro", markersize=8, label=f"$a = \\sigma(z) = {a:.3f}$")
ax.set_xlabel("z")
ax.set_ylabel("σ(z)")
ax.set_title("Step 3: activation σ(z)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_1.png", dpi=120, bbox_inches="tight")
print("graph_DAY_1.png saved")
