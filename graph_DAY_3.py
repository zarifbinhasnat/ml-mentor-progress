import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

z = np.linspace(-4, 4, 400)

relu     = np.maximum(0, z)
leaky    = np.where(z >= 0, z, 0.1 * z)
gelu     = 0.5 * z * (1 + erf(z / np.sqrt(2)))
silu     = z / (1 + np.exp(-z))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Day 3 — ReLU, LeakyReLU, GELU, SiLU", fontsize=13, fontweight="bold")

colors = ["#e06c75", "#61afef", "#98c379", "#e5c07b"]
labels = ["ReLU = max(0, z)", "LeakyReLU (α=0.1)", "GELU ≈ z·Φ(z)", "SiLU = z·σ(z)"]
funcs  = [relu, leaky, gelu, silu]

ax = axes[0]
for f, c, l in zip(funcs, colors, labels):
    ax.plot(z, f, color=c, linewidth=2, label=l)
ax.axhline(0, color="k", linewidth=0.5)
ax.axvline(0, color="k", linewidth=0.5)
ax.set_xlabel("z"); ax.set_ylabel("Activation")
ax.set_title("Activation functions (output)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 4)

# Derivatives
d_relu  = np.where(z > 0, 1.0, 0.0)
d_leaky = np.where(z >= 0, 1.0, 0.1)
d_gelu  = 0.5 * (1 + erf(z / np.sqrt(2))) + z * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
sig     = 1 / (1 + np.exp(-z))
d_silu  = sig + z * sig * (1 - sig)

ax = axes[1]
for df, c, l in zip([d_relu, d_leaky, d_gelu, d_silu], colors, labels):
    ax.plot(z, df, color=c, linewidth=2, label=l)
ax.axhline(0, color="k", linewidth=0.5)
ax.axhline(1, color="k", linewidth=0.5, linestyle=":")
ax.set_xlabel("z"); ax.set_ylabel("Derivative")
ax.set_title("Derivatives — gradient flow")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(-0.3, 1.6)
ax.text(3.5, 1.02, "1.0", fontsize=8, color="grey")

plt.tight_layout()
plt.savefig("graph_DAY_3.png", dpi=120, bbox_inches="tight")
print("graph_DAY_3.png saved")
