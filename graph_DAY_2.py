import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 400)

sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
d_sigmoid = sigmoid * (1 - sigmoid)
d_tanh = 1 - tanh ** 2

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Day 2 — Sigmoid & Tanh: Functions and Derivatives", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(z, sigmoid, color="#e06c75", linewidth=2, label=r"$\sigma(z)=\frac{1}{1+e^{-z}} \in (0,1)$")
ax.plot(z, tanh, color="#61afef", linewidth=2, label=r"$\tanh(z) \in (-1,1)$")
ax.axhline(0, color="k", linewidth=0.5)
ax.axvline(0, color="k", linewidth=0.5)
ax.set_xlabel("z"); ax.set_ylabel("Activation output")
ax.set_title("Activation functions")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.annotate("Saturates → gradient ≈ 0", xy=(4.5, sigmoid[-100]), fontsize=8, color="darkred",
            xytext=(2.5, 0.2), arrowprops=dict(arrowstyle="->", color="darkred"))

ax = axes[1]
ax.plot(z, d_sigmoid, color="#e06c75", linewidth=2, label=r"$\sigma'(z) = \sigma(1-\sigma)$, max = 0.25")
ax.plot(z, d_tanh,   color="#61afef", linewidth=2, label=r"$\tanh'(z) = 1-\tanh^2(z)$, max = 1.0")
ax.axhline(0.25, color="#e06c75", linewidth=0.8, linestyle=":")
ax.axhline(1.0,  color="#61afef", linewidth=0.8, linestyle=":")
ax.text(5.5, 0.27, "0.25", color="#e06c75", fontsize=8)
ax.text(5.5, 1.02, "1.00", color="#61afef", fontsize=8)
ax.set_xlabel("z"); ax.set_ylabel("Derivative")
ax.set_title("Derivatives — the vanishing gradient story")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_2.png", dpi=120, bbox_inches="tight")
print("graph_DAY_2.png saved")
