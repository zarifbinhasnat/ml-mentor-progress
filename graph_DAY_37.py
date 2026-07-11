"""
Day 37 — Receptive field.
Left: theoretical receptive field growth across layer depth for a plain
stride-1 stack of 3x3 convs vs. a stack that halves spatial size every
other layer (stride-2) -- shows linear vs. multiplicative growth.
Right: effective receptive field. Repeatedly self-convolving a small box
kernel (what a stack of conv layers does to an impulse) converges to a
Gaussian by the Central Limit Theorem -- the "effective" receptive field
concentrates near the center long before the theoretical square is full.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Left panel: theoretical RF growth ---
n_layers = 10
k = 3  # 3x3 kernel every layer

# Plain stack: stride 1 everywhere -> RF grows by (k-1) per layer (linear)
rf_plain = [1]
jump = 1
for _ in range(n_layers):
    rf_plain.append(rf_plain[-1] + (k - 1) * jump)

# Strided stack: stride 2 every other layer -> jump doubles periodically (multiplicative)
rf_strided = [1]
jump = 1
for i in range(n_layers):
    rf_strided.append(rf_strided[-1] + (k - 1) * jump)
    if i % 2 == 1:
        jump *= 2

# --- Right panel: effective receptive field via repeated self-convolution ---
box = np.array([1.0, 1.0, 1.0])  # a single 3-tap "conv layer" impulse response
depths_to_show = [1, 3, 6, 10]
profiles = {}
kernel = np.array([1.0])
for d in range(1, max(depths_to_show) + 1):
    kernel = np.convolve(kernel, box)
    kernel_norm = kernel / kernel.sum()
    if d in depths_to_show:
        profiles[d] = kernel_norm

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].plot(range(n_layers + 1), rf_plain, marker="o", label="stride-1 stack (linear)")
axes[0].plot(range(n_layers + 1), rf_strided, marker="s", label="stride-2 every other layer")
axes[0].set_xlabel("layer depth")
axes[0].set_ylabel("theoretical receptive field (px)")
axes[0].set_title("RF growth: linear vs. multiplicative")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(depths_to_show)))
for (d, prof), c in zip(profiles.items(), colors):
    centered_x = np.arange(len(prof)) - len(prof) // 2
    axes[1].plot(centered_x, prof, color=c, label=f"depth {d}")
axes[1].set_xlabel("position relative to center")
axes[1].set_ylabel("normalized contribution")
axes[1].set_title("Effective RF converges to a Gaussian (CLT)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_37.png", dpi=120, bbox_inches="tight")
