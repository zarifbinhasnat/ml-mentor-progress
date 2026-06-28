"""
Day 29: Layer Normalization vs Batch Normalization
Visualizes normalization axes on a synthetic (B=5, F=8) activation matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

B, F = 5, 8
# Synthetic activations with intentional scale differences per feature
scale = np.array([0.5, 2.0, 0.3, 1.5, 3.0, 0.8, 1.2, 2.5])
X = np.random.randn(B, F) * scale + np.array([0, 1, -1, 2, 0.5, -0.5, 1.5, -1.5])

eps = 1e-5

# --- BatchNorm: normalize over the Batch dimension for each Feature ---
mu_bn = X.mean(axis=0, keepdims=True)        # shape (1, F)
std_bn = X.std(axis=0, keepdims=True) + eps  # shape (1, F)
X_bn = (X - mu_bn) / std_bn

# --- LayerNorm: normalize over the Feature dimension for each Sample ---
mu_ln = X.mean(axis=1, keepdims=True)        # shape (B, 1)
std_ln = X.std(axis=1, keepdims=True) + eps  # shape (B, 1)
X_ln = (X - mu_ln) / std_ln

# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#0f1117")

vmin, vmax = -3, 3
cmap = "RdBu_r"

def draw_heatmap(ax, data, title, highlight_axis):
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.set_xlabel("Features (F=8)", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Samples (B=5)", color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Draw arrows showing which axis is normalized
    if highlight_axis == "batch":
        # Arrow pointing down along batch axis (column-wise)
        for f in range(F):
            ax.annotate("", xy=(f, B - 0.5), xytext=(f, -0.5),
                        arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.2),
                        annotation_clip=False)
        ax.text(F / 2 - 0.5, B + 0.3, "↓ normalized over batch",
                color="#ff6b6b", fontsize=9, ha="center", transform=ax.transData)
    elif highlight_axis == "layer":
        # Arrow pointing right along feature axis (row-wise)
        for b in range(B):
            ax.annotate("", xy=(F - 0.5, b), xytext=(-0.5, b),
                        arrowprops=dict(arrowstyle="->", color="#6bdfff", lw=1.2),
                        annotation_clip=False)
        ax.text(F + 0.5, B / 2, "→ normalized\nover features",
                color="#6bdfff", fontsize=9, ha="left", va="center", transform=ax.transData)
    return im

draw_heatmap(axes[0], X, "Raw Activations X", highlight_axis=None)
draw_heatmap(axes[1], X_bn, "After BatchNorm\n(normalize each column ↓)", highlight_axis="batch")
im = draw_heatmap(axes[2], X_ln, "After LayerNorm\n(normalize each row →)", highlight_axis="layer")

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
cbar.ax.yaxis.set_tick_params(color="white")
cbar.ax.tick_params(colors="white")
cbar.set_label("Activation value (normalized)", color="#aaaaaa", fontsize=9)

fig.suptitle("BatchNorm vs LayerNorm — Which axis gets normalized?",
             color="white", fontsize=14, y=1.02)

plt.tight_layout()
plt.savefig("graph_DAY_29.png", dpi=120, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved graph_DAY_29.png")
