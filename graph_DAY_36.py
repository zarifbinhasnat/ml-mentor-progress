"""Max-pool vs avg-pool downsampling on a small synthetic feature map."""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # reproducible synthetic data, no downloads

# Build an 8x8 "feature map" as a conv layer might emit it: a strong diagonal
# edge response plus a few sparse activation spikes and background noise.
H, W = 8, 8
feat = np.random.randn(H, W) * 0.15
for i in range(H):
    feat[i, i] += 2.0          # diagonal ridge of "edge detected here" activations
feat[1, 5] += 3.0               # a couple of sparse, strong spikes
feat[6, 2] += 2.5

def pool2x2(x, mode):
    # Stride-2, kernel-2 pooling, written by hand (no framework) so the
    # window-reduction step is visible instead of hidden behind a library call.
    h, w = x.shape
    out = np.zeros((h // 2, w // 2))
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            window = x[i:i + 2, j:j + 2]
            out[i // 2, j // 2] = window.max() if mode == "max" else window.mean()
    return out

max_pooled = pool2x2(feat, "max")
avg_pooled = pool2x2(feat, "avg")

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
vmin, vmax = feat.min(), feat.max()

for ax, data, title in zip(
    axes,
    [feat, max_pooled, avg_pooled],
    ["Input feature map (8x8)", "Max-pooled (4x4)\nkeeps sharp spikes", "Avg-pooled (4x4)\nblurs / smooths"],
):
    im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="activation")
fig.suptitle("Pooling as downsampling: max keeps the strongest signal, avg keeps the local mean", fontsize=11)
plt.savefig("graph_DAY_36.png", dpi=120, bbox_inches="tight")
