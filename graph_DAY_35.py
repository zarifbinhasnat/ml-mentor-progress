"""
Day 35 — Feature maps & channels.

One input "image" convolved with 4 different 3x3 kernels produces 4 distinct
feature maps (channels) — same spatial layout, different learned response.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Synthetic 16x16 "image": a bright square on a soft gradient background ----
n = 16
yy, xx = np.mgrid[0:n, 0:n]
img = 0.15 * xx / n + 0.05 * np.random.randn(n, n)          # gentle gradient + noise floor
img[4:11, 4:11] += 1.0                                      # a square "object"
img = np.clip(img, 0, None)

# ---- Four small, hand-designed kernels (stand-ins for four learned filters) ----
kernels = {
    "vertical edges (Sobel-x)": np.array([[-1, 0, 1],
                                           [-2, 0, 2],
                                           [-1, 0, 1]], dtype=float),
    "horizontal edges (Sobel-y)": np.array([[-1, -2, -1],
                                             [ 0,  0,  0],
                                             [ 1,  2,  1]], dtype=float),
    "blob detector (Gaussian)": np.array([[1, 2, 1],
                                           [2, 4, 2],
                                           [1, 2, 1]], dtype=float) / 16.0,
    "diagonal edges": np.array([[ 2,  1, 0],
                                 [ 1,  0, -1],
                                 [ 0, -1, -2]], dtype=float),
}


def conv2d_valid(x, k):
    """Direct 'valid' 2D convolution (no padding) — small images, so a plain loop is fine."""
    kh, kw = k.shape
    oh, ow = x.shape[0] - kh + 1, x.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i:i + kh, j:j + kw] * k)
    return out


feature_maps = {name: conv2d_valid(img, k) for name, k in kernels.items()}

# ---- Plot: input on the left, four output channels on the right ----
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Input\n(1 channel)")
axes[0].set_xticks([]); axes[0].set_yticks([])

for ax, (name, fmap) in zip(axes[1:], feature_maps.items()):
    ax.imshow(fmap, cmap="RdBu_r", vmin=-np.abs(fmap).max(), vmax=np.abs(fmap).max())
    ax.set_title(name, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("One input, four kernels -> four feature maps stacked as channels", y=1.05)
plt.savefig("graph_DAY_35.png", dpi=120, bbox_inches="tight")
