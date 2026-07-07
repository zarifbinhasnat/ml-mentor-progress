"""
Feature maps and channels, made visible.

One tiny input image, one conv layer with 4 different learned-looking
kernels (edge-vertical, edge-horizontal, blob, sharpen). Each kernel
produces its own output channel -- a "feature map" that lights up for
a different pattern in the same input. This is the picture behind
"a conv layer with C_out kernels produces a C_out-channel output."
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Tiny synthetic 16x16 input: a bright square + a soft blob ----
n = 16
img = np.zeros((n, n))
img[4:11, 4:11] = 1.0                                   # a hard-edged square
yy, xx = np.mgrid[0:n, 0:n]
img += 0.6 * np.exp(-((yy - 11) ** 2 + (xx - 11) ** 2) / 3.0)  # soft blob overlapping the corner
img += 0.03 * np.random.randn(n, n)                     # tiny sensor noise
img = np.clip(img, 0, None)

# ---- Four hand-picked 3x3 kernels -- stand-ins for four LEARNED filters ----
kernels = {
    "vertical edges": np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=float),   # Sobel-x: fires on left-right transitions
    "horizontal edges": np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=float),  # Sobel-y: fires on top-bottom transitions
    "blob / smooth": np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=float) / 16.0,  # low-pass: fires on smooth bright regions
    "sharpen": np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=float),          # high-pass: fires on local contrast
}


def conv2d_same(image, kernel):
    """Stride-1, same-padding cross-correlation (today's formula, p=(k-1)/2)."""
    k = kernel.shape[0]
    p = (k - 1) // 2
    padded = np.pad(image, p, mode="constant", constant_values=0.0)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded[i:i + k, j:j + k]
            out[i, j] = np.sum(patch * kernel)
    return out

feature_maps = {name: conv2d_same(img, k) for name, k in kernels.items()}

fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))

axes[0].imshow(img, cmap="gray")
axes[0].set_title(f"Input (1 channel)\n{img.shape[0]}x{img.shape[1]}", fontsize=10)
axes[0].set_xticks([]); axes[0].set_yticks([])

for ax, (name, fmap) in zip(axes[1:], feature_maps.items()):
    im = ax.imshow(fmap, cmap="coolwarm")
    ax.set_title(f"channel: {name}\n{fmap.shape[0]}x{fmap.shape[1]}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("1 input, 4 kernels -> 4-channel output: each channel is a different learned 'question' about the same image",
             fontsize=11)
plt.savefig("graph_DAY_35.png", dpi=120, bbox_inches="tight")
