"""
The convolution operation, made visible.

A small synthetic "image" contains a vertical edge and a diagonal line.
We slide a 3x3 vertical-edge kernel (a Sobel-style operator) across it
and plot the input alongside the resulting feature map, to make the
"kernel as a learned template detector" intuition concrete.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Build a tiny synthetic 16x16 "image" -----------------------------
n = 16
img = np.zeros((n, n))
img[:, n // 2:] = 1.0                      # a vertical step edge (left dark, right bright)
for i in range(n):                         # a faint diagonal line crossing the edge
    j = i
    if 0 <= j < n:
        img[i, j] += 0.5
img += 0.03 * np.random.randn(n, n)        # tiny sensor noise
img = np.clip(img, 0, None)

# ---- A vertical-edge-detecting kernel (Sobel-x) -----------------------
kernel = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
k = kernel.shape[0]

# ---- Manual "valid" cross-correlation (what nn.Conv2d actually computes) --
out_size = n - k + 1
feature_map = np.zeros((out_size, out_size))
for i in range(out_size):              # slide the kernel down...
    for j in range(out_size):          # ...and across the image
        patch = img[i:i + k, j:j + k]              # the k x k window under the kernel right now
        feature_map[i, j] = np.sum(patch * kernel)  # elementwise multiply + sum = one dot product

# ---- Plot input vs. output feature map --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

im0 = axes[0].imshow(img, cmap="gray", vmin=0, vmax=1.6)
axes[0].set_title("Input image\n(vertical edge + diagonal line)")
axes[0].set_xticks([]); axes[0].set_yticks([])
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(feature_map, cmap="RdBu_r",
                      vmin=-np.abs(feature_map).max(), vmax=np.abs(feature_map).max())
axes[1].set_title("Output feature map\n(3x3 vertical-edge kernel slid across it)")
axes[1].set_xticks([]); axes[1].set_yticks([])
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

fig.suptitle("Convolution = the same small kernel dot-producted against every local window", y=1.03)
fig.tight_layout()
fig.savefig("graph_DAY_33.png", dpi=120, bbox_inches="tight")
