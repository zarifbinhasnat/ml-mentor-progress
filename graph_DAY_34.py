"""
Kernels, stride, and padding, made visible.

Same 3x3 kernel, same input image -- three different (stride, padding)
settings produce three very different output feature-map sizes. This
makes the output-size formula concrete instead of abstract algebra.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Build a tiny synthetic 12x12 "image": a soft blob on a dark field ----
n = 12
yy, xx = np.mgrid[0:n, 0:n]
cy, cx = 5.5, 6.5
img = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 6.0)   # smooth Gaussian bump
img += 0.03 * np.random.randn(n, n)                       # tiny sensor noise

# ---- A simple averaging/blur kernel (keeps the demo about geometry, ----
# ---- not about what the kernel detects) --------------------------------
kernel = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]], dtype=float)
kernel /= kernel.sum()
k = kernel.shape[0]


def conv2d(image, kernel, stride=1, padding=0):
    """Manual cross-correlation with explicit stride and zero-padding."""
    k = kernel.shape[0]
    if padding > 0:
        image = np.pad(image, padding, mode="constant", constant_values=0.0)
    h, w = image.shape
    out_h = (h - k) // stride + 1                 # the formula from today's lesson
    out_w = (w - k) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride           # top-left corner of this window, jumping by `stride`
            patch = image[r:r + k, c:c + k]
            out[i, j] = np.sum(patch * kernel)
    return out


# Three settings, same kernel, same input:
valid = conv2d(img, kernel, stride=1, padding=0)      # "valid": shrinks
same = conv2d(img, kernel, stride=1, padding=(k - 1) // 2)   # "same": size-preserving
strided = conv2d(img, kernel, stride=2, padding=(k - 1) // 2)  # stride-2 downsample

panels = [
    (img, f"Input\n{img.shape[0]}x{img.shape[1]}"),
    (valid, f"stride=1, padding=0 ('valid')\n{valid.shape[0]}x{valid.shape[1]}"),
    (same, f"stride=1, padding=1 ('same')\n{same.shape[0]}x{same.shape[1]}"),
    (strided, f"stride=2, padding=1\n{strided.shape[0]}x{strided.shape[1]}"),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (data, title) in zip(axes, panels):
    im = ax.imshow(data, cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Same 3x3 kernel, same input -- stride & padding change the output geometry", fontsize=11)
plt.savefig("graph_DAY_34.png", dpi=120, bbox_inches="tight")
