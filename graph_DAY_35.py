"""Day 35 - Feature maps & channels: same input, three kernels, three feature maps."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

size = 16
xx, yy = np.meshgrid(np.arange(size), np.arange(size))
img = (xx / size) * 0.3                      # soft diagonal-ish gradient background
img[4:10, 4:10] += 1.0                       # a bright square (a "structure" for kernels to react to)
img += 0.05 * np.random.randn(size, size)    # small sensor-noise term
img = img.astype(np.float64)


def conv2d_same(image, kernel):
    """Manual same-padding 2D correlation (what nn.Conv2d actually computes, no kernel flip)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


kernels = {
    "vertical-edge (Sobel-x)": np.array([[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]], dtype=np.float64),
    "horizontal-edge (Sobel-y)": np.array([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]], dtype=np.float64),
    "blur (box)": np.ones((3, 3), dtype=np.float64) / 9.0,
}

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("input (1 channel)", fontsize=10)
axes[0].axis("off")

for ax, (name, k) in zip(axes[1:], kernels.items()):
    fmap = conv2d_same(img, k)
    ax.imshow(fmap, cmap="gray")
    ax.set_title(name, fontsize=9)
    ax.axis("off")

fig.suptitle("One input, three learned kernels -> three different output-channel feature maps", fontsize=11)
plt.savefig("graph_DAY_35.png", dpi=120, bbox_inches="tight")
print("saved graph_DAY_35.png")
