"""
Day 30 – Data Augmentation
Shows a synthetic image (grayscale circle on noise) alongside several
common augmentations: horizontal flip, random crop, brightness jitter,
Gaussian blur, and cutout. All transforms are implemented from scratch
with numpy — no torchvision dependency required.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(42)

# ── Build a tiny synthetic "image" (64×64 grayscale) ──────────────────────────
H, W = 64, 64
img = np.random.rand(H, W) * 0.15          # low-amplitude noise background

# Draw a bright circle at centre-left — acts as our "object"
cy, cx = 32, 20
for y in range(H):
    for x in range(W):
        if (y - cy) ** 2 + (x - cx) ** 2 < 10 ** 2:
            img[y, x] = 0.85 + np.random.rand() * 0.1


def hflip(x):
    """Horizontal flip — mirror along vertical axis."""
    return x[:, ::-1].copy()


def random_crop(x, crop_h=48, crop_w=48):
    """Crop then resize (here we just crop; no resize for simplicity)."""
    top  = np.random.randint(0, x.shape[0] - crop_h + 1)
    left = np.random.randint(0, x.shape[1] - crop_w + 1)
    cropped = x[top:top + crop_h, left:left + crop_w]
    # Pad back to original size so shapes match for display
    pad_top  = (H - crop_h) // 2
    pad_left = (W - crop_w) // 2
    out = np.zeros_like(x)
    out[pad_top:pad_top + crop_h, pad_left:pad_left + crop_w] = cropped
    return out


def brightness_jitter(x, delta=0.3):
    """Add a random brightness offset — simulates lighting variation."""
    shift = np.random.uniform(-delta, delta)
    return np.clip(x + shift, 0, 1)


def gaussian_blur(x, sigma=1.5):
    """Apply a small Gaussian blur by hand (3×3 kernel)."""
    k = 3
    half = k // 2
    ax = np.arange(-half, half + 1)
    gk = np.exp(-ax ** 2 / (2 * sigma ** 2))
    gk /= gk.sum()
    kernel = np.outer(gk, gk)              # 2-D separable kernel
    out = np.zeros_like(x)
    for i in range(half, H - half):
        for j in range(half, W - half):
            patch = x[i - half:i + half + 1, j - half:j + half + 1]
            out[i, j] = (patch * kernel).sum()
    return out


def cutout(x, size=16):
    """Zero out a random square patch — forces model to use context."""
    out = x.copy()
    top  = np.random.randint(0, H - size + 1)
    left = np.random.randint(0, W - size + 1)
    out[top:top + size, left:left + size] = 0.0
    return out, top, left, size


# ── Generate augmented versions ───────────────────────────────────────────────
np.random.seed(7)
aug_flip    = hflip(img)
aug_crop    = random_crop(img)
aug_bright  = brightness_jitter(img)
aug_blur    = gaussian_blur(img)
aug_cut, ct, cl, cs = cutout(img)

# ── Plot ──────────────────────────────────────────────────────────────────────
titles = [
    "Original",
    "H-Flip",
    "Random Crop\n(+ pad)",
    "Brightness\nJitter",
    "Gaussian\nBlur",
    "Cutout",
]
images = [img, aug_flip, aug_crop, aug_bright, aug_blur, aug_cut]

fig, axes = plt.subplots(1, 6, figsize=(14, 3))
fig.suptitle("Data Augmentation — same image, multiple views", fontsize=13, fontweight="bold")

for ax, title, im in zip(axes, titles, images):
    ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

# Highlight cutout region on the last panel
rect = patches.Rectangle(
    (cl, ct), cs, cs,
    linewidth=1.5, edgecolor="red", facecolor="none"
)
axes[-1].add_patch(rect)

plt.tight_layout()
plt.savefig("graph_DAY_30.png", dpi=120, bbox_inches="tight")
print("Saved graph_DAY_30.png")
