import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N = 64
x = np.arange(N)
xx, yy = np.meshgrid(x, x)

# Synthetic image: smooth low-frequency base + a fine checkerboard texture
# (the classic Nyquist-frequency signature of transposed-conv/upsampling artifacts).
base = np.sin(2 * np.pi * xx / 40) * np.cos(2 * np.pi * yy / 40)
checkerboard = 0.4 * np.sin(np.pi * xx) * np.sin(np.pi * yy)
img = base + checkerboard + 0.05 * np.random.randn(N, N)


def bilinear_resize(patch, out_size):
    h, w = patch.shape
    ys = np.linspace(0, h - 1, out_size)
    xs = np.linspace(0, w - 1, out_size)
    y0 = np.floor(ys).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.floor(xs).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = patch[y0][:, x0] * (1 - wx) + patch[y0][:, x1] * wx
    bot = patch[y1][:, x0] * (1 - wx) + patch[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def random_resized_crop(image, scale):
    n = image.shape[0]
    crop_size = max(4, int(n * scale))
    top = np.random.randint(0, n - crop_size + 1)
    left = np.random.randint(0, n - crop_size + 1)
    patch = image[top:top + crop_size, left:left + crop_size]
    return bilinear_resize(patch, n)


def radial_power_spectrum(image):
    F = np.fft.fftshift(np.fft.fft2(image))
    mag2 = np.abs(F) ** 2
    n = image.shape[0]
    cy, cx = n // 2, n // 2
    yy2, xx2 = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = np.sqrt((yy2 - cy) ** 2 + (xx2 - cx) ** 2).astype(int)
    r_max = r.max()
    radial = np.zeros(r_max + 1)
    for i in range(r_max + 1):
        mask = r == i
        if mask.any():
            radial[i] = mag2[mask].mean()
    return radial


scales = [1.0, 0.5, 0.2]
views = {1.0: img}
for s in scales[1:]:
    views[s] = random_resized_crop(img, s)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

titles = {1.0: "Original (scale=1.0)", 0.5: "RandomResizedCrop scale=0.5", 0.2: "RandomResizedCrop scale=0.2"}
for ax, s in zip(axes[0], scales):
    ax.imshow(views[s], cmap="gray")
    ax.set_title(titles[s])
    ax.axis("off")

ax = axes[1, 0]
colors = {1.0: "black", 0.5: "tab:orange", 0.2: "tab:red"}
for s in scales:
    radial = radial_power_spectrum(views[s])
    ax.plot(radial, label=f"scale={s}", color=colors[s])
ax.set_yscale("log")
ax.set_xlabel("spatial frequency radius (cycles/image)")
ax.set_ylabel("radial power (log)")
ax.set_title("Radially-averaged power spectrum")
ax.axvline(N / 2, color="gray", linestyle="--", linewidth=0.8)
ax.text(N / 2, ax.get_ylim()[1], " Nyquist", va="top", fontsize=8)
ax.legend(fontsize=8)

axes[1, 1].imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(views[1.0])))), cmap="inferno")
axes[1, 1].set_title("log|FFT| scale=1.0")
axes[1, 1].axis("off")

axes[1, 2].imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(views[0.2])))), cmap="inferno")
axes[1, 2].set_title("log|FFT| scale=0.2")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("graph_DAY_91.png", dpi=120, bbox_inches="tight")
