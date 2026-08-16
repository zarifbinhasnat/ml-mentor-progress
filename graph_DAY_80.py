"""
Day 80 -- VAE decoder & upsampling artifacts.

Demonstrates checkerboard artifacts from transposed convolution in both the
spatial AND frequency domain. The classic, cleanest way to see this (per
Odena et al., "Deconvolution and Checkerboard Artifacts") is the *overlap
count* map: how many times does each output pixel get touched by the
kernel as it strides across? Feed the transposed conv an all-ones input
and an all-ones kernel, and the output IS the overlap-count map. When
kernel_size % stride != 0, that count is spatially periodic -> a sharp
peak appears in the 2D FFT at the checkerboard frequency. When
kernel_size % stride == 0, the count is uniform -> flat spectrum (just DC).

No downloads, no torch dependency -- transposed conv implemented directly
with numpy via the standard "insert zeros, then overlap-add the kernel"
definition, which is mathematically identical to nn.ConvTranspose2d.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def conv_transpose2d(x, kernel, stride):
    """Fractionally-strided (transposed) convolution via overlap-add."""
    in_h, in_w = x.shape
    k = kernel.shape[0]
    out_h = (in_h - 1) * stride + k
    out_w = (in_w - 1) * stride + k
    out = np.zeros((out_h, out_w))
    for i in range(in_h):
        for j in range(in_w):
            out[i * stride:i * stride + k, j * stride:j * stride + k] += x[i, j] * kernel
    return out


stride = 2
ones_input = np.ones((16, 16))            # constant "activation" -> isolates overlap count
bad_kernel = np.ones((3, 3))               # 3 % 2 != 0 -> uneven overlap
good_kernel = np.ones((4, 4))              # 4 % 2 == 0 -> uniform overlap

bad_overlap = conv_transpose2d(ones_input, bad_kernel, stride)
good_overlap = conv_transpose2d(ones_input, good_kernel, stride)

# Crop borders where edge effects (not periodicity) dominate the count.
bad_overlap_c = bad_overlap[6:-6, 6:-6]
good_overlap_c = good_overlap[6:-6, 6:-6]


def log_magnitude_spectrum(img):
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    return np.log1p(np.abs(f))


bad_spec = log_magnitude_spectrum(bad_overlap_c)
good_spec = log_magnitude_spectrum(good_overlap_c)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

im0 = axes[0, 0].imshow(bad_overlap_c, cmap="viridis")
axes[0, 0].set_title("Overlap count: k=3, s=2 (checkerboard)")
axes[0, 0].axis("off")
fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(good_overlap_c, cmap="viridis")
axes[0, 1].set_title("Overlap count: k=4, s=2 (uniform)")
axes[0, 1].axis("off")
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

axes[1, 0].imshow(bad_spec, cmap="magma")
axes[1, 0].set_title("|FFT| (log, DC removed) -- Nyquist spike")
axes[1, 0].axis("off")

axes[1, 1].imshow(good_spec, cmap="magma")
axes[1, 1].set_title("|FFT| (log, DC removed) -- flat, no spike")
axes[1, 1].axis("off")

fig.suptitle("Checkerboard artifacts: spatial overlap count vs. its FFT", fontsize=13)
fig.tight_layout()
plt.savefig("graph_DAY_80.png", dpi=120, bbox_inches="tight")
