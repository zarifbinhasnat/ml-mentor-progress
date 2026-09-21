"""
Day 112 -- Evaluation, benchmarks & pitfalls
Signal Lab: radially-averaged power spectral density (PSD) as a spectral
evaluation metric that catches generative-model artifacts pixel-space
metrics (FID/IS) can miss.

Synthetic-only, no downloads. Seeds fixed for reproducibility.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N = 64  # small synthetic image side length


def radial_psd(img):
    """Radially-averaged power spectral density of a 2D image."""
    F = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(F) ** 2

    cy, cx = N // 2, N // 2
    y, x = np.indices((N, N))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    r_max = r.max()
    radial_mean = np.zeros(r_max + 1)
    for radius in range(r_max + 1):
        mask = r == radius
        if mask.any():
            radial_mean[radius] = power[mask].mean()
    return radial_mean


# "real" image: natural images are ~1/f (pink-noise) in the frequency domain.
# Build one directly in the frequency domain, then inverse-FFT to pixel space.
freqs_y = np.fft.fftfreq(N).reshape(-1, 1)
freqs_x = np.fft.fftfreq(N).reshape(1, -1)
radius = np.sqrt(freqs_y ** 2 + freqs_x ** 2)
radius[0, 0] = radius[0, 0] + 1e-6  # avoid divide-by-zero at DC

pink_spectrum = 1.0 / (radius ** 1.0)
random_phase = np.exp(2j * np.pi * np.random.rand(N, N))
real_img = np.real(np.fft.ifft2(pink_spectrum * random_phase))
real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min())

# "generated" image: same pink-noise content, but pass it through a
# nearest-neighbor 2x upsample (a cheap stand-in for a transposed-conv /
# checkerboard-artifact decoder) -- this stamps a periodic high-frequency
# comb onto the spectrum that FID/IS on their own would likely miss.
small = real_img[::2, ::2]
generated_img = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)

psd_real = radial_psd(real_img)
psd_fake = radial_psd(generated_img)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

axes[0].imshow(real_img, cmap="gray")
axes[0].set_title("real (1/f pink noise)")
axes[0].axis("off")

freqs = np.arange(len(psd_real))
axes[1].plot(freqs[1:], psd_real[1:], label="real", color="#1f77b4")
axes[1].plot(freqs[1:], psd_fake[1:], label="generated (NN-upsampled)", color="#d62728")
axes[1].set_yscale("log")
axes[1].set_xlabel("radial spatial frequency (bin)")
axes[1].set_ylabel("power (log scale)")
axes[1].set_title("radially-averaged PSD")
axes[1].legend()

plt.tight_layout()
plt.savefig("graph_DAY_112.png", dpi=120, bbox_inches="tight")
