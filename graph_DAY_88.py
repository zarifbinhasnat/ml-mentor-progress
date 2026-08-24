import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- Build two synthetic 64x64 "images" ----------
N = 64

# "Real" image: 1/f (pink) noise -- roughly matches the smooth radial power-law
# falloff of natural image statistics, with no periodic structure.
freq_y = np.fft.fftfreq(N).reshape(-1, 1)
freq_x = np.fft.fftfreq(N).reshape(1, -1)
radius = np.sqrt(freq_y ** 2 + freq_x ** 2)
radius[0, 0] = radius[0, 1]  # avoid divide-by-zero at the DC term

white = np.random.randn(N, N) + 1j * np.random.randn(N, N)
pink_spectrum = white / (radius ** 1.0)          # amplitude falls off ~1/r -> power falls off ~1/r^2
real_img = np.fft.ifft2(pink_spectrum).real
real_img = (real_img - real_img.mean()) / real_img.std()

# "Fake" image: simulate a decoder that upsamples a coarse 8x8 latent via
# naive nearest-neighbor (like a naive transposed-conv / PixelShuffle artifact)
# then applies a mild smoothing conv -- this is exactly the Day 87 decoder-hole
# story, and it injects energy at the periodic upsampling grid frequency.
factor = 8
coarse = np.random.randn(N // factor, N // factor)
upsampled = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)  # nearest-neighbor upsample
blur_kernel = np.outer(np.hanning(5), np.hanning(5))
blur_kernel /= blur_kernel.sum()
fake_img = np.real(np.fft.ifft2(np.fft.fft2(upsampled) * np.fft.fft2(blur_kernel, s=upsampled.shape)))
fake_img = np.fft.fftshift(fake_img)
fake_img = (fake_img - fake_img.mean()) / fake_img.std()

# ---------- 2D log-magnitude spectra ----------
def log_mag_spectrum(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F))

spec_real = log_mag_spectrum(real_img)
spec_fake = log_mag_spectrum(fake_img)

# ---------- Azimuthal average: log-power vs radius ----------
def azimuthal_average(spec):
    yy, xx = np.indices(spec.shape)
    cy, cx = N // 2, N // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    r_max = r.max()
    tbin = np.bincount(r.ravel(), weights=spec.ravel(), minlength=r_max + 1)
    count = np.bincount(r.ravel(), minlength=r_max + 1)
    return tbin / np.maximum(count, 1)

profile_real = azimuthal_average(spec_real)
profile_fake = azimuthal_average(spec_fake)
radii = np.arange(len(profile_real))

# ---------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

axes[0].imshow(spec_real, cmap="magma")
axes[0].set_title("real (1/f noise) log-spectrum")
axes[0].axis("off")

axes[1].imshow(spec_fake, cmap="magma")
axes[1].set_title("fake (upsample-artifact) log-spectrum")
axes[1].axis("off")

ax = axes[2]
ax.plot(radii, profile_real, color="black", label="real: smooth falloff")
ax.plot(radii, profile_fake, color="firebrick", linestyle="--", label="fake: periodic spike")
nyquist_grid = N / (2 * factor)
ax.axvline(nyquist_grid, color="gray", linestyle=":", linewidth=1, label="upsample-grid frequency")
ax.set_xlabel("radial frequency (pixels from DC)")
ax.set_ylabel("azimuthally-averaged log power")
ax.set_title("Radial spectrum: real vs. generated")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_88.png", dpi=120, bbox_inches="tight")
