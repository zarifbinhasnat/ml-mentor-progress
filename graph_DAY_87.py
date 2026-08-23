import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- Panel 1: self-attention FLOPs, pixel-space diffusion vs latent-space diffusion ----------
resolutions = np.array([64, 128, 256, 512, 1024])   # H = W, pixel-space image side length
downsample = 8                                       # SD's VAE: 8x spatial downsample per side

n_pixels = resolutions ** 2                          # tokens if you ran diffusion directly on pixels
n_latents = (resolutions // downsample) ** 2         # tokens actually seen by the U-Net in latent space

flops_pixel = n_pixels ** 2                          # self-attention cost ~ O(n^2 * d), d held fixed
flops_latent = n_latents ** 2

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(resolutions, flops_pixel, marker="o", color="firebrick", label="pixel-space diffusion")
ax.plot(resolutions, flops_latent, marker="o", color="steelblue", label=f"latent-space ({downsample}x8 VAE downsample)")
ax.set_yscale("log")
ax.set_xscale("log", base=2)
ax.set_xticks(resolutions)
ax.set_xticklabels(resolutions)
ax.set_xlabel("image side length (pixels)")
ax.set_ylabel(r"relative self-attention cost ($n^2$, log scale)")
ax.set_title("Why diffuse in latent space: attention cost")
ax.legend(fontsize=8)

# ---------- Panel 2: the DSP cost of that compression -- VAE bottleneck = low-pass + downsample ----------
n = 80
t = np.linspace(0, 1, n, endpoint=False)
# a synthetic "image row": a coarse structure term plus fine high-frequency texture
signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 18 * t)  # low-freq shape + high-freq texture

factor = 8  # mirrors the VAE's 8x spatial downsample
# simulate the encoder: anti-alias low-pass (box filter) then decimate -- this is what a real VAE encoder
# approximates by learning to discard information a decoder can't reconstruct anyway
kernel = np.ones(factor) / factor
smoothed = np.convolve(signal, kernel, mode="same")
latent = smoothed[::factor]                     # the compressed "latent" row, 8x fewer samples
recon = np.repeat(latent, factor)               # naive decoder: nearest-neighbor upsample back to n

freqs = np.fft.rfftfreq(n, d=t[1] - t[0])
psd_orig = np.abs(np.fft.rfft(signal)) ** 2
psd_recon = np.abs(np.fft.rfft(recon)) ** 2

ax = axes[1]
ax.plot(freqs, psd_orig, color="black", label="original PSD")
ax.plot(freqs, psd_recon, color="steelblue", linestyle="--", label="post VAE-bottleneck PSD")
ax.axvline(n / (2 * factor), color="gray", linestyle=":", linewidth=1, label="latent Nyquist limit")
ax.set_xlabel("frequency")
ax.set_ylabel("power")
ax.set_title("VAE bottleneck = low-pass: texture band lost")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_87.png", dpi=120, bbox_inches="tight")
