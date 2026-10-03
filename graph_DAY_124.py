"""
Day 124 - Consolidation Phase 9: why U-Net skips (and ViT/MAE/DINO design) care about frequency.

Left : a 1D "image row" = low-frequency structure + fine detail. A bottleneck (downsample 8x, upsample)
       is a low-pass filter, so reconstruction error lives in the high band.
Right: adding the skip (the high-frequency residual from the encoder) restores that band.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 128
t = np.arange(n)
x = np.sin(2 * np.pi * 3 * t / n) + 0.5 * np.sin(2 * np.pi * 9 * t / n) \
    + 0.4 * np.sin(2 * np.pi * 40 * t / n) + 0.3 * np.sin(2 * np.pi * 52 * t / n)

# bottleneck: keep only the lowest 16 rFFT bins (what an 8x down/up path can represent)
X = np.fft.rfft(x)
Xb = X.copy()
Xb[16:] = 0
x_bottleneck = np.fft.irfft(Xb, n)

# skip connection: decoder also receives the encoder's high-frequency detail
x_skip = x_bottleneck + (x - np.fft.irfft(np.where(np.arange(X.size) < 16, X, 0), n))

def spec(v):
    return np.abs(np.fft.rfft(v)) / n

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(t, x, label="input", color="black", lw=1.5)
axes[0].plot(t, x_bottleneck, label="bottleneck only", color="tab:red", lw=1.2)
axes[0].plot(t, x_skip, label="with skip", color="tab:green", ls="--", lw=1.2)
axes[0].set_title("Reconstruction of one image row")
axes[0].set_xlabel("pixel")
axes[0].legend(fontsize=8)

f = np.arange(X.size)
axes[1].semilogy(f, spec(x - x_bottleneck) + 1e-12, label="error, bottleneck only", color="tab:red")
axes[1].semilogy(f, spec(x - x_skip) + 1e-12, label="error, with skip", color="tab:green")
axes[1].axvline(16, color="gray", ls="--", lw=0.8)
axes[1].set_title("Error spectrum: bottleneck loses the high band")
axes[1].set_xlabel("frequency bin")
axes[1].set_ylabel("|error spectrum|")
axes[1].legend(fontsize=8)

plt.savefig("graph_DAY_124.png", dpi=120, bbox_inches="tight")
