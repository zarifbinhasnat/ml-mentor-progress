"""
Day 123 - Consolidation Phase 8: generators leave spectral fingerprints.

Left : forward diffusion x_t = sqrt(a)*x0 + sqrt(1-a)*eps on a 1D signal with a 1/f^2-like spectrum.
       White noise has a flat PSD, so the high-frequency end of the signal is buried first.
Right: zero-insertion (stride-2 transposed-conv style) upsampling creates a spectral replica
       (mirror image) of the low band, the classic checkerboard / upsampling artifact.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 256

# ---- synthetic "natural" signal: random phases, amplitude ~ 1/f ----
freqs = np.fft.rfftfreq(n)
amp = np.zeros_like(freqs)
amp[1:] = 1.0 / freqs[1:]
phase = np.exp(2j * np.pi * np.random.rand(freqs.size))
x0 = np.fft.irfft(amp * phase, n)
x0 = x0 / x0.std()


def psd(x, reps=1):
    return np.abs(np.fft.rfft(x)) ** 2 / n


# ---- left: forward noising ----
alphas_bar = [1.0, 0.5, 0.1, 0.01]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for a in alphas_bar:
    # average over 20 noise draws to smooth the PSD
    p = np.mean([psd(np.sqrt(a) * x0 + np.sqrt(1 - a) * np.random.randn(n)) for _ in range(20)], axis=0)
    axes[0].loglog(freqs[1:], p[1:], label=f"abar = {a}")
axes[0].set_xlabel("normalized frequency")
axes[0].set_ylabel("PSD")
axes[0].set_title("Forward diffusion: noise floor swallows HF first")
axes[0].legend()

# ---- right: zero-insertion upsampling ----
m = 128
low = np.fft.irfft(np.r_[0, 1.0 / np.arange(1, 17), np.zeros(m // 2 - 16)] * np.exp(2j * np.pi * np.random.rand(m // 2 + 1)), m)
up = np.zeros(2 * m)
up[::2] = low                                       # zero insertion, no interpolation filter
f_up = np.fft.rfftfreq(2 * m)
axes[1].semilogy(f_up, np.abs(np.fft.rfft(up)) ** 2 / (2 * m) + 1e-12)
axes[1].axvline(0.25, color="gray", ls="--", lw=0.8)
axes[1].set_xlabel("normalized frequency (new rate)")
axes[1].set_ylabel("PSD")
axes[1].set_title("Zero-insertion upsampling: mirrored spectral replica")

ratio = np.abs(np.fft.rfft(up))[f_up > 0.25].max() / np.abs(np.fft.rfft(up))[f_up < 0.25].max()
print("replica / baseband peak amplitude ratio:", round(float(ratio), 2))
plt.savefig("graph_DAY_123.png", dpi=120, bbox_inches="tight")
