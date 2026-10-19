"""
Day 139 - Consolidation: the spectral thread
(convolution theorem -> stride/pooling aliasing -> transposed-conv upsampling replicas -> sinusoidal positions).
Left : stride-2 downsampling of a 1-D signal with a tone at 0.375 cycles/sample (above the new Nyquist 0.25).
       Without a low-pass filter the tone folds to 0.25; a blur-then-subsample ("BlurPool") suppresses it.
Right: zero-insertion upsampling (what a transposed conv does before its kernel) creates a spectral replica;
       a smooth kernel removes it, a poorly matched kernel leaves it (the checkerboard artifact).
Synthetic, seed 42, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 256
t = np.arange(n)


def psd(x):
    X = np.fft.rfft(x * np.hanning(len(x)))
    return np.abs(X) ** 2 / len(x), np.fft.rfftfreq(len(x))


# ---- Left: aliasing under stride-2 ----------------------------------------------------
x = np.sin(2 * np.pi * 0.0625 * t) + np.sin(2 * np.pi * 0.375 * t) + 0.01 * np.random.randn(n)
raw = x[::2]                                         # plain stride-2 conv with a delta kernel
k = np.array([1, 4, 6, 4, 1], dtype=float) / 16      # binomial low-pass
blur = np.convolve(x, k, mode="same")[::2]           # blur first, then subsample
Pr, fr = psd(raw)
Pb, fb = psd(blur)
fold = np.argmin(np.abs(fr - 0.25))                  # 0.375 cycles/sample at the old rate is 0.75 at the new rate, which folds to 0.25
print(f"stride-2 energy at folded freq 0.25: no filter={Pr[fold]:.2f}, blur first={Pb[fold]:.4f} "
      f"({10*np.log10(Pr[fold]/Pb[fold]):.1f} dB lower)")

# ---- Right: zero-insertion upsampling --------------------------------------------------
m = 128
s = np.sin(2 * np.pi * 0.08 * np.arange(m)) + 0.01 * np.random.randn(m)
up = np.zeros(2 * m)
up[::2] = s                                          # transposed conv with stride 2 starts here
smooth = np.convolve(up, [0.5, 1.0, 0.5], mode="same")   # linear-interp kernel (cancels the replica at 0.5-0.04)
bad = np.convolve(up, [1.0, 0.0, 0.0], mode="same")      # kernel that ignores the zero gaps
Pu, fu = psd(up)
Ps, _ = psd(smooth)
Pbd, _ = psd(bad)
rep = np.argmin(np.abs(fu - (0.5 - 0.04)))           # replica of 0.04 cycles/sample
print(f"replica at 0.46: zero-insert={Pu[rep]:.3f}, linear kernel={Ps[rep]:.4f}, "
      f"ignoring-gap kernel={Pbd[rep]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(fr, Pr + 1e-8, label="stride-2, no filter")
ax[0].semilogy(fb, Pb + 1e-8, label="blur then stride-2")
ax[0].axvline(0.25, color="gray", ls=":")
ax[0].set(title="Aliasing: 0.375 folds to 0.25", xlabel="cycles/sample (after stride)", ylabel="PSD")
ax[0].legend()
ax[1].semilogy(fu, Pu + 1e-8, label="zero-insertion")
ax[1].semilogy(fu, Ps + 1e-8, label="linear-interp kernel")
ax[1].semilogy(fu, Pbd + 1e-8, label="kernel ignoring gaps", alpha=0.6)
ax[1].set(title="Upsampling replica near 0.46", xlabel="cycles/sample (output)", ylabel="PSD")
ax[1].legend()
plt.savefig("graph_DAY_139.png", dpi=120, bbox_inches="tight")
