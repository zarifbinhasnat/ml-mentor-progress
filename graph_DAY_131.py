"""
Day 131 - Consolidation: the aliasing / resampling thread.

Left : stride-2 downsampling (strided conv / pooling). A tone at 0.35 cycles/sample is above the new
       Nyquist (0.25 of the old rate), so naive decimation folds it to a false lower frequency.
       A [1,2,1]/4 blur before decimating (anti-aliased pooling) attenuates it.
Right: stride-2 upsampling (transposed conv). Zero-insertion copies the spectrum into a mirror image;
       a smoothing kernel after it suppresses the image. The residual image is the checkerboard artifact.
All data synthetic, seed 42, N = 128 samples.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 128
n = np.arange(N)
f0 = 0.35                                          # cycles/sample, above the post-decimation Nyquist (0.25)
x = np.sin(2 * np.pi * f0 * n) + 0.01 * np.random.randn(N)
blur = np.array([1, 2, 1]) / 4.0


def spec(sig):
    """One-sided magnitude spectrum, frequency in cycles/sample of that signal's own rate."""
    S = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) / len(sig) * 2
    return np.fft.rfftfreq(len(sig)), S


# ---- left: decimation with / without anti-aliasing ----
naive = x[::2]                                     # strided conv / max-style decimation, no filter
filtered = np.convolve(x, blur, mode="same")[::2]  # blur-pool: low-pass, then decimate
fx, Sx = spec(x)
fn, Sn = spec(naive)
ff, Sf = spec(filtered)

# ---- right: zero-insertion upsampling with / without a smoothing kernel ----
base = np.sin(2 * np.pi * 0.1 * np.arange(N // 2)) + 0.01 * np.random.randn(N // 2)   # band-limited input
up = np.zeros(N)
up[::2] = base                                     # transposed conv, stride 2: insert zeros
up_smooth = np.convolve(up, np.array([1, 2, 1]) / 2.0, mode="same")   # linear-interp kernel
fu, Su = spec(up)
fs, Ss = spec(up_smooth)
fb, Sb = spec(base)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(fx * 0.5, Sx[: len(fx)], color="0.6", label="original (rate 1)")
ax[0].plot(fn * 0.5, Sn, label="naive stride-2: tone aliased")
ax[0].plot(ff * 0.5, Sf, label="blur then stride-2")
ax[0].axvline(0.25, color="k", ls=":", lw=1)
ax[0].set_xlabel("frequency (cycles per ORIGINAL sample)")
ax[0].set_ylabel("magnitude")
ax[0].set_title("Downsampling folds 0.35 onto 0.15")
ax[0].legend(fontsize=8)

ax[1].semilogy(fb * 0.5, Sb + 1e-6, color="0.6", label="input (rate 1/2)")
ax[1].semilogy(fu, Su + 1e-6, label="zero-insertion: mirror image")
ax[1].semilogy(fs, Ss + 1e-6, label="then [1,2,1]/2 kernel")
ax[1].set_xlabel("frequency (cycles per OUTPUT sample)")
ax[1].set_ylabel("magnitude (log)")
ax[1].set_title("Upsampling creates a spectral image at 0.45")
ax[1].legend(fontsize=8)

plt.savefig("graph_DAY_131.png", dpi=120, bbox_inches="tight")

k = lambda f, S, t: S[np.argmin(np.abs(f - t))]
print("left : peak freq naive  = %.3f  (expect 0.15)" % (fn[np.argmax(Sn)] * 0.5))
print("left : peak amp  naive %.3f  blurred %.3f" % (Sn.max(), Sf.max()))
print("right: image @0.45 zero-insert %.3f  smoothed %.4f  | tone @0.05 %.3f" % (k(fu, Su, 0.45), k(fs, Ss, 0.45), k(fu, Su, 0.05)))
H = lambda w: (2 + 2 * np.cos(w)) / 4
print("blur gain at 0.35 cyc/sample: %.3f ; fold target 0.15" % H(2 * np.pi * 0.35))
