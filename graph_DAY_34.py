"""
Day 34 — Kernels, stride, padding: strided convolution as filter + decimate.
Shows that stride-2 "convolution" without adequate lowpass content in the
kernel lets a high-frequency component alias into the passband, while a
simple blurring kernel suppresses it first. Tiny synthetic 1D signal,
no downloads, seeded for reproducibility.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N = 64                                     # signal length: small, synthetic, hardcoded
n = np.arange(N)

f_low, f_high = 3, 20                      # cycles over the whole signal
signal = np.sin(2 * np.pi * f_low * n / N) + 0.8 * np.sin(2 * np.pi * f_high * n / N)

lowpass_kernel = np.array([1.0, 2.0, 1.0]) / 4.0   # tiny binomial blur -> acts as a lowpass "conv kernel"
allpass_kernel = np.array([0.0, 1.0, 0.0])         # identity kernel -> naive strided conv, no filtering at all


def conv_same(x, k):
    pad = len(k) // 2
    xp = np.pad(x, pad, mode="wrap")        # circular padding keeps this a clean circular convolution
    return np.convolve(xp, k, mode="valid")


def spectrum(x):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0) * len(x)   # convert back to "cycles per signal" for readability
    return freqs, np.abs(X)


stride = 2
filtered_low = conv_same(signal, lowpass_kernel)
filtered_all = conv_same(signal, allpass_kernel)
dec_low = filtered_low[::stride]           # stride-2 conv WITH a lowpass kernel first
dec_all = filtered_all[::stride]           # stride-2 conv with NO real filtering (naive strided conv)

fig, axes = plt.subplots(3, 1, figsize=(7, 8.5))

for ax, (title, sig) in zip(
    axes,
    [
        ("Original signal spectrum (N=64) — two tones: f=3 and f=20", signal),
        ("Lowpass kernel + stride-2 (high tone suppressed, no fold-back)", dec_low),
        ("All-pass kernel + stride-2 (f=20 ALIASES back near f=12)", dec_all),
    ],
):
    freqs, mags = spectrum(sig)
    ax.stem(freqs, mags)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("cycles / signal")
    ax.set_ylabel("|X(f)|")
    ax.set_xlim(0, 32)

plt.tight_layout()
plt.savefig("graph_DAY_34.png", dpi=120, bbox_inches="tight")
print("saved graph_DAY_34.png")
