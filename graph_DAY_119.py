"""
Day 119 - Consolidation Phase 4: aliasing from naive strided downsampling.

A synthetic 1D signal (8 Hz + 100 Hz components, fs=256 Hz) is decimated by
factor 2 two ways: naively (stride-2, like a plain strided conv/maxpool) and
after a low-pass FIR anti-alias filter (like blur-pooling). The naive path
folds the 100 Hz component into a fake 28 Hz peak below the new Nyquist
(64 Hz); the filtered path removes it before it can fold.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Synthetic signal: two tones, fs = 256 Hz, 1 second (< 100 points after decimation) ---
fs = 256
N = 256
t = np.arange(N) / fs
f_low, f_high = 8, 100
signal = np.sin(2 * np.pi * f_low * t) + 0.8 * np.sin(2 * np.pi * f_high * t)
signal += 0.02 * np.random.randn(N)

# --- Anti-alias low-pass FIR: windowed-sinc, cutoff at the *post-decimation* Nyquist ---
fs2 = fs / 2          # sample rate after decimating by 2
cutoff = fs2 / 2      # 64 Hz -- the new Nyquist; anything above this must be removed first
fc = cutoff / fs      # normalized cutoff (fraction of original fs)
numtaps = 31
m = np.arange(numtaps) - (numtaps - 1) / 2
h = np.sinc(2 * fc * m) * np.hamming(numtaps)
h /= np.sum(h)  # unity DC gain

filtered = np.convolve(signal, h, mode="same")

# --- Decimate by 2, with and without the anti-alias filter ---
y_naive = signal[::2]
y_clean = filtered[::2]


def spectrum(x, fs_x):
    mag = np.abs(np.fft.rfft(x)) / len(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs_x)
    return freqs, mag


f0, m0 = spectrum(signal, fs)
f1, m1 = spectrum(y_naive, fs2)
f2, m2 = spectrum(y_clean, fs2)

alias_freq = abs(f_high - fs2 * round(f_high / fs2))  # where 100 Hz folds to under fs2=128
peak_idx = np.argmax(m1[(f1 > 15) & (f1 < 45)])
peak_freq = f1[(f1 > 15) & (f1 < 45)][peak_idx]
print(f"predicted alias frequency: {alias_freq:.1f} Hz, observed peak: {peak_freq:.1f} Hz")

# --- Plot: original spectrum vs naive-decimated (aliased) vs filtered-decimated (clean) ---
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

axes[0].plot(f0, m0, color="#444444")
axes[0].set_title(f"Original spectrum (fs={fs} Hz) -- true tones at {f_low} Hz and {f_high} Hz")
axes[0].set_ylabel("|X(f)|")

axes[1].plot(f1, m1, color="#C44E52")
axes[1].axvline(fs2 / 2, color="black", linestyle="--", linewidth=1, label=f"new Nyquist ({fs2/2:.0f} Hz)")
axes[1].axvline(alias_freq, color="#C44E52", linestyle=":", linewidth=1, label=f"alias of {f_high} Hz -> {alias_freq:.0f} Hz")
axes[1].set_title("Naive stride-2 decimation (no filtering) -- 100 Hz folds into a fake peak")
axes[1].set_ylabel("|X(f)|")
axes[1].legend(fontsize=8, loc="upper right")

axes[2].plot(f2, m2, color="#55A868")
axes[2].axvline(fs2 / 2, color="black", linestyle="--", linewidth=1, label=f"new Nyquist ({fs2/2:.0f} Hz)")
axes[2].set_title("Low-pass filter (blur) before decimation -- no fold, alias-free")
axes[2].set_xlabel("frequency (Hz)")
axes[2].set_ylabel("|X(f)|")
axes[2].legend(fontsize=8, loc="upper right")

fig.tight_layout()
fig.savefig("graph_DAY_119.png", dpi=120, bbox_inches="tight")
