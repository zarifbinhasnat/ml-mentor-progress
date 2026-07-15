"""
Day 41 -- Inception / GoogLeNet
Left:  a synthetic two-tone signal run through the same three "branch" kernel
       sizes an Inception module uses (1, 3, 5 taps) -- each branch is just a
       box filter of a different width, standing in for a learned conv.
Right: their frequency responses -- wider taps = lower cutoff. Concatenating
       the three branch outputs channel-wise is literally handing the network
       a pre-built multi-resolution filter bank instead of forcing one kernel
       size to cover every frequency band alone.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # deterministic synthetic data

# ---------------------------------------------------------------------------
# Synthetic two-tone signal: a slow trend + a fast ripple + a little noise
# ---------------------------------------------------------------------------
n = 96
t = np.arange(n)
low_freq = np.sin(2 * np.pi * 0.02 * t)          # slow component
high_freq = 0.5 * np.sin(2 * np.pi * 0.25 * t)   # fast component
signal = low_freq + high_freq + 0.1 * np.random.randn(n)

def box_filter(x, k):
    if k == 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(xp, kernel, mode="valid")[: len(x)]

branch1 = box_filter(signal, 1)   # 1x1 "branch" -- passes everything through
branch3 = box_filter(signal, 3)   # 3-tap branch -- mild smoothing
branch5 = box_filter(signal, 5)   # 5-tap branch -- stronger smoothing

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(t, signal, color="#7f7f7f", alpha=0.6, label="input signal")
ax.plot(t, branch1, color="#2ca02c", label="1x1 branch (identity)")
ax.plot(t, branch3, color="#1f77b4", label="3-tap branch")
ax.plot(t, branch5, color="#d62728", label="5-tap branch")
ax.set_xlabel("position")
ax.set_ylabel("amplitude")
ax.set_title("Three Inception branches = three views of one signal")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ---------------------------------------------------------------------------
# Frequency response of each branch's box kernel
# ---------------------------------------------------------------------------
n_fft = 512
freqs = np.fft.rfftfreq(n_fft, d=1.0)

def mag_db(k):
    kernel = np.ones(k) / k if k > 1 else np.array([1.0])
    H = np.fft.rfft(kernel, n=n_fft)
    return 20 * np.log10(np.maximum(np.abs(H), 1e-6))

ax = axes[1]
ax.plot(freqs, mag_db(1), color="#2ca02c", label="1x1 branch")
ax.plot(freqs, mag_db(3), color="#1f77b4", label="3-tap branch")
ax.plot(freqs, mag_db(5), color="#d62728", label="5-tap branch")
ax.set_xlabel("normalized frequency")
ax.set_ylabel("magnitude (dB)")
ax.set_title("Wider branch = lower cutoff = a de-facto filter bank")
ax.set_ylim(-40, 5)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_41.png", dpi=120, bbox_inches="tight")
