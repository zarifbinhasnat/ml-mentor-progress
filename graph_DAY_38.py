"""
Day 38 — 1x1 conv & bottlenecks
Left:  MAC-count comparison, plain 3x3 conv vs 1x1-bottleneck-3x3-1x1, across channel width.
Right: spatial-frequency magnitude response of a 1x1 kernel (flat / all-pass) vs a 3x3 box
       kernel (low-pass) -- a 1x1 conv has no spatial support, so it cannot reshape the
       spectrum of any channel; it can only linearly recombine channels.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # deterministic synthetic data

# ---------------------------------------------------------------------------
# Left panel: MACs vs channel width, plain 3x3 vs 1x1-bottleneck-3x3-1x1
# ---------------------------------------------------------------------------
H = W = 56          # a typical mid-network feature-map spatial size
reduction = 4        # standard ResNet bottleneck reduction ratio
channels = np.array([32, 64, 128, 256, 512, 1024])

macs_plain = H * W * channels * channels * 9  # k=3 -> 9 taps, C_in == C_out == C

c_mid = channels // reduction
macs_reduce = H * W * channels * c_mid * 1      # 1x1 squeeze:  C -> C/r
macs_mid = H * W * c_mid * c_mid * 9            # 3x3 on the squeezed width
macs_restore = H * W * c_mid * channels * 1     # 1x1 expand:   C/r -> C
macs_bottleneck = macs_reduce + macs_mid + macs_restore

speedup = macs_plain / macs_bottleneck

# ---------------------------------------------------------------------------
# Right panel: frequency magnitude response, 1x1 kernel vs 3x3 box kernel
# ---------------------------------------------------------------------------
N = 64  # grid size for the 2D FFT
delta = np.zeros((N, N))
delta[N // 2, N // 2] = 1.0  # a 1x1 kernel is a spatial delta function

box = np.zeros((N, N))
box[N // 2 - 1:N // 2 + 2, N // 2 - 1:N // 2 + 2] = 1.0 / 9.0  # 3x3 averaging kernel

mag_delta = np.abs(np.fft.fftshift(np.fft.fft2(delta)))
mag_box = np.abs(np.fft.fftshift(np.fft.fft2(box)))

freqs = np.fft.fftshift(np.fft.fftfreq(N))
row = N // 2
slice_delta = mag_delta[row, :]
slice_box = mag_box[row, :]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(channels, macs_plain / 1e6, "o-", color="#d62728", label="plain 3x3 conv")
ax.plot(channels, macs_bottleneck / 1e6, "o-", color="#1f77b4", label="1x1 -> 3x3 -> 1x1 bottleneck (r=4)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("channel width C (in = out)")
ax.set_ylabel("MACs (millions)")
ax.set_title("Compute cost: plain vs bottleneck conv")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
for c, s in zip(channels[::2], speedup[::2]):
    ax.annotate(f"{s:.1f}x", (c, macs_bottleneck[channels == c][0] / 1e6),
                textcoords="offset points", xytext=(0, -14), fontsize=7.5, color="#1f77b4")

ax = axes[1]
ax.plot(freqs, slice_delta, color="#1f77b4", label="1x1 kernel (delta) -- flat / all-pass")
ax.plot(freqs, slice_box, color="#d62728", label="3x3 box kernel -- low-pass")
ax.set_xlabel("normalized spatial frequency")
ax.set_ylabel("|H(f)| (center row slice)")
ax.set_title("Frequency response: 1x1 vs 3x3 kernel")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_38.png", dpi=120, bbox_inches="tight")
