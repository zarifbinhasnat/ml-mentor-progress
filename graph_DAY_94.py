"""
Day 94 -- U-Net & skip connections.

Demonstrates, on a 1D synthetic 'signal' (stand-in for an image row),
why a plain encoder-decoder bottleneck acts as a low-pass filter and
loses high-frequency detail, while a U-Net-style skip connection
restores it.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Build a synthetic 1D signal: low-frequency base + a sharp edge (high-freq content) ---
n = 128
x = np.linspace(0, 1, n)
low_freq = np.sin(2 * np.pi * 2 * x)                      # smooth, low-frequency structure
edge = np.where((x > 0.45) & (x < 0.55), 1.0, 0.0)         # sharp edge -> broadband high-freq energy
signal = low_freq + edge

def avg_pool(v, factor):
    return v.reshape(-1, factor).mean(axis=1)

def upsample_linear(v, target_len):
    src_x = np.linspace(0, 1, len(v))
    tgt_x = np.linspace(0, 1, target_len)
    return np.interp(tgt_x, src_x, v)

# --- "Encoder": two downsampling stages (bottleneck), like conv+pool blocks ---
enc1 = avg_pool(signal, 4)   # skip connection tapped here (n/4 resolution)
bottleneck = avg_pool(enc1, 4)  # n/16 resolution, most detail destroyed

# --- Decoder WITHOUT skip connections: just upsample straight from the bottleneck ---
dec_no_skip = upsample_linear(bottleneck, n)

# --- Decoder WITH a U-Net skip connection: upsample bottleneck to enc1's resolution,
#     then fuse (add) the high-frequency detail carried by the skip tensor, then upsample to n ---
up_to_enc1 = upsample_linear(bottleneck, len(enc1))
high_freq_detail = enc1 - up_to_enc1          # what the skip connection actually contributes
fused = up_to_enc1 + high_freq_detail         # = enc1, recovered losslessly at that resolution
dec_with_skip = upsample_linear(fused, n)

# --- Plot: reconstructions + frequency-domain view of what got lost/recovered ---
fig, axes = plt.subplots(2, 1, figsize=(8, 7))

ax = axes[0]
ax.plot(x, signal, color="black", lw=2, label="original signal")
ax.plot(x, dec_no_skip, color="crimson", lw=2, ls="--", label="decoder, no skip (blurred edge)")
ax.plot(x, dec_with_skip, color="royalblue", lw=2, ls=":", label="decoder + skip (edge restored)")
ax.set_title("U-Net skip connections restore high-frequency detail lost in the bottleneck")
ax.set_xlabel("position")
ax.set_ylabel("amplitude")
ax.legend(loc="upper right", fontsize=9)

# Magnitude spectrum: the bottleneck-only path is a low-pass filter.
freqs = np.fft.rfftfreq(n, d=x[1] - x[0])
spec_orig = np.abs(np.fft.rfft(signal))
spec_noskip = np.abs(np.fft.rfft(dec_no_skip))
spec_skip = np.abs(np.fft.rfft(dec_with_skip))

ax2 = axes[1]
ax2.plot(freqs, spec_orig, color="black", lw=2, label="original |FFT|")
ax2.plot(freqs, spec_noskip, color="crimson", lw=2, ls="--", label="no skip |FFT| (high freqs killed)")
ax2.plot(freqs, spec_skip, color="royalblue", lw=2, ls=":", label="with skip |FFT| (high freqs recovered)")
ax2.set_title("Bottleneck-only encoding = low-pass filter; skip connections put the high band back")
ax2.set_xlabel("frequency")
ax2.set_ylabel("magnitude")
ax2.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("graph_DAY_94.png", dpi=120, bbox_inches="tight")
