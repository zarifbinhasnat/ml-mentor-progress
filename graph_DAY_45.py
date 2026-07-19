"""
Day 45 — Dilated / Atrous Convolution
Shows (1) how dilation grows the receptive field for free by spacing out
taps, and (2) the frequency-domain cost of that trick: the "gridding"
artifact, where the kernel's frequency response grows spurious periodic
copies (comb-like sidelobes) once holes are punched into it.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# base smoothing kernel (odd-length, unit sum)
base_kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
base_kernel /= base_kernel.sum()
k = len(base_kernel)

dilations = [1, 2, 4]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# --- Top: tap positions (receptive field growth) ---
ax = axes[0]
for i, (d, c) in enumerate(zip(dilations, colors)):
    span = (k - 1) * d + 1  # receptive field width for this dilation
    positions = np.arange(0, span, d)
    y = np.full_like(positions, fill_value=len(dilations) - i, dtype=float)
    ax.stem(positions - span // 2, y, linefmt=c, markerfmt="o", basefmt=" ")
    ax.text(span // 2 + 1, len(dilations) - i, f"dilation={d}, RF={span}", color=c, va="center")

ax.set_yticks([])
ax.set_xlabel("input position (relative to center)")
ax.set_title("Same 7-tap kernel, growing receptive field via dilation")

# --- Bottom: frequency response (gridding artifact) ---
ax2 = axes[1]
N = 512  # FFT length for fine frequency resolution
freqs = np.fft.rfftfreq(N, d=1.0)

for d, c in zip(dilations, colors):
    # insert (d-1) zeros between taps -- this IS what dilation does to the kernel
    dilated_kernel = np.zeros((k - 1) * d + 1)
    dilated_kernel[::d] = base_kernel
    padded = np.zeros(N)
    padded[: len(dilated_kernel)] = dilated_kernel
    spectrum = np.abs(np.fft.rfft(padded))
    ax2.plot(freqs, spectrum, color=c, label=f"dilation={d}")

ax2.set_xlabel("normalized frequency (cycles/sample)")
ax2.set_ylabel("|H(f)|")
ax2.set_title("Gridding artifact: dilation replicates the spectrum (comb sidelobes)")
ax2.legend()
ax2.set_xlim(0, 0.5)

plt.tight_layout()
plt.savefig("graph_DAY_45.png", dpi=120, bbox_inches="tight")
