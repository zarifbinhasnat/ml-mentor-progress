"""
DenseNet concatenation vs. ResNet-style addition, viewed as feature fusion.

Two synthetic "feature-map" signals share a component at 20 Hz but with
OPPOSITE PHASE. Summing them (the ResNet residual-fusion move) destructively
cancels that shared component -- it's just gone from the output. Keeping
the two signals as separate channels (the DenseNet concatenation move)
never combines amplitudes at all, so nothing is lost: both channels still
carry the full 20 Hz component intact for any later layer to use.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fs = 500                     # sample rate (Hz)
t = np.arange(0, 1, 1 / fs)  # 1 second of signal

# Channel 1 ("layer A" feature map): 10 Hz unique + 20 Hz shared component
x1 = np.sin(2 * np.pi * 10 * t) + 0.8 * np.sin(2 * np.pi * 20 * t)
# Channel 2 ("layer B" feature map): 35 Hz unique + 20 Hz shared, OPPOSITE phase
x2 = 0.6 * np.sin(2 * np.pi * 35 * t) + 0.8 * np.sin(2 * np.pi * 20 * t + np.pi)

x_sum = x1 + x2  # ResNet-style additive fusion -- 20 Hz components cancel exactly

freqs = np.fft.rfftfreq(len(t), d=1 / fs)
X1 = np.abs(np.fft.rfft(x1)) / len(t)
X2 = np.abs(np.fft.rfft(x2)) / len(t)
Xsum = np.abs(np.fft.rfft(x_sum)) / len(t)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0, 0].stem(freqs, X1 * 2, basefmt=" ")
axes[0, 0].set_title("Channel A spectrum (10 Hz + 20 Hz)")
axes[0, 0].set_xlim(0, 60)
axes[0, 0].set_ylabel("|X(f)|")

axes[0, 1].stem(freqs, X2 * 2, basefmt=" ", linefmt="C1-", markerfmt="C1o")
axes[0, 1].set_title("Channel B spectrum (35 Hz + 20 Hz, phase-shifted)")
axes[0, 1].set_xlim(0, 60)

axes[1, 0].stem(freqs, Xsum * 2, basefmt=" ", linefmt="C3-", markerfmt="C3o")
axes[1, 0].set_title("Addition (ResNet-style fusion): 20 Hz cancelled")
axes[1, 0].set_xlim(0, 60)
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("|X(f)|")
axes[1, 0].axvline(20, color="gray", linestyle=":", linewidth=1)
axes[1, 0].annotate("gone!", xy=(20, 0.02), xytext=(28, 0.35),
                    arrowprops=dict(arrowstyle="->", color="gray"))

# "Concatenation" view: both channels' spectra stacked, nothing combined/lost
stacked = np.vstack([X1 * 2, X2 * 2])
im = axes[1, 1].imshow(stacked, aspect="auto", cmap="magma",
                        extent=[freqs[0], 60, 2, 0])
axes[1, 1].set_title("Concatenation (DenseNet-style): both channels intact")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_yticks([0.5, 1.5])
axes[1, 1].set_yticklabels(["Channel A", "Channel B"])
axes[1, 1].axvline(20, color="cyan", linestyle=":", linewidth=1)
fig.colorbar(im, ax=axes[1, 1], label="magnitude")

fig.suptitle("Feature fusion: addition can cancel shared spectral content; "
             "concatenation never combines amplitudes, so nothing cancels",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("graph_DAY_43.png", dpi=120, bbox_inches="tight")
