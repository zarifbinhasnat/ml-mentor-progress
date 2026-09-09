"""
Day 102 - ONNX/TorchScript export
Illustrates the SIGNAL LAB bug: a torch.jit.trace of a spectral-gating module
freezes whichever "trim last sample" branch fired for the *trace-time* input
length. Feed it a different-parity length at inference and the frozen graph
silently applies the wrong branch -- here we simulate that with plain numpy
so the script has zero external dependencies beyond numpy/matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def spectral_gate(x, threshold, force_trim):
    """force_trim mimics what a frozen (traced) graph would always do,
    versus the correct shape-dependent behavior a real forward() computes."""
    n = len(x)
    if force_trim:
        x = x[:-1]
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    mask = (mag > threshold).astype(float)
    y = np.fft.irfft(spec * mask, n=len(x))
    return y


n_samples = 51  # ODD length -- the correct branch must NOT trim
t = np.arange(n_samples)
x = (
    1.0 * np.sin(2 * np.pi * 3 * t / n_samples)
    + 0.4 * np.sin(2 * np.pi * 11 * t / n_samples)
    + 0.15 * np.random.randn(n_samples)
)
threshold = 0.5

# correct: forward() re-checks n % 2 every call -> for odd n, trim=False
correct = spectral_gate(x, threshold, force_trim=False)

# traced_frozen: graph was torch.jit.trace'd on an EVEN-length example input,
# so the "if n % 2 == 0: trim" branch got baked in as an unconditional trim
traced_frozen = spectral_gate(x, threshold, force_trim=True)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

ax = axes[0]
ax.plot(t, x, color="#888888", lw=1.2, label="input signal", alpha=0.7)
ax.plot(t, correct, color="#1f77b4", lw=2.0, label="correct forward() (n=51, no trim)")
ax.plot(
    t[:-1],
    traced_frozen,
    color="#d62728",
    lw=2.0,
    ls="--",
    label="frozen trace output (wrongly trims sample 51)",
)
ax.set_title("Traced graph freezes the wrong branch for odd-length input")
ax.set_xlabel("sample index")
ax.set_ylabel("amplitude")
ax.legend(loc="upper right", fontsize=8)

ax2 = axes[1]
freqs = np.fft.rfftfreq(n_samples)
spec_correct = np.abs(np.fft.rfft(x))
ax2.stem(freqs, spec_correct, basefmt=" ", linefmt="#1f77b4", markerfmt="o")
ax2.axhline(threshold, color="#d62728", ls="--", lw=1.2, label=f"threshold={threshold}")
ax2.set_title("Magnitude spectrum + gating threshold (computed on correct, untrimmed signal)")
ax2.set_xlabel("normalized frequency")
ax2.set_ylabel("|X(f)|")
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_102.png", dpi=120, bbox_inches="tight")
