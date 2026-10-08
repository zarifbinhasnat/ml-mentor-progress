"""
Day 129 - Consolidation: the softmax / exponential-normalisation thread.

Left : softmax of fixed logits at several temperatures T (T->0 is argmax, T->inf is uniform),
       plus the entropy of each distribution.
Right: soft-argmax over an FFT magnitude spectrum. A tone at 10.4 bins falls between bins; hard argmax
       says 10, soft-argmax (softmax-weighted mean of bin index) recovers a sub-bin estimate.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    z = z - z.max()                      # max-subtraction: the stability trick
    e = np.exp(z)
    return e / e.sum()

# ---- left: temperature sweep ----
logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
temps = [0.25, 1.0, 4.0]
ent = {}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
w = 0.25
for i, T in enumerate(temps):
    p = softmax(logits, T)
    ent[T] = -(p * np.log(p + 1e-12)).sum()
    ax[0].bar(np.arange(5) + (i - 1) * w, p, w, label=f"T={T}  H={ent[T]:.2f}")
ax[0].set_xlabel("class index")
ax[0].set_ylabel("probability")
ax[0].set_title("Temperature: sharp (T<1) to flat (T>1)")
ax[0].legend()

# ---- right: soft-argmax on a spectrum ----
N = 64
f0 = 10.4
n = np.arange(N)
x = np.sin(2 * np.pi * f0 * n / N) + 0.05 * np.random.randn(N)
mag = np.abs(np.fft.rfft(x * np.hanning(N)))
bins = np.arange(len(mag))
hard = int(np.argmax(mag))
# normalise so the temperature is scale-free, then weight the bins
z = mag / mag.max()
soft = {}
for T in (0.05, 0.1, 0.3):
    p = softmax(z, T)
    soft[T] = float((p * bins).sum())
p_best = softmax(z, 0.05)
ax[1].stem(bins[:24], mag[:24] / mag.max(), linefmt="C0-", markerfmt="C0o", basefmt=" ", label="|X[k]| (normalised)")
ax[1].axvline(f0, color="k", ls="--", label=f"true f0 = {f0}")
ax[1].axvline(hard, color="C3", ls=":", label=f"argmax = {hard}")
ax[1].axvline(soft[0.1], color="C2", ls="-.", label=f"soft-argmax T=0.1 = {soft[0.1]:.2f}")
ax[1].set_xlabel("FFT bin k")
ax[1].set_title("Soft-argmax gives a sub-bin frequency estimate")
ax[1].legend(fontsize=8)

plt.savefig("graph_DAY_129.png", dpi=120, bbox_inches="tight")
print("entropy:", {k: round(v, 3) for k, v in ent.items()})
print("hard argmax:", hard, "soft:", {k: round(v, 3) for k, v in soft.items()})
