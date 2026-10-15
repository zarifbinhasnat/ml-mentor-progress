"""
Day 136 - Consolidation: the masking thread (dropout -> causal mask -> BERT MLM -> MAE -> diffusion noising).
Masking = multiplying by a 0/1 sequence m[n]. Multiplication in time = convolution in frequency:
    FFT(x * m) = FFT(x) (*) FFT(m)
Left : a clean two-tone signal and three masks (random 50%, one contiguous block, causal/prefix).
Right: magnitude spectra of the masked signals. Random masks smear energy into a broadband floor;
       a block mask leaves sinc-like sidelobes around the tones. Synthetic, seed 42, 64 samples.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 64
n = np.arange(N)
x = np.sin(2 * np.pi * 5 * n / N) + 0.5 * np.sin(2 * np.pi * 12 * n / N)   # tones at bins 5 and 12

masks = {
    "random 50% (MAE / dropout style)": (np.random.rand(N) > 0.5).astype(float),
    "one block of 32 (inpainting / span mask)": np.r_[np.ones(16), np.zeros(32), np.ones(16)],
    "causal prefix (first 32 visible)": np.r_[np.ones(32), np.zeros(32)],
}
colors = ["tab:red", "tab:blue", "tab:green"]


def spec(sig):
    return np.abs(np.fft.rfft(sig)) / N


S0 = spec(x)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
ax1.plot(n, x, "k", lw=1.5, label="clean x[n]")
for (name, m), c, off in zip(masks.items(), colors, [-2.6, -3.8, -5.0]):
    ax1.step(n, m * 0.8 + off, c=c, lw=1.5, where="mid", label=name)
ax1.set_xlabel("sample n")
ax1.set_title("Signal and three masks m[n]")
ax1.legend(fontsize=7, loc="upper right")

k = np.arange(len(S0))
ax2.stem(k, S0, linefmt="k-", markerfmt="ko", basefmt=" ", label="clean")
for (name, m), c in zip(masks.items(), colors):
    S = spec(x * m)
    ax2.plot(k, S, c=c, lw=1.8, label=name.split(" (")[0])
    tone = S[[5, 12]].sum()
    floor = np.delete(S, [4, 5, 6, 11, 12, 13]).mean()
    print(f"{name:42s} keep={m.mean():.2f}  tone-bin mag={tone:.3f}  off-tone floor={floor:.4f}")
print(f"clean: tone-bin mag={S0[[5, 12]].sum():.3f}  off-tone floor={np.delete(S0, [4, 5, 6, 11, 12, 13]).mean():.4f}")
ax2.set_xlabel("frequency bin k")
ax2.set_ylabel("|FFT| / N")
ax2.set_title("FFT(x*m) = FFT(x) convolved with FFT(m)")
ax2.legend(fontsize=8)

plt.savefig("graph_DAY_136.png", dpi=120, bbox_inches="tight")
