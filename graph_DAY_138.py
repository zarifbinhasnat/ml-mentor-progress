"""
Day 138 - Consolidation: the discrete-made-differentiable thread
(argmax -> softmax -> straight-through quantization -> reparameterization -> top-k routing -> score function).
Left : a b-bit uniform quantizer. True derivative is 0 almost everywhere, so the straight-through estimator (STE)
       pretends d q(x)/dx = 1 inside the clip range.
Right: PSD of the quantization error of a sine at 4 and 8 bits. The error is ~white with power delta^2/12,
       so each extra bit lowers the floor by ~6 dB.
Synthetic, seed 42, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def quantize(x, bits, lim=1.0):
    delta = 2 * lim / (2 ** bits)
    return np.clip(np.round(x / delta) * delta, -lim, lim - delta), delta


# ---- Left: staircase and its STE gradient ---------------------------------------------
x = np.linspace(-1.2, 1.2, 97)
q3, d3 = quantize(x, 3)
true_grad = np.gradient(q3, x)
ste_grad = ((x >= -1.0) & (x <= 1.0)).astype(float)
print(f"3-bit quantizer: step={d3:.3f}, fraction of points with true grad == 0: {np.mean(true_grad == 0):.2f}, "
      f"STE grad == 1 on [-1,1]")

# ---- Right: quantization-noise PSD ------------------------------------------------------
n = 4096
t = np.arange(n)
sig = 0.9 * np.sin(2 * np.pi * 0.0517 * t) + 0.05 * np.random.randn(n)
win = np.hanning(n)
res = {}
for bits in (4, 8):
    q, delta = quantize(sig, bits)
    err = q - sig
    psd = np.abs(np.fft.rfft(err * win)) ** 2 / (win ** 2).sum()
    res[bits] = (psd, err.var(), delta ** 2 / 12)
    print(f"{bits}-bit: error var={err.var():.2e}, theory delta^2/12={delta ** 2 / 12:.2e}, "
          f"mean PSD floor={10 * np.log10(psd[5:].mean()):.1f} dB")
drop = 10 * np.log10(res[4][0][5:].mean() / res[8][0][5:].mean())
print(f"floor drop 4->8 bits: {drop:.1f} dB (theory 6.02*4 = {6.02 * 4:.1f} dB)")

f = np.fft.rfftfreq(n)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
ax1.plot(x, x, "k--", lw=1, label="identity (what STE pretends)")
ax1.step(x, q3, "tab:blue", lw=2, where="mid", label="3-bit quantizer q(x)")
ax1.plot(x, true_grad, "tab:red", lw=1.5, label="true dq/dx: 0, plus delta spikes at jumps (clipped)")
ax1.plot(x, ste_grad, "tab:green", lw=1.5, label="STE gradient (1 inside clip)")
ax1.set_ylim(-1.5, 1.6); ax1.set_xlabel("x"); ax1.set_ylabel("value")
ax1.set_title("Straight-through estimator")
ax1.legend(fontsize=8)

for bits, c in ((4, "tab:red"), (8, "tab:green")):
    psd, var, th = res[bits]
    ax2.semilogy(f, psd + 1e-18, c, lw=1, label=f"{bits}-bit error PSD")
    ax2.axhline(th, color=c, ls="--", lw=1)
ax2.set_xlabel("normalised frequency"); ax2.set_ylabel("PSD (dashed: delta^2/12 level)")
ax2.set_title("Quantization noise is ~white; 6 dB per bit")
ax2.legend(fontsize=8)

plt.savefig("graph_DAY_138.png", dpi=120, bbox_inches="tight")
