"""
Day 125 - Consolidation Phase 10: two trade-offs that govern training and serving.

Left : batching. Step latency = t0 (fixed weight-load cost) + c*b (per-sequence compute).
       Throughput b/latency saturates while latency keeps climbing -> pick b at the knee.
Right: reduced precision is additive noise. Quantize a two-tone signal to 8 / 4 bits (uniform)
       and look at the noise floor in the spectrum: ~6.02 dB lower per extra bit.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: throughput vs latency ----
b = np.arange(1, 65)
t0, c = 20.0, 0.5                     # ms: fixed cost, per-sequence cost
lat = t0 + c * b
thr = b / lat * 1000                  # sequences / second

# ---- right: quantization noise floor ----
n = 1024
t = np.arange(n)
x = 0.6 * np.sin(2 * np.pi * 50 * t / n) + 0.3 * np.sin(2 * np.pi * 180 * t / n)
x = x + 0.01 * np.random.randn(n)
win = np.hanning(n)

def quant(v, bits):
    q = 2 ** (bits - 1) - 1
    s = np.max(np.abs(v)) / q         # symmetric per-tensor scale
    return np.round(v / s) * s

def psd_db(v):
    return 20 * np.log10(np.abs(np.fft.rfft(v * win)) / n + 1e-12)

f = np.arange(n // 2 + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(b, thr, color="tab:blue", label="throughput (seq/s)")
ax.set_xlabel("batch size")
ax.set_ylabel("throughput (seq/s)", color="tab:blue")
ax2 = ax.twinx()
ax2.plot(b, lat, color="tab:red", label="step latency (ms)")
ax2.set_ylabel("step latency (ms)", color="tab:red")
ax.axvline(t0 / c, color="gray", ls="--", lw=0.8)
ax.set_title("Batching: throughput saturates, latency does not")

ax = axes[1]
ax.plot(f, psd_db(x), color="black", lw=1.2, label="fp32 signal")
for bits, col in [(8, "tab:green"), (4, "tab:red")]:
    err = quant(x, bits) - x
    ax.plot(f, psd_db(err), color=col, lw=0.9, label=f"{bits}-bit quantization error")
ax.set_xlim(0, 400)
ax.set_xlabel("frequency bin")
ax.set_ylabel("magnitude (dB)")
ax.set_title("Quantization = noise floor, ~6 dB per bit")
ax.legend(fontsize=8)

plt.savefig("graph_DAY_125.png", dpi=120, bbox_inches="tight")
