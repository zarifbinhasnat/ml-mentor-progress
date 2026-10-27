"""
Day 146 - Consolidation: the efficiency thread (mixed precision, KV cache, quantization, batching).
Left : KV-cache memory vs context length for fp16 / int8 / int4 (toy 32-layer, 32-head, d_head=128 model).
Right: uniform b-bit quantization noise is ~white, real signals are ~1/f, so per-frequency SNR dies
       at high frequency first -- fewer bits push the usable-band edge down.
Synthetic, deterministic, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: KV cache bytes = 2 (K and V) * layers * heads * d_head * seq * bytes_per_elem
L, H, D = 32, 32, 128
seq = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768])
def kv_gb(bytes_per): return 2 * L * H * D * seq * bytes_per / 1e9
curves = {"fp16 (2 B)": 2.0, "int8 (1 B)": 1.0, "int4 (0.5 B)": 0.5}

# ---- right: quantization noise vs 1/f signal
N = 256
f = np.fft.rfftfreq(N)
amp = 1.0 / np.maximum(np.arange(N // 2 + 1), 1)
sig_P, qn = np.zeros(N // 2 + 1), {}
B = 40
for _ in range(B):
    X = amp * np.exp(2j * np.pi * np.random.rand(N // 2 + 1)); X[0] = 0; X[-1] = X[-1].real
    x = np.fft.irfft(X, n=N); x = x / np.abs(x).max()          # scale into [-1, 1]
    sig_P += np.abs(np.fft.rfft(x)) ** 2 / B
    for bits in (8, 4, 2):
        step = 2.0 / (2 ** bits)                                # uniform quantizer on [-1, 1]
        e = np.round(x / step) * step - x
        qn.setdefault(bits, np.zeros_like(sig_P))
        qn[bits] += np.abs(np.fft.rfft(e)) ** 2 / B

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, b in curves.items():
    ax[0].plot(seq, kv_gb(b), marker="o", label=name)
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
ax[0].set_xlabel("context length (tokens)"); ax[0].set_ylabel("KV cache per sequence (GB)")
ax[0].set_title("KV cache grows linearly with context"); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].semilogy(f[1:], sig_P[1:], "k", lw=2, label="signal (~1/f)")
for bits, c in zip((8, 4, 2), ("tab:blue", "tab:orange", "tab:red")):
    ax[1].semilogy(f[1:], qn[bits][1:], color=c, label=f"{bits}-bit quant. noise")
ax[1].set_xlabel("frequency (cycles/sample)"); ax[1].set_ylabel("power")
ax[1].set_title("Flat noise floor swallows high frequencies first"); ax[1].legend(); ax[1].grid(alpha=0.3)

print("KV GB at 32768 tokens:", {k: round(float(kv_gb(v)[-1]), 1) for k, v in curves.items()})
for bits in (8, 4, 2):
    snr = sig_P[1:] / qn[bits][1:]
    ok = f[1:][snr >= 1]
    print(f"{bits}-bit: highest freq with SNR>=1 = {ok.max() if ok.size else 0:.3f}; "
          f"SNR@f=0.4 = {10*np.log10(np.interp(0.4, f[1:], snr)):.1f} dB")
plt.savefig("graph_DAY_146.png", dpi=120, bbox_inches="tight")
