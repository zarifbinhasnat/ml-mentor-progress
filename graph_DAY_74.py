"""Quantization error: fp32 weights vs int8/int4 uniform quantization.

Synthesizes a small Gaussian weight tensor (the kind you'd find in a linear
layer), quantizes it with a simple symmetric uniform scheme at 8-bit and
4-bit, and plots (a) the value distribution with quantization levels overlaid
and (b) reconstruction error vs. bit width. This is the core tradeoff QLoRA
exploits: push the frozen base weights to 4-bit (huge memory savings) while
keeping the small LoRA adapters in higher precision (where errors matter).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 4000
weights = np.random.randn(n).astype(np.float32) * 0.5  # typical fp32 layer-weight-ish spread


def quantize_symmetric(x, bits):
    qmax = 2 ** (bits - 1) - 1  # symmetric range: e.g. bits=8 -> [-127, 127]
    scale = np.max(np.abs(x)) / qmax
    q = np.round(x / scale)
    q = np.clip(q, -qmax, qmax)
    return q * scale, scale  # dequantized values + scale factor


dequant_int8, scale8 = quantize_symmetric(weights, 8)
dequant_int4, scale4 = quantize_symmetric(weights, 4)

err_int8 = np.abs(weights - dequant_int8)
err_int4 = np.abs(weights - dequant_int4)

bits_range = np.arange(2, 9)
rmse_by_bits = []
for b in bits_range:
    dq, _ = quantize_symmetric(weights, b)
    rmse_by_bits.append(np.sqrt(np.mean((weights - dq) ** 2)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].hist(weights, bins=60, alpha=0.5, label="fp32 (original)", color="#2b6cb0", density=True)
axes[0].hist(dequant_int4, bins=60, alpha=0.5, label="int4 dequantized", color="#e53e3e", density=True)
levels4 = np.unique(np.round(np.linspace(-2 ** 3 + 1, 2 ** 3 - 1, 15)) * scale4)
for lvl in levels4:
    axes[0].axvline(lvl, color="#e53e3e", linewidth=0.4, alpha=0.4)
axes[0].set_xlabel("weight value")
axes[0].set_ylabel("density")
axes[0].set_title("fp32 weights vs. int4 quantization levels")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(bits_range, rmse_by_bits, marker="o", color="#2f855a")
axes[1].axvline(4, color="#e53e3e", linestyle="--", linewidth=1, label="QLoRA base precision (4-bit)")
axes[1].axvline(8, color="#805ad5", linestyle="--", linewidth=1, label="int8 baseline")
axes[1].set_xlabel("quantization bit width")
axes[1].set_ylabel("RMSE (fp32 vs. dequantized)")
axes[1].set_title("Reconstruction error vs. bit width")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.suptitle("Uniform symmetric quantization: fewer bits, coarser grid, more error", fontsize=11)
plt.savefig("graph_DAY_74.png", dpi=120, bbox_inches="tight")

print(f"int8 RMSE: {np.sqrt(np.mean(err_int8 ** 2)):.5f}")
print(f"int4 RMSE: {np.sqrt(np.mean(err_int4 ** 2)):.5f}")
