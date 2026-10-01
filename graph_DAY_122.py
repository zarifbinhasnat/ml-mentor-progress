"""
Day 122 - Consolidation Phase 7: scaling laws and quantization are both 'power-law / log-scale' stories.

Left: synthetic loss-vs-parameters points following L(N) = E + A * N^-alpha (with small noise).
      On log-log axes the reducible loss (L - E) is a straight line whose slope is -alpha.
Right: symmetric uniform quantization of Gaussian weights. Each extra bit roughly quarters
      the MSE (about 6 dB per bit), which is why int8 is nearly free and int4 needs care (QLoRA).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: scaling law ----
E, A, alpha = 1.7, 40.0, 0.34
N = np.logspace(6, 10, 12)                          # 1M .. 10B parameters
loss = E + A * N ** (-alpha)
loss_noisy = loss * (1 + 0.004 * np.random.randn(N.size))
slope, intercept = np.polyfit(np.log10(N), np.log10(loss_noisy - E), 1)
print("fitted alpha:", round(-slope, 3), "(true", alpha, ")")

# ---- right: quantization error ----
w = np.random.randn(100)                            # tiny synthetic weight vector
bits = np.arange(2, 9)
mse = []
for b in bits:
    levels = 2 ** (b - 1) - 1
    scale = np.abs(w).max() / levels
    wq = np.clip(np.round(w / scale), -levels, levels) * scale
    mse.append(np.mean((w - wq) ** 2))
mse = np.array(mse)
print("MSE ratio per extra bit:", np.round(mse[:-1] / mse[1:], 2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].loglog(N, loss_noisy - E, "o", label="synthetic runs (L - E)")
axes[0].loglog(N, 10 ** intercept * N ** slope, "-", label=f"fit: slope = {slope:.2f}")
axes[0].set_xlabel("parameters N")
axes[0].set_ylabel("reducible loss  L - E")
axes[0].set_title("Scaling law: straight line on log-log")
axes[0].legend()

axes[1].semilogy(bits, mse, "s-")
axes[1].set_xlabel("bits per weight")
axes[1].set_ylabel("quantization MSE")
axes[1].set_title("Each extra bit ~ 4x lower error")
axes[1].grid(True, which="both", alpha=0.3)

plt.savefig("graph_DAY_122.png", dpi=120, bbox_inches="tight")
