"""
Day 128 - Consolidation: the bottleneck / low-rank thread.

Left : relative reconstruction error vs rank k for a smooth (low-rank-ish) matrix and a pure-noise matrix.
       Smooth structure compresses; noise does not (why LoRA / autoencoders / latent diffusion work).
Right: a 1-D signal compressed two ways with the same coefficient budget:
       keep the k largest Fourier coefficients vs keep the first k samples. Frequency-domain bottleneck.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: SVD truncation error ----
n = 64
u = np.linspace(0, 1, n)
smooth = (np.outer(np.sin(2 * np.pi * u), np.cos(3 * np.pi * u))
          + 0.5 * np.outer(u, u ** 2)
          + 0.02 * np.random.randn(n, n))
noise = np.random.randn(n, n)

def rel_err(M):
    U, s, Vt = np.linalg.svd(M)
    errs = []
    for k in range(1, 33):
        Mk = (U[:, :k] * s[:k]) @ Vt[:k]
        errs.append(np.linalg.norm(M - Mk) / np.linalg.norm(M))
    return np.array(errs)

# ---- right: Fourier top-k vs first-k samples ----
m = 128
t = np.arange(m)
x = np.sin(2 * np.pi * 3 * t / m) + 0.6 * np.sin(2 * np.pi * 7 * t / m) + 0.1 * np.random.randn(m)
X = np.fft.rfft(x)
k = 6
keep = np.argsort(np.abs(X))[-k:]
Xk = np.zeros_like(X); Xk[keep] = X[keep]
x_fft = np.fft.irfft(Xk, n=m)
x_first = np.concatenate([x[:k], np.zeros(m - k)])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(range(1, 33), rel_err(smooth), "o-", label="smooth structure + small noise")
ax[0].semilogy(range(1, 33), rel_err(noise), "s-", label="pure noise")
ax[0].set_xlabel("rank k"); ax[0].set_ylabel("relative error")
ax[0].set_title("Bottleneck: error vs rank"); ax[0].legend(fontsize=8)

ax[1].plot(t, x, c="0.7", lw=2, label="original")
ax[1].plot(t, x_fft, c="tab:green", label=f"top-{k} Fourier coeffs")
ax[1].plot(t, x_first, c="tab:red", ls="--", label=f"first {k} samples")
ax[1].set_xlabel("sample"); ax[1].set_title("Same budget, different basis")
ax[1].legend(fontsize=8)

plt.savefig("graph_DAY_128.png", dpi=120, bbox_inches="tight")
