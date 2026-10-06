"""
Day 127 - Full-roadmap review (Phases 1-11).

Left : backward gradient norm through 50 layers: plain ReLU net (He init, gain 0.9 / 1.0 / 1.1)
       vs a residual net y = x + F(x). Thread 1: signal propagation.
Right: spectrum of a band-limited signal upsampled x2 by zero insertion (transposed-conv style)
       vs zero insertion + low-pass interpolation. Thread 3: the spectral view.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: gradient norm vs depth ----
width, depth = 32, 50

def backward_norms(gain, residual):
    g = np.random.randn(width); g /= np.linalg.norm(g)
    norms = [1.0]
    for _ in range(depth):
        W = np.random.randn(width, width) * gain * np.sqrt(2.0 / width)
        mask = (np.random.rand(width) > 0.5).astype(float)       # ReLU derivative ~ Bernoulli(0.5)
        J = W.T * mask                                            # backward Jacobian of one layer
        if residual:
            J = np.eye(width) + 0.1 * J                           # y = x + 0.1*F(x)
        g = J @ g
        norms.append(np.linalg.norm(g))
    return np.array(norms)

# ---- right: upsampling spectra ----
n = 128
t = np.arange(n)
x = np.sin(2 * np.pi * 4 * t / n) + 0.5 * np.sin(2 * np.pi * 9 * t / n)   # low-frequency content only
up = np.zeros(2 * n); up[::2] = x                                          # zero insertion
kernel = np.array([0.5, 1.0, 0.5])                                         # linear-interp low-pass
up_lp = np.convolve(up, kernel, mode="same")

def spec(s):
    S = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    return S / S.max()

f = np.fft.rfftfreq(2 * n)                                                 # cycles/sample at the new rate

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for gain, c in [(0.9, "tab:blue"), (1.0, "tab:green"), (1.1, "tab:red")]:
    ax[0].semilogy(backward_norms(gain, False), c=c, label=f"plain, gain {gain}")
ax[0].semilogy(backward_norms(1.0, True), c="k", lw=2, label="residual (x + 0.1 F(x))")
ax[0].set_xlabel("layers back from the loss"); ax[0].set_ylabel("gradient norm")
ax[0].set_title("Thread 1: signal propagation"); ax[0].legend(fontsize=8)

ax[1].semilogy(f, spec(up) + 1e-6, c="tab:red", label="zero insertion (mirror image at 0.5-f)")
ax[1].semilogy(f, spec(up_lp) + 1e-6, c="tab:blue", label="+ low-pass interpolation")
ax[1].set_xlabel("frequency (cycles/sample)"); ax[1].set_ylabel("normalized |FFT|")
ax[1].set_title("Thread 3: upsampling leaves spectral replicas"); ax[1].legend(fontsize=8)
for k in (0, 1):
    ax[k].grid(alpha=0.3)

fig.tight_layout()
plt.savefig("graph_DAY_127.png", dpi=120, bbox_inches="tight")

print("final grad norm  plain 0.9/1.0/1.1:",
      [f"{backward_norms(g, False)[-1]:.3g}" for g in (0.9, 1.0, 1.1)], " residual:", f"{backward_norms(1.0, True)[-1]:.3g}")
hi = f > 0.25
print(f"energy above 0.25 cyc/sample: zero-insert {np.sum(spec(up)[hi]**2):.2f}, low-passed {np.sum(spec(up_lp)[hi]**2):.4f}")
