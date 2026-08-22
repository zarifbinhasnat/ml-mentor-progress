import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- Panel 1: CFG score extrapolation on a toy 1D bimodal setup ----------
v = 0.49  # shared variance for both toy "modes" (arbitrary small value, not from real data)


def gaussian_score(x, mu, var):
    return -(x - mu) / var


def mixture_score(x, mus, var):
    # p(x) = 0.5*N(x; mus[0], var) + 0.5*N(x; mus[1], var); score = d/dx log p(x)
    pdfs = [np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var) for mu in mus]
    p = 0.5 * pdfs[0] + 0.5 * pdfs[1]
    dpdfs = [-(x - mu) / var * pdf for mu, pdf in zip(mus, pdfs)]
    dp = 0.5 * dpdfs[0] + 0.5 * dpdfs[1]
    return dp / p


x = np.linspace(-6, 6, 400)
s_uncond = mixture_score(x, (-2.0, 2.0), v)          # unconditional: mixture of two classes
s_cond = gaussian_score(x, 2.0, v)                    # conditioned on class "+2" only

ws = [0.0, 1.0, 3.0, 7.0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(ws)))
for w, c in zip(ws, colors):
    s_cfg = (1 - w) * s_uncond + w * s_cond
    tag = " (unconditional)" if w == 0 else (" (pure conditional)" if w == 1 else "")
    ax.plot(x, np.clip(s_cfg, -20, 20), color=c, label=f"w={w:g}{tag}")
ax.axvline(2.0, color="gray", linestyle=":", linewidth=1)
ax.set_title("CFG score: extrapolating past the conditional")
ax.set_xlabel("x")
ax.set_ylabel(r"guided score $\hat{s}(x)$")
ax.legend(fontsize=8)
ax.set_ylim(-20, 20)

# ---------- Panel 2: unsharp masking is the DSP twin of the same formula ----------
n = 80
t = np.linspace(0, 1, n)
signal = (np.abs(t - 0.5) < 0.15).astype(float)  # clean rectangular pulse, plays the role of s_cond

kernel_size = 9
kernel = np.exp(-0.5 * ((np.arange(kernel_size) - kernel_size // 2) / 1.5) ** 2)
kernel /= kernel.sum()
blurred = np.convolve(signal, kernel, mode="same")  # low-pass version, plays the role of s_uncond

ax = axes[1]
ax.plot(t, signal, color="black", linewidth=1, label="original (sharp)")
ax.plot(t, blurred, color="gray", linestyle="--", label="blurred (low-pass)")
for w, c in zip([1, 3, 7], plt.cm.viridis(np.linspace(0.3, 0.9, 3))):
    sharpened = blurred + w * (signal - blurred)  # unsharp mask: identical algebraic form to CFG
    ax.plot(t, sharpened, color=c, label=f"unsharp w={w}")
ax.set_title("Unsharp masking: same formula, ringing at high w")
ax.set_xlabel("t")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_86.png", dpi=120, bbox_inches="tight")
