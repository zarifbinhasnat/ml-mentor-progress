"""
Day 85 — Score-based generative models.

Toy 1D density: a bimodal mixture of two Gaussians (think: two "real image"
manifolds). We perturb it with noise at several scales sigma (exactly what a
score network is trained on — Gaussian-perturbed data at multiple noise
levels), then look at the resulting score field s_sigma(x) = d/dx log p_sigma(x).

Because convolving a Gaussian mixture with a Gaussian kernel of variance
sigma^2 just adds sigma^2 to each component's variance, p_sigma is available
in closed form -- no sampling/KDE needed.

Panel 1: the score field itself, for increasing sigma -> the two "modes" of
attraction blur together and the field gets smoother/shallower.
Panel 2: FFT magnitude of that same score field vs spatial frequency -> the
high-frequency content collapses as sigma grows. Noise-conditional score
matching is, quite literally, a family of learned low-pass filters indexed
by sigma -- which is exactly why diffusion-generated signals carry a
characteristic spectral signature (relevant to frequency-domain forensics).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- toy bimodal density: p(x) = 0.5*N(x; -2, base_var) + 0.5*N(x; 2, base_var)
mu1, mu2 = -2.0, 2.0
base_var = 0.49  # 0.7^2

x = np.linspace(-6.0, 6.0, 2000)
sigmas = [0.1, 0.5, 1.0, 2.0]


def log_p_sigma(x, sigma):
    var = base_var + sigma ** 2
    logn1 = -0.5 * np.log(2 * np.pi * var) - (x - mu1) ** 2 / (2 * var)
    logn2 = -0.5 * np.log(2 * np.pi * var) - (x - mu2) ** 2 / (2 * var)
    # log-sum-exp of the two mixture components (each weight 0.5)
    m = np.maximum(logn1, logn2)
    return m + np.log(0.5 * np.exp(logn1 - m) + 0.5 * np.exp(logn2 - m))


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

dx = x[1] - x[0]
freqs = np.fft.rfftfreq(len(x), d=dx)

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigmas)))

for sigma, c in zip(sigmas, colors):
    logp = log_p_sigma(x, sigma)
    score = np.gradient(logp, x)  # s_sigma(x) = d/dx log p_sigma(x)
    axes[0].plot(x, score, color=c, lw=2, label=f"$\\sigma$={sigma}")

    spectrum = np.abs(np.fft.rfft(score - score.mean()))
    spectrum /= spectrum.max()  # normalize each curve to its own peak
    axes[1].plot(freqs, spectrum, color=c, lw=2, label=f"$\\sigma$={sigma}")

axes[0].axhline(0, color="gray", lw=0.7)
axes[0].set_xlabel("x")
axes[0].set_ylabel(r"score $s_\sigma(x) = \nabla_x \log p_\sigma(x)$")
axes[0].set_title("Score field smooths out as noise scale grows")
axes[0].legend(fontsize=8)

axes[1].set_xlim(0, 1.5)
axes[1].set_xlabel("spatial frequency")
axes[1].set_ylabel("normalized |FFT(score)|")
axes[1].set_title("High-frequency content decays with $\\sigma$ (low-pass)")
axes[1].legend(fontsize=8)

fig.suptitle("Score-based models: noise level $\\sigma$ = low-pass cutoff on the score field")
fig.tight_layout()
fig.savefig("graph_DAY_85.png", dpi=120, bbox_inches="tight")
