"""
Day 79 — VAE & reparameterization trick.

Illustrates the effect of the KL-divergence weight (beta) on the organization
of a VAE's latent space. We don't train a real network here — instead we
hardcode plausible per-class (mu, sigma) encoder outputs for three synthetic
"digit-like" classes under three KL-weight regimes, then draw samples via
the reparameterization trick z = mu + sigma * epsilon. This isolates the
one effect we care about (how beta reshapes mu/sigma) from confounds like
training dynamics or dataset choice.

Regimes:
  beta = 0   -> KL term has no pull; encoder can shrink sigma toward 0 and
               push means far apart (behaves like a plain autoencoder,
               Day 78) -> tight, disjoint clusters with dead space between.
  beta = 1   -> standard VAE (Kingma & Welling 2013); sigma stays large
               enough that clusters overlap slightly and samples from the
               N(0,1) prior land in a region that actually decodes to
               something class-like.
  beta = 10  -> KL term dominates; means get pulled toward 0 and sigma
               pulled toward 1 (posterior collapse) -> classes fuse into
               the prior and per-class information is destroyed.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_per_class = 10  # 3 classes x 10 points x 3 panels = 90 points total, < 100 per constraint
class_names = ["class A", "class B", "class C"]
colors = ["#d62728", "#1f77b4", "#2ca02c"]

# "true" well-separated encoder means before any KL pressure is applied
base_mu = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])

regimes = [
    {"beta": 0, "mu_scale": 1.0, "sigma": 0.05},
    {"beta": 1, "mu_scale": 1.0, "sigma": 0.55},
    {"beta": 10, "mu_scale": 0.15, "sigma": 1.0},
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True, sharey=True)

theta = np.linspace(0, 2 * np.pi, 200)
for ax, regime in zip(axes, regimes):
    mu_c = base_mu * regime["mu_scale"]
    sigma = regime["sigma"]

    # unit-std reference circle: where the N(0,1) prior puts most of its mass
    ax.plot(np.cos(theta), np.sin(theta), color="gray", lw=1, ls="--", alpha=0.6)

    for c in range(3):
        mu = mu_c[c]
        eps = np.random.randn(n_per_class, 2)      # epsilon ~ N(0, I), the noise source
        z = mu + sigma * eps                        # reparameterization trick: z = mu + sigma * epsilon
        ax.scatter(z[:, 0], z[:, 1], s=28, color=colors[c], alpha=0.85,
                   label=class_names[c] if regime["beta"] == 0 else None)
        ax.scatter(*mu, marker="x", s=90, color=colors[c], linewidths=2.2)

    ax.set_title(f"β = {regime['beta']}", fontsize=12)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

axes[0].set_ylabel("$z_2$")
for ax in axes:
    ax.set_xlabel("$z_1$")

axes[0].annotate("gaps: prior samples\nhere decode to garbage",
                  xy=(0, 0), xytext=(-3.8, -3.6), fontsize=8, color="dimgray")
axes[2].annotate("classes fused —\nposterior collapse",
                  xy=(0, 0), xytext=(-3.8, -3.6), fontsize=8, color="dimgray")

fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
fig.suptitle("Reparameterized latent samples $z = \\mu + \\sigma \\odot \\epsilon$ under increasing KL weight $\\beta$",
             y=1.14, fontsize=12)

plt.savefig("graph_DAY_79.png", dpi=120, bbox_inches="tight")
