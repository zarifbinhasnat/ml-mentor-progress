"""
Day 82 -- GAN instability & Wasserstein GAN
Shows why the vanilla GAN discriminator saturates (vanishing gradient for G)
when the real/fake distributions have little overlap, versus how a
1-Lipschitz Wasserstein critic keeps a useful gradient everywhere.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x = np.linspace(-8, 8, 400)


def gauss(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Real and fake distributions with (nearly) disjoint support -- the common
# early-training regime that breaks vanilla GANs.
pr = gauss(x, 3, 1)
pg = gauss(x, -3, 1)

# Optimal vanilla discriminator: D*(x) = pr(x) / (pr(x) + pg(x))
D_star = pr / (pr + pg + 1e-12)
grad_D = np.gradient(D_star, x)

# A valid 1-Lipschitz critic separating the two modes (slope capped at 1).
f_critic = np.clip(x, -5, 5)
grad_critic = np.gradient(f_critic, x)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x, D_star, label="D*(x)  (optimal discriminator)")
axes[0].plot(x, np.abs(grad_D), label="|dD*/dx|  (G's gradient signal)")
axes[0].axvspan(-1, 1, color="gray", alpha=0.2, label="low-density overlap zone")
axes[0].set_title("Vanilla GAN: D saturates -> G gets ~0 gradient")
axes[0].set_xlabel("x")
axes[0].legend(fontsize=8)

axes[1].plot(x, f_critic, label="f(x)  (Wasserstein critic)")
axes[1].plot(x, grad_critic, label="df/dx  (stays informative)")
axes[1].set_title("WGAN: critic gradient is nonzero everywhere")
axes[1].set_xlabel("x")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_82.png", dpi=120, bbox_inches="tight")
