"""
Day 107 -- Actor-critic & PPO
Two-panel figure:
  (left)  PPO clipped surrogate objective vs. probability ratio r_t(theta),
          for positive and negative advantage.
  (right) GAE(lambda) credit-assignment weights (gamma*lambda)^l as an
          exponential-decay impulse response -- the "IIR low-pass filter"
          view of advantage estimation.
No downloads, no internet -- pure numpy/matplotlib on hardcoded arrays.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Panel 1: PPO clipped surrogate objective ----
eps = 0.2
r = np.linspace(0.0, 2.0, 400)

def clipped_obj(r, A, eps):
    unclipped = r * A
    clipped = np.clip(r, 1 - eps, 1 + eps) * A
    return np.minimum(unclipped, clipped)

L_pos = clipped_obj(r, A=1.0, eps=eps)   # positive advantage
L_neg = clipped_obj(r, A=-1.0, eps=eps)  # negative advantage

# ---- Panel 2: GAE(lambda) exponential decay weights ----
gamma = 0.99
lambdas = [0.0, 0.9, 0.95, 1.0]
lags = np.arange(0, 40)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel 1
ax = axes[0]
ax.plot(r, L_pos, color="#1f77b4", lw=2.2, label=r"$\hat{A}_t > 0$")
ax.plot(r, L_neg, color="#d62728", lw=2.2, label=r"$\hat{A}_t < 0$")
ax.axvline(1 - eps, color="gray", ls="--", lw=1)
ax.axvline(1 + eps, color="gray", ls="--", lw=1)
ax.axvline(1.0, color="black", ls=":", lw=1)
ax.set_xlabel(r"probability ratio $r_t(\theta) = \pi_\theta(a|s) / \pi_{\theta_{old}}(a|s)$")
ax.set_ylabel(r"$L^{CLIP}$")
ax.set_title(f"PPO clipped objective (eps={eps})")
ax.legend(frameon=False, loc="lower center")
ax.grid(alpha=0.25)

# Panel 2
ax2 = axes[1]
for lam in lambdas:
    weights = (gamma * lam) ** lags
    weights = weights / weights.sum()  # normalize to compare "window shape"
    ax2.plot(lags, weights, marker="o", ms=3, lw=1.6, label=fr"$\lambda={lam}$")
ax2.set_xlabel(r"lag $l$ (steps into the future)")
ax2.set_ylabel(r"normalized weight on $\delta_{t+l}$")
ax2.set_title(r"GAE($\lambda$) weights $(\gamma\lambda)^l$ = IIR decay")
ax2.legend(frameon=False)
ax2.grid(alpha=0.25)
ax2.set_xlim(0, 25)

fig.tight_layout()
fig.savefig("graph_DAY_107.png", dpi=120, bbox_inches="tight")
print("saved graph_DAY_107.png")
