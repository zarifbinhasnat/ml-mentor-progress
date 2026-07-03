"""
Label smoothing bounds the optimal logit gap.

For a K-way softmax classifier, assume the true-class logit is z and every
other logit sits at 0 (symmetric worst case). We plot cross-entropy loss as
a function of z for (a) a one-hot target and (b) a label-smoothed target,
and mark where each curve is minimized.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

K = 10          # number of classes, e.g. a small image classifier
eps = 0.1       # smoothing factor
z = np.linspace(-2, 12, 400)

# softmax probabilities when true-class logit = z, all other logits = 0
p_true = np.exp(z) / (np.exp(z) + (K - 1))
p_other = 1.0 / (np.exp(z) + (K - 1))

# one-hot cross-entropy: only the true-class term contributes
loss_onehot = -np.log(p_true)

# label-smoothed cross-entropy: (1-eps) on true class, eps spread over the rest
loss_smooth = -(1 - eps) * np.log(p_true) - eps * np.log(p_other)

z_star_smooth = z[np.argmin(loss_smooth)]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(z, loss_onehot, label="one-hot target", color="#d62728", linewidth=2)
ax.plot(z, loss_smooth, label=f"smoothed target (ε={eps})", color="#1f77b4", linewidth=2)
ax.axvline(z_star_smooth, color="#1f77b4", linestyle="--", alpha=0.6)
ax.scatter([z_star_smooth], [loss_smooth.min()], color="#1f77b4", zorder=5)
ax.annotate(f"finite minimum\nz* ≈ {z_star_smooth:.1f}",
            xy=(z_star_smooth, loss_smooth.min()),
            xytext=(z_star_smooth + 0.6, loss_smooth.min() + 1.3),
            fontsize=9, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.7))
ax.annotate("one-hot loss keeps\nfalling as z → ∞",
            xy=(9.5, loss_onehot[np.searchsorted(z, 9.5)]),
            xytext=(6.5, 2.6),
            fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.7))

ax.set_xlabel("true-class logit gap  z  (other logits fixed at 0)")
ax.set_ylabel("cross-entropy loss")
ax.set_title("Why label smoothing bounds the optimal logit gap")
ax.legend(loc="upper right")
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig("graph_DAY_32.png", dpi=120, bbox_inches="tight")
