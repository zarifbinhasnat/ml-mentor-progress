import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("L1 Regularization & Sparsity", fontsize=15, fontweight="bold", y=0.98)

w = np.linspace(-2, 2, 400)

# --- Panel 1: Penalty functions ---
ax1 = axes[0, 0]
ax1.plot(w, np.abs(w),       color="#e74c3c", lw=2.5, label=r"L1: $|w|$")
ax1.plot(w, w**2,            color="#3498db", lw=2.5, label=r"L2: $w^2$")
ax1.axhline(0, color="gray", lw=0.8, ls="--")
ax1.axvline(0, color="gray", lw=0.8, ls="--")
ax1.set_title("Penalty Functions", fontsize=12)
ax1.set_xlabel("w")
ax1.set_ylabel("Penalty")
ax1.legend(fontsize=10)
ax1.set_ylim(-0.1, 2.2)
ax1.annotate("L1 has constant\nslope near zero", xy=(0.6, 0.6), xytext=(1.0, 0.2),
             arrowprops=dict(arrowstyle="->", color="#e74c3c"), color="#e74c3c", fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Gradients (subgradient for L1) ---
ax2 = axes[0, 1]
l1_grad = np.sign(w)
l2_grad = 2 * w
ax2.plot(w, l1_grad, color="#e74c3c", lw=2.5, label=r"L1 subgradient: $\mathrm{sign}(w)$")
ax2.plot(w, l2_grad, color="#3498db", lw=2.5, label=r"L2 gradient: $2w$")
ax2.axhline(0, color="gray", lw=0.8, ls="--")
ax2.axvline(0, color="gray", lw=0.8, ls="--")
ax2.scatter([0], [0], color="#e74c3c", s=80, zorder=5)  # subgradient discontinuity at 0
ax2.set_title("Gradients / Subgradients", fontsize=12)
ax2.set_xlabel("w")
ax2.set_ylabel("Gradient contribution")
ax2.legend(fontsize=10)
ax2.annotate("L2 gradient → 0\nas w → 0\n(never kills weight)", xy=(0.3, 0.6), xytext=(0.7, 1.5),
             arrowprops=dict(arrowstyle="->", color="#3498db"), color="#3498db", fontsize=9)
ax2.annotate("L1 constant push\n→ exactly zeros", xy=(-0.3, -1.0), xytext=(-1.8, -0.4),
             arrowprops=dict(arrowstyle="->", color="#e74c3c"), color="#e74c3c", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-2.5, 2.5)

# --- Panel 3: Soft thresholding (proximal operator of L1) ---
ax3 = axes[1, 0]
lambdas = [0.2, 0.5, 1.0]
colors_st = ["#f39c12", "#e74c3c", "#8e44ad"]
for lam, col in zip(lambdas, colors_st):
    # soft threshold: sign(w) * max(|w| - lam, 0)
    w_new = np.sign(w) * np.maximum(np.abs(w) - lam, 0)
    ax3.plot(w, w_new, color=col, lw=2, label=rf"$\lambda={lam}$")
ax3.plot(w, w, color="gray", lw=1.5, ls="--", label="Identity (no reg)")
ax3.axhline(0, color="gray", lw=0.5)
ax3.axvline(0, color="gray", lw=0.5)
ax3.set_title("Soft Thresholding (Proximal Operator of L1)", fontsize=11)
ax3.set_xlabel("w (before update)")
ax3.set_ylabel("w (after proximal step)")
ax3.legend(fontsize=9)
ax3.annotate("Dead zone:\nweights in [-λ, λ]\nare set to exactly 0", xy=(0.0, 0.0),
             xytext=(0.8, -1.4), arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
ax3.grid(True, alpha=0.3)

# --- Panel 4: 2D geometric view — constraint sets ---
ax4 = axes[1, 1]
theta = np.linspace(0, 2 * np.pi, 500)

# L2 ball (circle)
ax4.plot(np.cos(theta), np.sin(theta), color="#3498db", lw=2.5, label="L2 ball ($\|w\|_2 \leq r$)")

# L1 ball (diamond)
diamond_x = [1, 0, -1, 0, 1]
diamond_y = [0, 1, 0, -1, 0]
ax4.plot(diamond_x, diamond_y, color="#e74c3c", lw=2.5, label="L1 ball ($\|w\|_1 \leq r$)")

# Elliptical loss contours (minimum at (1.5, 0.8) — outside the constraint balls)
loss_center = np.array([1.4, 0.75])
for scale, alpha in zip([0.4, 0.7, 1.1, 1.5], [0.7, 0.55, 0.4, 0.25]):
    ellipse_x = loss_center[0] + scale * 1.2 * np.cos(theta)
    ellipse_y = loss_center[1] + scale * 0.7 * np.sin(theta)
    ax4.plot(ellipse_x, ellipse_y, color="#2ecc71", lw=1.2, alpha=alpha)

ax4.text(1.3, 0.7, "Loss\nminimum", fontsize=8, color="#27ae60", ha="center")

# Mark optimal points
# L2: closest point on circle to loss center
l2_opt = loss_center / np.linalg.norm(loss_center)
ax4.scatter(*l2_opt, color="#3498db", s=120, zorder=10, label=f"L2 optimum ({l2_opt[0]:.2f}, {l2_opt[1]:.2f})")

# L1: closest point on diamond (typically a corner → sparse)
ax4.scatter(1, 0, color="#e74c3c", s=120, zorder=10, marker="*", label="L1 optimum (1, 0) — sparse!")

ax4.set_xlim(-1.8, 2.0)
ax4.set_ylim(-1.5, 1.6)
ax4.set_aspect("equal")
ax4.axhline(0, color="gray", lw=0.5, ls="--")
ax4.axvline(0, color="gray", lw=0.5, ls="--")
ax4.set_title("Geometric View: Why L1 → Sparsity", fontsize=11)
ax4.set_xlabel("$w_1$")
ax4.set_ylabel("$w_2$")
ax4.legend(fontsize=8, loc="upper left")
ax4.annotate("Corner touch\n→ $w_2 = 0$", xy=(1, 0), xytext=(0.2, -1.1),
             arrowprops=dict(arrowstyle="->", color="#e74c3c"), color="#e74c3c", fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("graph_DAY_25.png", dpi=120, bbox_inches="tight")
print("graph_DAY_25.png saved successfully.")
