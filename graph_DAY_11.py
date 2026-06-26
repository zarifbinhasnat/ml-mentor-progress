import numpy as np
import matplotlib.pyplot as plt

# 1D quadratic loss: L(theta) = 0.5 * theta^2  (lambda = 1, optimum at 0)
# Exact GD: theta_{t+1} = (1 - lr) * theta_t
def gd_quadratic(lr, steps=60, theta0=2.0):
    theta = theta0
    path = [theta]
    for _ in range(steps):
        theta = theta - lr * theta   # grad = theta for L = 0.5*theta^2
        path.append(theta)
    return np.array(path)

steps = 60
cases = [
    (0.05, "#98c379", "η=0.05 (slow but stable)"),
    (0.5,  "#61afef", "η=0.5  (good)"),
    (1.5,  "#e5c07b", "η=1.5  (near boundary, oscillates)"),
    (2.2,  "#e06c75", "η=2.2  (> 2/λ, DIVERGES)"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Day 11 — The Learning Rate: Convergence & Divergence", fontsize=13, fontweight="bold")

ax = axes[0]
for lr, color, label in cases:
    path = gd_quadratic(lr, steps)
    path_clipped = np.clip(path, -20, 20)
    ax.plot(range(steps + 1), path_clipped, color=color, linewidth=1.8, label=label)
ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
ax.set_xlabel("Step"); ax.set_ylabel("θ (log-scaled abs)")
ax.set_yscale("symlog", linthresh=1e-4)
ax.set_title("θ trajectories (symlog scale)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax2 = axes[1]
for lr, color, label in cases:
    path = gd_quadratic(lr, steps)
    losses = 0.5 * path**2
    losses_clipped = np.clip(losses, 1e-16, 1e6)
    ax2.plot(range(steps + 1), losses_clipped, color=color, linewidth=1.8, label=label)
ax2.set_yscale("log")
ax2.set_xlabel("Step"); ax2.set_ylabel("Loss  L(θ) = 0.5 θ²")
ax2.set_title("Loss curves — stability boundary at η = 2/λ = 2")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# Annotate the stability boundary
ax2.axvline(0, color="k", linewidth=0)
ax2.text(35, 1e3, "Diverges when η > 2/λ", fontsize=9, color="#e06c75", fontweight="bold")

plt.tight_layout()
plt.savefig("graph_DAY_11.png", dpi=120, bbox_inches="tight")
print("graph_DAY_11.png saved")
