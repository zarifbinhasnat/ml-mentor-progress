import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Loss surface: elongated quadratic (ill-conditioned)
def loss(theta):
    return 0.5 * (theta[0]**2 / 0.05 + theta[1]**2 / 2.0)

def grad_loss(theta):
    return np.array([theta[0] / 0.05, theta[1] / 2.0])

def run_gd(start, lr, steps, noise_std=0.0):
    theta = np.array(start, dtype=float)
    path = [theta.copy()]
    for _ in range(steps):
        g = grad_loss(theta) + np.random.randn(2) * noise_std
        theta = theta - lr * g
        path.append(theta.copy())
    return np.array(path)

start = [1.5, 1.5]
path_batch = run_gd(start, lr=0.04, steps=40, noise_std=0.0)
path_sgd   = run_gd(start, lr=0.04, steps=40, noise_std=3.0)
path_mini  = run_gd(start, lr=0.04, steps=40, noise_std=0.7)

# Contour grid
t1 = np.linspace(-2, 2, 200)
t2 = np.linspace(-2, 2, 200)
T1, T2 = np.meshgrid(t1, t2)
Z = 0.5 * (T1**2 / 0.05 + T2**2 / 2.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Day 10 — Gradient Descent: Batch vs SGD vs Mini-batch", fontsize=13, fontweight="bold")

for ax, (path, label, color) in zip(
    axes,
    [(path_batch, "Batch GD (exact gradient)", "#61afef"),
     (path_mini,  "Mini-batch GD (low noise)", "#98c379")]
):
    ax.contour(T1, T2, Z, levels=np.logspace(-1, 2, 18), colors="grey", linewidths=0.6, alpha=0.6)
    ax.contourf(T1, T2, Z, levels=np.logspace(-1, 2, 18), cmap="Blues", alpha=0.25)
    ax.plot(path[:, 0], path[:, 1], "o-", color=color, markersize=3, linewidth=1.5, label=label, zorder=4)
    ax.plot(*start, "k^", markersize=8, zorder=5, label="Start")
    ax.plot(0, 0, "r*", markersize=12, zorder=5, label="Minimum")
    ax.set_xlabel("θ₁  (high curvature)"); ax.set_ylabel("θ₂  (low curvature)")
    ax.set_title(label)
    ax.legend(fontsize=8.5); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

# Also show SGD path on right panel
ax2 = axes[1]
ax2.plot(path_sgd[:, 0], path_sgd[:, 1], "o--", color="#e06c75", markersize=2, linewidth=1,
         label="SGD (high noise)", alpha=0.8, zorder=3)
ax2.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig("graph_DAY_10.png", dpi=120, bbox_inches="tight")
print("graph_DAY_10.png saved")
