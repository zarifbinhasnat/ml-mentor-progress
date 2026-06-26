import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Elongated quadratic: L = 0.5*(theta1^2/0.02 + theta2^2/2)
# Grad: [theta1/0.02, theta2/2]
def grad_f(theta):
    return np.array([theta[0] / 0.02, theta[1] / 2.0])

def sgd_path(start, lr=0.018, steps=80):
    theta = np.array(start, dtype=float)
    path = [theta.copy()]
    for _ in range(steps):
        theta = theta - lr * grad_f(theta)
        path.append(theta.copy())
    return np.array(path)

def momentum_path(start, lr=0.018, beta=0.9, steps=80):
    theta = np.array(start, dtype=float)
    v = np.zeros(2)
    path = [theta.copy()]
    for _ in range(steps):
        g = grad_f(theta)
        v = beta * v + g
        theta = theta - lr * v
        path.append(theta.copy())
    return np.array(path)

start = [0.9, 0.9]
path_sgd = sgd_path(start, lr=0.018, steps=100)
path_mom = momentum_path(start, lr=0.018, beta=0.9, steps=100)

# Contour
t1 = np.linspace(-1.2, 1.2, 300)
t2 = np.linspace(-1.2, 1.2, 300)
T1, T2 = np.meshgrid(t1, t2)
Z = 0.5 * (T1**2 / 0.02 + T2**2 / 2.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Day 14 — Momentum: Dampening Oscillations on Ill-Conditioned Landscapes", fontsize=13, fontweight="bold")

for ax, path, title, color in [
    (axes[0], path_sgd, "SGD (no momentum) — zigzags", "#61afef"),
    (axes[1], path_mom, "SGD + Momentum (β=0.9) — smooth", "#e06c75"),
]:
    ax.contour(T1, T2, Z, levels=np.logspace(-1, 2, 15), colors="grey", linewidths=0.5, alpha=0.5)
    ax.contourf(T1, T2, Z, levels=np.logspace(-1, 2, 15), cmap="Blues", alpha=0.2)
    ax.plot(path[:, 0], path[:, 1], "o-", color=color, markersize=2.5, linewidth=1.5, label=title, zorder=4)
    ax.plot(*start, "k^", markersize=9, zorder=5, label="Start")
    ax.plot(0, 0, "r*", markersize=12, zorder=5, label="Minimum (0,0)")
    ax.set_xlabel("θ₁ (high curvature κ=100)"); ax.set_ylabel("θ₂ (low curvature κ=1)")
    ax.set_title(title)
    ax.legend(fontsize=8.5); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

plt.tight_layout()
plt.savefig("graph_DAY_14.png", dpi=120, bbox_inches="tight")
print("graph_DAY_14.png saved")
