import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Narrow valley: L = 0.5*(100*t1^2 + t2^2), optimum at origin
def loss_f(theta): return 0.5 * (100*theta[0]**2 + theta[1]**2)
def grad_f(theta): return np.array([100*theta[0], theta[1]])

def run_momentum(start, lr, beta, steps=120, nesterov=False):
    theta = np.array(start, dtype=float)
    v = np.zeros(2)
    path = [theta.copy()]
    losses = [loss_f(theta)]
    for _ in range(steps):
        if nesterov:
            theta_look = theta - beta * v
            g = grad_f(theta_look)
        else:
            g = grad_f(theta)
        v = beta * v + lr * g
        theta = theta - v
        path.append(theta.copy())
        losses.append(loss_f(theta))
    return np.array(path), np.array(losses)

start = [0.8, 0.8]
lr, beta = 0.008, 0.9

path_mom, loss_mom = run_momentum(start, lr, beta, steps=120, nesterov=False)
path_nav, loss_nav = run_momentum(start, lr, beta, steps=120, nesterov=True)

# Contour
t1 = np.linspace(-1, 1, 300)
t2 = np.linspace(-1, 1, 300)
T1, T2 = np.meshgrid(t1, t2)
Z = 0.5 * (100*T1**2 + T2**2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Day 15 — Nesterov Accelerated Gradient vs Standard Momentum", fontsize=13, fontweight="bold")

ax = axes[0]
ax.contour(T1, T2, Z, levels=np.logspace(-1, 2, 15), colors="grey", linewidths=0.5, alpha=0.5)
ax.contourf(T1, T2, Z, levels=np.logspace(-1, 2, 15), cmap="Blues", alpha=0.2)
ax.plot(path_mom[:, 0], path_mom[:, 1], "o-", color="#e06c75", markersize=2, linewidth=1.4,
        label="Momentum (β=0.9)", alpha=0.85)
ax.plot(path_nav[:, 0], path_nav[:, 1], "o-", color="#98c379", markersize=2, linewidth=1.4,
        label="Nesterov (β=0.9)", alpha=0.85)
ax.plot(*start, "k^", markersize=8, zorder=5, label="Start")
ax.plot(0, 0, "r*", markersize=10, zorder=5, label="Minimum")
ax.set_xlabel("θ₁"); ax.set_ylabel("θ₂")
ax.set_title("Trajectories on narrow valley (κ=100)")
ax.legend(fontsize=8.5); ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9)

ax2 = axes[1]
ax2.semilogy(loss_mom, color="#e06c75", linewidth=2, label="Momentum loss")
ax2.semilogy(loss_nav, color="#98c379", linewidth=2, label="Nesterov loss")
ax2.set_xlabel("Step"); ax2.set_ylabel("Loss (log)")
ax2.set_title("Convergence: Nesterov overshoots less, converges earlier")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.annotate("Less overshoot\n→ faster convergence", xy=(40, loss_nav[40]),
             xytext=(55, loss_nav[20]),
             arrowprops=dict(arrowstyle="->", color="darkgreen"), fontsize=8, color="darkgreen")

plt.tight_layout()
plt.savefig("graph_DAY_15.png", dpi=120, bbox_inches="tight")
print("graph_DAY_15.png saved")
