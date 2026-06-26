import numpy as np
import matplotlib.pyplot as plt

# Simulate Adam moment estimates and bias correction on a synthetic gradient stream
np.random.seed(7)
steps = 100
g = np.random.randn(steps) * 0.5 + 0.3     # gradient stream with nonzero mean

beta1, beta2, eps = 0.9, 0.999, 1e-8
lr = 0.01

m, v = 0.0, 0.0
m_hat_hist, v_hat_hist = [], []
raw_m_hist, raw_v_hist = [], []
eff_lr_hist = []
theta = 0.5; theta_hist = [theta]
theta_sgd = 0.5; theta_sgd_hist = [theta_sgd]

for t, gt in enumerate(g, 1):
    m = beta1 * m + (1 - beta1) * gt
    v = beta2 * v + (1 - beta2) * gt**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    raw_m_hist.append(m); raw_v_hist.append(v)
    m_hat_hist.append(m_hat); v_hat_hist.append(v_hat)
    eff_lr = lr / (np.sqrt(v_hat) + eps)
    eff_lr_hist.append(eff_lr)
    theta = theta - eff_lr * m_hat
    theta_hist.append(theta)
    theta_sgd = theta_sgd - lr * gt
    theta_sgd_hist.append(theta_sgd)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Day 16 — Adam & AdamW: Bias Correction and Adaptive Steps", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(raw_m_hist,   color="#e06c75", linewidth=1.5, label="Raw m_t (biased)")
ax.plot(m_hat_hist,   color="#98c379", linewidth=1.5, label="m̂_t (bias-corrected)")
ax.axhline(np.mean(g), color="k", linewidth=1, linestyle="--", label=f"True mean = {np.mean(g):.2f}")
ax.set_xlabel("Step"); ax.set_ylabel("1st moment estimate")
ax.set_title("Bias correction for 1st moment (m)")
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(eff_lr_hist, color="#c678dd", linewidth=2, label="Adam eff. LR")
ax2.axhline(lr, color="k", linewidth=1, linestyle="--", label=f"SGD LR = {lr}")
ax2.set_xlabel("Step"); ax2.set_ylabel("Effective learning rate")
ax2.set_title("Per-parameter adaptive LR over time")
ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3)

ax3 = axes[2]
ax3.plot(theta_hist, color="#e06c75", linewidth=2, label="Adam θ trajectory")
ax3.plot(theta_sgd_hist, color="#61afef", linewidth=2, linestyle="--", label="SGD θ trajectory")
ax3.set_xlabel("Step"); ax3.set_ylabel("θ")
ax3.set_title("Parameter update trajectory")
ax3.legend(fontsize=8.5); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_16.png", dpi=120, bbox_inches="tight")
print("graph_DAY_16.png saved")
