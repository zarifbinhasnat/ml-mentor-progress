import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Day 7 — Chain Rule Refresher", fontsize=13, fontweight="bold")

# Panel 1: visualise y = sin(x²) — composition with annotated local gradients
x_range = np.linspace(-2, 2, 400)
u_range = x_range ** 2
y_range = np.sin(u_range)
du_dx = 2 * x_range
dy_du = np.cos(u_range)
dy_dx = dy_du * du_dx    # chain rule

ax = axes[0]
ax.plot(x_range, y_range, color="#61afef", linewidth=2, label=r"$y = \sin(x^2)$")
ax.plot(x_range, dy_dx,  color="#e06c75", linewidth=1.5, linestyle="--", label=r"$dy/dx = \cos(x^2)\cdot 2x$")
ax.axhline(0, color="k", linewidth=0.6)
ax.axvline(0, color="k", linewidth=0.6)

# annotate one chain-rule point
x0 = 1.0; u0 = x0**2; y0 = np.sin(u0)
dy_du0 = np.cos(u0); du_dx0 = 2*x0; chain0 = dy_du0 * du_dx0
ax.plot(x0, y0, "ro", markersize=7, zorder=5)
ax.annotate(f"x={x0}: dy/du={dy_du0:.2f},\ndu/dx={du_dx0:.1f} → dy/dx={chain0:.2f}",
            xy=(x0, y0), xytext=(0.2, -0.8),
            arrowprops=dict(arrowstyle="->", color="darkred"), fontsize=8, color="darkred")
ax.set_xlabel("x"); ax.set_ylabel("value")
ax.set_title(r"$y = \sin(x^2)$ and its chain-rule derivative")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 2: multi-path sum rule — parameter w feeds two outputs u1=w*a and u2=w*b
# total dL/dw = (dL/du1)*a + (dL/du2)*b
# sweep a and show how summing vs averaging changes the result
np.random.seed(7)
steps = np.arange(1, 51)
a_vals = np.random.randn(50) + 1.0    # upstream gradient via path 1
b_vals = np.random.randn(50) + 0.5    # upstream gradient via path 2

dl_du1 = np.random.randn(50) * 0.8
dl_du2 = np.random.randn(50) * 0.6

true_grad  = dl_du1 * a_vals + dl_du2 * b_vals   # correct: sum
wrong_grad = (dl_du1 * a_vals + dl_du2 * b_vals) / 2   # wrong: average

ax2 = axes[1]
ax2.plot(steps, true_grad,  color="#98c379", linewidth=1.8, label=r"Correct: $\sum$ over paths")
ax2.plot(steps, wrong_grad, color="#e06c75", linewidth=1.8, linestyle="--", label=r"Bug: $\frac{1}{2}\sum$ (average)")
ax2.axhline(0, color="k", linewidth=0.6)
ax2.fill_between(steps, true_grad, wrong_grad, alpha=0.2, color="grey", label="Difference (silent LR shrink)")
ax2.set_xlabel("Training step")
ax2.set_ylabel("Gradient for parameter w")
ax2.set_title("Multi-path sum rule: sum ≠ average")
ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_7.png", dpi=120, bbox_inches="tight")
print("graph_DAY_7.png saved")
