import numpy as np
import matplotlib.pyplot as plt

# Classic bias-variance tradeoff curve (polynomial fitting on noisy data)
np.random.seed(42)

def true_fn(x): return np.sin(2 * np.pi * x)

N_train = 25
x_train = np.sort(np.random.rand(N_train))
y_train = true_fn(x_train) + 0.3 * np.random.randn(N_train)

x_test = np.linspace(0, 1, 200)
y_test = true_fn(x_test)

degrees = list(range(1, 16))
bias2_list, var_list, total_list = [], [], []
n_trials = 50

for d in degrees:
    preds = []
    for _ in range(n_trials):
        idx = np.random.choice(N_train, N_train, replace=True)
        x_b = x_train[idx]; y_b = y_train[idx]
        coef = np.polyfit(x_b, y_b, d)
        p = np.polyval(coef, x_test)
        preds.append(p)
    preds = np.array(preds)
    mean_pred = preds.mean(0)
    bias2 = np.mean((mean_pred - y_test)**2)
    var   = np.mean(preds.var(0))
    bias2_list.append(bias2)
    var_list.append(var)
    total_list.append(bias2 + var + 0.09)   # add irreducible noise floor

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Day 22 — Bias–Variance Tradeoff", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(degrees, bias2_list, "o-", color="#e06c75", linewidth=2, label="Bias²  (underfitting error)")
ax.plot(degrees, var_list,   "s-", color="#61afef", linewidth=2, label="Variance (overfitting error)")
ax.plot(degrees, total_list, "^-", color="#98c379", linewidth=2.5, label="Total test error (≈Bias²+Var+σ²)")
ax.axhline(0.09, color="k", linewidth=0.8, linestyle=":", alpha=0.5, label="Irreducible noise σ²=0.09")
best_d = degrees[np.argmin(total_list)]
ax.axvline(best_d, color="grey", linewidth=1, linestyle="--", label=f"Sweet spot (degree={best_d})")
ax.set_xlabel("Model complexity (polynomial degree)")
ax.set_ylabel("Error")
ax.set_title("U-shaped total error — the classic tradeoff curve")
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

ax2 = axes[1]
for d, color, label in [(1, "#e06c75", "Degree 1 — high bias (underfit)"),
                         (best_d, "#98c379", f"Degree {best_d} — sweet spot"),
                         (14, "#61afef", "Degree 14 — high variance (overfit)")]:
    coef = np.polyfit(x_train, y_train, d)
    ax2.plot(x_test, np.polyval(coef, x_test), linewidth=2, label=label, color=color)
ax2.scatter(x_train, y_train, color="k", s=20, zorder=5, alpha=0.6, label="Training data")
ax2.plot(x_test, y_test, "k--", linewidth=1.5, alpha=0.5, label="True function sin(2πx)")
ax2.set_xlabel("x"); ax2.set_ylabel("y")
ax2.set_title("Three models: underfit, just right, overfit")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig("graph_DAY_22.png", dpi=120, bbox_inches="tight")
print("graph_DAY_22.png saved")
