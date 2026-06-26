import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

residual = np.linspace(-3, 3, 400)
mse_loss = residual ** 2
mae_loss = np.abs(residual)
huber    = np.where(np.abs(residual) <= 1, 0.5 * residual**2, np.abs(residual) - 0.5)

# Simulated 1-D regression data
x_data = np.linspace(0, 5, 30)
y_true = 1.5 * x_data + 2.0
y_obs  = y_true + np.random.normal(0, 1.0, len(x_data))
y_obs[5]  = y_obs[5] + 6   # add an outlier
y_obs[22] = y_obs[22] - 5  # add another

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Day 4 — MSE Loss", fontsize=13, fontweight="bold")

# Panel 1: loss shapes
ax = axes[0]
ax.plot(residual, mse_loss, color="#e06c75", linewidth=2, label="MSE = r²")
ax.plot(residual, mae_loss, color="#61afef", linewidth=2, label="|r| (MAE)")
ax.plot(residual, huber,    color="#98c379", linewidth=2, label="Huber (δ=1)")
ax.set_xlabel("Residual  r = ŷ − y")
ax.set_ylabel("Loss")
ax.set_title("Loss vs residual (per sample)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.annotate("Outlier penalty\ngrows as r²", xy=(2.5, 6.25), xytext=(1.2, 7.5),
            arrowprops=dict(arrowstyle="->", color="darkred"), fontsize=8, color="darkred")

# Panel 2: MSE gradient magnitude
ax = axes[1]
ax.plot(residual, 2 * residual, color="#e06c75", linewidth=2, label=r"$\partial\mathcal{L}/\partial\hat{y} = 2r$")
ax.axhline(0, color="k", linewidth=0.6)
ax.set_xlabel("Residual  r = ŷ − y")
ax.set_ylabel("Gradient magnitude")
ax.set_title("MSE gradient — linear in residual")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 3: scatter plot with fitted line and residuals
ax = axes[2]
y_pred = np.polyval(np.polyfit(x_data, y_obs, 1), x_data)
ax.scatter(x_data, y_obs, color="#61afef", s=30, zorder=3, label="Observations")
ax.plot(x_data, y_pred, color="#e06c75", linewidth=2, label="Fitted line")
for xi, yo, yp in zip(x_data, y_obs, y_pred):
    ax.plot([xi, xi], [yo, yp], color="grey", linewidth=0.7, alpha=0.6)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Residuals on toy data (with outliers)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_4.png", dpi=120, bbox_inches="tight")
print("graph_DAY_4.png saved")
