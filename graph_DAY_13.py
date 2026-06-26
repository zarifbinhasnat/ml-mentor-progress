import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Gradient variance vs batch size  (1/B law)
batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
sigma2 = 1.0   # single-sample gradient variance
variance = sigma2 / batch_sizes
std_dev  = np.sqrt(variance)

# Simulate actual gradient estimates
N_true = 10000
true_grads = np.random.randn(N_true)     # "population" of per-sample gradients

def estimate_variance(B, trials=500):
    means = [true_grads[np.random.choice(N_true, B, replace=True)].mean() for _ in range(trials)]
    return np.var(means)

measured_var = np.array([estimate_variance(B, trials=300) for B in batch_sizes])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Day 13 — Epochs, Batches, Iterations: Gradient Noise vs Batch Size", fontsize=13, fontweight="bold")

ax = axes[0]
ax.loglog(batch_sizes, variance,    "o--", color="#e06c75", linewidth=2, markersize=6, label=r"Theory: $\sigma^2/B$")
ax.loglog(batch_sizes, measured_var,"s-",  color="#61afef", linewidth=2, markersize=6, label="Measured variance")
ax.set_xlabel("Batch size B  (log)"); ax.set_ylabel("Gradient variance (log)")
ax.set_title(r"Variance $\propto 1/B$ — doubling B only gives $\sqrt{2}$ noise reduction")
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)

# Panel 2: steps per epoch and total steps vs batch size, for N=10000, E=50
N, E = 10000, 50
steps_per_epoch = np.ceil(N / batch_sizes).astype(int)
total_steps = E * steps_per_epoch

ax2 = axes[1]
ax2c = ax2.twinx()
ax2.semilogx(batch_sizes, steps_per_epoch, "o-", color="#98c379", linewidth=2, markersize=6, label="Steps/epoch")
ax2c.semilogx(batch_sizes, total_steps,   "s--", color="#c678dd", linewidth=2, markersize=6, label="Total steps (50 epochs)")
ax2.set_xlabel("Batch size B  (log)")
ax2.set_ylabel("Steps per epoch", color="#98c379")
ax2c.set_ylabel("Total gradient steps", color="#c678dd")
ax2.set_title(f"N={N}, E={E}: fewer but cleaner steps as B grows")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2c.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_13.png", dpi=120, bbox_inches="tight")
print("graph_DAY_13.png saved")
