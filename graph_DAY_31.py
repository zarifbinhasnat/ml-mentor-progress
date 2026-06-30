import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

epochs = np.arange(1, 61)

# Synthetic training loss: monotonically decreasing, slight noise
train_loss = 1.0 * np.exp(-epochs / 18.0) + 0.02 + np.random.normal(0, 0.005, size=epochs.shape)

# Synthetic validation loss: decreases then rises after overfitting kicks in (~epoch 28)
val_base = 1.05 * np.exp(-epochs / 16.0) + 0.05
overfit_kick = 0.0009 * np.maximum(epochs - 28, 0) ** 1.4
val_loss = val_base + overfit_kick + np.random.normal(0, 0.006, size=epochs.shape)

# Early stopping: patience = 7 epochs without improvement on val_loss
best_val = np.inf
best_epoch = None
patience = 7
wait = 0
stop_epoch = epochs[-1]
for i, v in enumerate(val_loss):
    if v < best_val:
        best_val = v
        best_epoch = epochs[i]
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            stop_epoch = epochs[i]
            break

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, train_loss, label="Train loss", color="#1f77b4", linewidth=2)
ax.plot(epochs, val_loss, label="Validation loss", color="#d62728", linewidth=2)

ax.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5,
           label=f"Best checkpoint (epoch {best_epoch})")
ax.axvline(stop_epoch, color="#7f7f7f", linestyle=":", linewidth=1.5,
           label=f"Stop training (epoch {stop_epoch})")

ax.fill_betweenx([0, 1.1], best_epoch, stop_epoch, color="gray", alpha=0.08)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Early Stopping: Train vs. Validation Loss")
ax.set_ylim(0, 1.1)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25)

plt.savefig("graph_DAY_31.png", dpi=120, bbox_inches="tight")
