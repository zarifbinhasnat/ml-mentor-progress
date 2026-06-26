import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
epochs = np.arange(1, 81)

def smooth(x, alpha=0.85):
    s = [x[0]]
    for v in x[1:]:
        s.append(alpha * s[-1] + (1 - alpha) * v)
    return np.array(s)

# --- Scenario 1: Healthy ---
train1 = smooth(0.9 * np.exp(-0.05 * epochs) + 0.05 + 0.02 * np.random.randn(80))
val1   = smooth(0.9 * np.exp(-0.045 * epochs) + 0.07 + 0.025 * np.random.randn(80))

# --- Scenario 2: Overfitting ---
train2 = smooth(0.9 * np.exp(-0.08 * epochs) + 0.02 + 0.01 * np.random.randn(80))
val2   = smooth(0.9 * np.exp(-0.04 * epochs) + 0.1 +
                np.maximum(0, 0.003 * (epochs - 20)) + 0.02 * np.random.randn(80))

# --- Scenario 3: Underfitting ---
train3 = smooth(0.65 * np.exp(-0.01 * epochs) + 0.55 + 0.02 * np.random.randn(80))
val3   = smooth(0.65 * np.exp(-0.009 * epochs) + 0.56 + 0.025 * np.random.randn(80))

# --- Scenario 4: LR too large (oscillations) ---
base4 = 0.5 * np.exp(-0.02 * epochs)
osc4  = 0.15 * np.sin(0.6 * epochs) * np.exp(-0.01 * epochs)
train4 = smooth(base4 + np.abs(osc4) + 0.03 * np.random.randn(80), alpha=0.4)
val4   = smooth(base4 * 1.05 + np.abs(osc4) + 0.04 * np.random.randn(80), alpha=0.4)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Day 23 — Reading Loss Curves: The Four Canonical Shapes", fontsize=13, fontweight="bold")

scenarios = [
    (axes[0][0], train1, val1, "1. Healthy — both converge together",      "#98c379"),
    (axes[0][1], train2, val2, "2. Overfitting — 'scissors' gap opens",    "#e06c75"),
    (axes[1][0], train3, val3, "3. Underfitting — both stay high and flat", "#e5c07b"),
    (axes[1][1], train4, val4, "4. LR too large — wild oscillations",       "#61afef"),
]

for ax, tr, va, title, color in scenarios:
    ax.plot(epochs, tr, color=color, linewidth=2, label="Train loss")
    ax.plot(epochs, va, color=color, linewidth=2, linestyle="--", alpha=0.7, label="Val loss")
    ax.fill_between(epochs, tr, va, alpha=0.12, color=color)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("graph_DAY_23.png", dpi=120, bbox_inches="tight")
print("graph_DAY_23.png saved")
