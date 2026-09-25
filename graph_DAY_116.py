"""
Day 116 -- Consolidation Day: Phase 1 (Concepts 1-12)

Fits a 1-parameter linear regression (concept 1: forward pass, concept 4: MSE
loss) with plain gradient descent (concept 10) under three learning rates, to
make concept 11's claim literal: the learning rate is not a knob you tune for
comfort, it decides whether the same update rule converges, crawls, or blows up.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- tiny synthetic dataset: y = 3x + 2 + noise, < 100 points ---
x = np.linspace(-1, 1, 40)
y = 3 * x + 2 + np.random.normal(0, 0.15, size=x.shape)


def gradient_descent(lr, iters=40):
    w, b = 0.0, 0.0
    losses = []
    for _ in range(iters):
        y_hat = w * x + b
        err = y_hat - y
        loss = np.mean(err ** 2)
        losses.append(loss)
        if not np.isfinite(loss) or loss > 1e6:
            losses += [np.nan] * (iters - len(losses))
            break
        dw = np.mean(2 * err * x)
        db = np.mean(2 * err)
        w -= lr * dw
        b -= lr * db
    return np.array(losses)


runs = {
    "too small (lr=0.05) -- crawls": gradient_descent(0.05),
    "good (lr=0.6) -- converges fast": gradient_descent(0.6),
    "too large (lr=1.55) -- diverges": gradient_descent(1.55),
}

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for label, losses in runs.items():
    ax.plot(losses, marker="o", markersize=3, linewidth=1.6, label=label)

ax.set_yscale("symlog")
ax.set_xlabel("iteration")
ax.set_ylabel("MSE loss (symlog scale)")
ax.set_title("Same gradient descent update, three learning rates")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.savefig("graph_DAY_116.png", dpi=120, bbox_inches="tight")
