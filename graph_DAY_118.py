"""
Day 118 - Consolidation Phase 3: L1 vs L2 regularization geometry.

Fits the same sparse linear regression problem with an L2 (ridge) penalty
and an L1 (lasso, via subgradient) penalty, then shows the learned weight
vectors against the true sparse weights. L2 shrinks every weight a little;
L1 drives the irrelevant ones to exactly zero.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Synthetic sparse-signal regression problem (< 100 points) ---
n_samples, n_features = 60, 12
X = np.random.randn(n_samples, n_features)
w_true = np.zeros(n_features)
active = [1, 4, 7, 9]  # only 4 of 12 features actually matter
w_true[active] = np.array([2.5, -1.8, 1.2, -3.0])
noise = 0.3 * np.random.randn(n_samples)
y = X @ w_true + noise


def fit(penalty, lam=0.15, lr=0.05, epochs=2000):
    w = np.zeros(n_features)
    for _ in range(epochs):
        residual = X @ w - y
        grad = X.T @ residual / n_samples
        if penalty == "l2":
            grad += lam * w
        elif penalty == "l1":
            grad += lam * np.sign(w)
        w -= lr * grad
    return w


w_l2 = fit("l2")
w_l1 = fit("l1")

# --- Plot: true vs L2 vs L1 weight vectors, side by side ---
idx = np.arange(n_features)
width = 0.27

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(idx - width, w_true, width, label="true weights", color="#444444")
ax.bar(idx, w_l2, width, label="L2 (ridge)", color="#4C72B0")
ax.bar(idx + width, w_l1, width, label="L1 (lasso)", color="#DD8452")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(idx)
ax.set_xlabel("feature index")
ax.set_ylabel("learned weight")
ax.set_title("L2 shrinks every weight a little; L1 zeroes out the irrelevant ones")
ax.legend()
fig.tight_layout()
fig.savefig("graph_DAY_118.png", dpi=120, bbox_inches="tight")

n_l1_zero = np.sum(np.abs(w_l1) < 1e-2)
n_l2_zero = np.sum(np.abs(w_l2) < 1e-2)
print(f"Near-zero weights -> L1: {n_l1_zero}/{n_features}, L2: {n_l2_zero}/{n_features}")
