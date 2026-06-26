import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n0, n1, n2 = 4, 8, 1
W1 = np.random.randn(n1, n0) * 0.1
b1 = np.zeros(n1)
W2 = np.random.randn(n2, n1) * 0.1
b2 = np.zeros(n2)
x  = np.random.randn(n0)
y  = np.array([1.0])

# Forward
z1   = W1 @ x + b1
a1   = np.maximum(0, z1)
z2   = W2 @ a1 + b2
yhat = z2

# Backprop (4-equation derivation)
d2  = yhat - y
gW2 = d2[:, None] @ a1[None, :]
gb2 = d2
d1  = (W2.T @ d2) * (z1 > 0)
gW1 = d1[:, None] @ x[None, :]
gb1 = d1

# PyTorch-style finite-difference check
eps = 1e-5
def fwd(W1_, b1_, W2_, b2_):
    a1_ = np.maximum(0, W1_ @ x + b1_)
    yhat_ = W2_ @ a1_ + b2_
    return 0.5 * np.sum((yhat_ - y)**2)

L0 = fwd(W1, b1, W2, b2)
fd_gW1 = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        dW = np.zeros_like(W1); dW[i, j] = eps
        fd_gW1[i, j] = (fwd(W1+dW, b1, W2, b2) - fwd(W1-dW, b1, W2, b2)) / (2*eps)

# Visualise: our gradient vs finite-diff for W1
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Day 9 — Backprop II: Full Math (One Hidden Layer)", fontsize=13, fontweight="bold")

ax = axes[0]
vmax = max(np.abs(gW1).max(), np.abs(fd_gW1).max())
im = ax.imshow(gW1, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_title("∂L/∂W¹  (4-equation backprop)")
ax.set_xlabel("input dim"); ax.set_ylabel("hidden unit")
plt.colorbar(im, ax=ax, fraction=0.046)

ax2 = axes[1]
im2 = ax2.imshow(fd_gW1, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
ax2.set_title("∂L/∂W¹  (finite-difference ground truth)")
ax2.set_xlabel("input dim"); ax2.set_ylabel("hidden unit")
plt.colorbar(im2, ax=ax2, fraction=0.046)

ax3 = axes[2]
diff = np.abs(gW1 - fd_gW1)
ax3.bar(range(diff.size), diff.ravel(), color="#61afef", edgecolor="k", linewidth=0.5)
ax3.set_title(f"Absolute error  (max={diff.max():.2e})")
ax3.set_xlabel("W¹ element index")
ax3.set_ylabel("|our gradient − FD gradient|")
ax3.set_yscale("log"); ax3.grid(True, alpha=0.3, axis="y")
ax3.axhline(1e-10, color="red", linewidth=1, linestyle="--", label="1e-10 line")
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_9.png", dpi=120, bbox_inches="tight")
print("graph_DAY_9.png saved")
