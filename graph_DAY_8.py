import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

L = 6
n_units = [10, 8, 8, 8, 8, 4, 1]
W = [np.random.randn(n_units[l+1], n_units[l]) * 0.5 for l in range(L)]
b = [np.zeros(n_units[l+1]) for l in range(L)]

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def d_sigmoid(z): s = sigmoid(z); return s * (1 - s)

x = np.random.randn(n_units[0])
y_target = np.array([1.0])

zs, acts = [], [x]
for l in range(L):
    z = W[l] @ acts[-1] + b[l]
    zs.append(z)
    acts.append(sigmoid(z))

dL_da = acts[-1] - y_target
delta = dL_da * d_sigmoid(zs[-1])
delta_norms = [np.linalg.norm(delta)]
for l in range(L-2, -1, -1):
    delta = (W[l+1].T @ delta) * d_sigmoid(zs[l])
    delta_norms.insert(0, np.linalg.norm(delta))

def relu(z): return np.maximum(0, z)
def d_relu(z): return (z > 0).astype(float)

acts_r = [x]; zs_r = []
for l in range(L):
    z = W[l] @ acts_r[-1] + b[l]
    zs_r.append(z)
    acts_r.append(relu(z))

dL_da_r = acts_r[-1] - y_target
delta_r = dL_da_r * d_relu(zs_r[-1])
delta_norms_r = [np.linalg.norm(delta_r)]
for l in range(L-2, -1, -1):
    delta_r = (W[l+1].T @ delta_r) * d_relu(zs_r[l])
    delta_norms_r.insert(0, np.linalg.norm(delta_r))

# Replace any zeros with a small floor so log scale works
delta_norms   = [max(v, 1e-12) for v in delta_norms]
delta_norms_r = [max(v, 1e-12) for v in delta_norms_r]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Day 8 — Backprop I: Credit-Assignment via Error Signal δ", fontsize=13, fontweight="bold")

layer_labels = [f"L{i+1}" for i in range(L)]

ax = axes[0]
ax.bar(range(L), delta_norms, color=["#e06c75" if v < 0.01 else "#61afef" for v in delta_norms], edgecolor="k", linewidth=0.7)
ax.set_xticks(range(L)); ax.set_xticklabels(layer_labels, fontsize=9)
ax.set_ylabel("‖δ‖  (log scale)")
ax.set_title("Sigmoid net: δ vanishes toward input")
ax.set_yscale("log"); ax.grid(True, alpha=0.3, axis="y")
ax.axhline(1e-2, color="darkred", linestyle=":", linewidth=1.2, label="1e-2 floor")
ax.legend(fontsize=8)

ax2 = axes[1]
ax2.bar(range(L), delta_norms_r,
        color=["#98c379" if v >= 1e-4 else "#e06c75" for v in delta_norms_r],
        edgecolor="k", linewidth=0.7)
ax2.set_xticks(range(L)); ax2.set_xticklabels(layer_labels, fontsize=9)
ax2.set_ylabel("‖δ‖  (log scale)")
ax2.set_title("ReLU net: δ stays alive (where neurons fire)")
ax2.set_yscale("log"); ax2.grid(True, alpha=0.3, axis="y")
ax2.axhline(1e-2, color="darkred", linestyle=":", linewidth=1.2, label="1e-2 floor")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_8.png", dpi=120, bbox_inches="tight")
print("graph_DAY_8.png saved")
