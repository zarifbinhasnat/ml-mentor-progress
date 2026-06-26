import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_layers = 10
n_units  = 64

def propagate_variance(n_layers, n_units, init_fn, activation_fn, d_activation_fn=None, n_trials=2000):
    """Track forward activation variance and backward gradient variance."""
    act_vars, grad_norms = [], []
    for _ in range(n_trials):
        x = np.random.randn(n_units)
        zs, acts = [x], [x]
        Ws = []
        for l in range(n_layers):
            W = init_fn(n_units)
            Ws.append(W)
            z = W @ acts[-1]
            zs.append(z)
            acts.append(activation_fn(z))
        act_vars.append([np.var(a) for a in acts])
        # backward: track delta norms
        delta = np.ones(n_units)
        g_norms = [np.linalg.norm(delta)]
        for l in range(n_layers - 1, -1, -1):
            d_act = d_activation_fn(zs[l+1]) if d_activation_fn else np.ones(n_units)
            delta = (Ws[l].T @ delta) * d_act
            g_norms.insert(0, np.linalg.norm(delta))
        grad_norms.append(g_norms)
    return np.array(act_vars).mean(0), np.array(grad_norms).mean(0)

relu  = lambda z: np.maximum(0, z)
d_relu = lambda z: (z > 0).astype(float)
xavier_init = lambda n: np.random.randn(n, n) * np.sqrt(2.0 / (n + n))
he_init     = lambda n: np.random.randn(n, n) * np.sqrt(2.0 / n)

act_xavier, grad_xavier = propagate_variance(n_layers, n_units, xavier_init, relu, d_relu)
act_he,     grad_he     = propagate_variance(n_layers, n_units, he_init,     relu, d_relu)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Day 20 — He Init (ReLU Nets): Preserving Variance Through ReLU", fontsize=13, fontweight="bold")

ax = axes[0]
ax.semilogy(act_xavier, color="#e06c75", linewidth=2, label="Xavier init (under-corrects for ReLU)")
ax.semilogy(act_he,     color="#98c379", linewidth=2, label="He init (correct for ReLU) ✓")
ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":", alpha=0.5, label="Var=1 (target)")
ax.set_xlabel("Layer depth"); ax.set_ylabel("Mean activation variance (log)")
ax.set_title("Activation variance: Xavier decays, He stays flat")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xticks(range(n_layers + 1))

ax2 = axes[1]
ax2.semilogy(grad_xavier, color="#e06c75", linewidth=2, label="Xavier: gradient signal shrinks")
ax2.semilogy(grad_he,     color="#98c379", linewidth=2, label="He: gradient stays alive ✓")
ax2.axhline(1.0, color="k", linewidth=0.8, linestyle=":", alpha=0.5)
ax2.set_xlabel("Layer depth (from input=0 to output=L)")
ax2.set_ylabel("Mean ‖δ‖ (log scale)")
ax2.set_title("Backward gradient signal: He avoids vanishing")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_20.png", dpi=120, bbox_inches="tight")
print("graph_DAY_20.png saved")
