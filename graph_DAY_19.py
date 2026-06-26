import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def simulate_variance(n_layers, n_units, init_scale, activation="tanh"):
    """Track activation variance across layers for a given init scale."""
    x = np.random.randn(1000)   # batch of activations, start var=1
    variances = [np.var(x)]
    for _ in range(n_layers):
        n_in = n_units
        W = np.random.randn(n_units, n_units) * init_scale
        x = W @ np.reshape(x, (1, -1))[0, :n_units]  # simplified: 1D projection
        if activation == "tanh":
            x = np.tanh(x)
        elif activation == "linear":
            pass
        variances.append(np.var(x))
    return variances

n_layers = 10
n_units = 50

# Xavier init scale (normal): sqrt(2/(n_in + n_out)) ≈ sqrt(1/n) for square
xavier_scale = np.sqrt(2.0 / (n_units + n_units))
zero_scale   = 0.0
small_scale  = 0.01
large_scale  = 0.5

# Use linear activation for clean variance propagation demo
var_zero    = simulate_variance(n_layers, n_units, zero_scale,   activation="linear")
var_small   = simulate_variance(n_layers, n_units, small_scale,  activation="linear")
var_xavier  = simulate_variance(n_layers, n_units, xavier_scale, activation="linear")
var_large   = simulate_variance(n_layers, n_units, large_scale,  activation="linear")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Day 19 — Xavier/Glorot Init: Preserving Activation Variance", fontsize=13, fontweight="bold")

ax = axes[0]
ax.semilogy(var_zero,   "k--",  linewidth=1.5, alpha=0.7, label="Zero init (stuck)")
ax.semilogy(var_small,  color="#e06c75", linewidth=2, label=f"Too small (σ={small_scale})")
ax.semilogy(var_xavier, color="#98c379", linewidth=2.2, label=f"Xavier (σ={xavier_scale:.3f}) ✓")
ax.semilogy(var_large,  color="#e5c07b", linewidth=2, label=f"Too large (σ={large_scale})")
ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":", alpha=0.5, label="Var=1 (target)")
ax.set_xlabel("Layer depth"); ax.set_ylabel("Activation variance (log)")
ax.set_title("Activation variance through 10 linear layers")
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
ax.set_xticks(range(n_layers + 1))

# Panel 2: show the variance formula
n_range = np.arange(10, 500, 10)
xavier_var = 2.0 / (n_range + n_range)  # 2/(n_in+n_out) normal variant
xavier_std = np.sqrt(xavier_var)

ax2 = axes[1]
ax2.plot(n_range, xavier_var, color="#61afef", linewidth=2, label=r"Var(W) = $\frac{2}{n_{in}+n_{out}}$")
ax2.plot(n_range, xavier_std, color="#e06c75", linewidth=2, linestyle="--",
         label=r"Std(W) = $\sqrt{\frac{2}{n_{in}+n_{out}}}$")
ax2.set_xlabel("Layer width n (square layer, n_in=n_out=n)")
ax2.set_ylabel("Initialization scale")
ax2.set_title("Xavier scale shrinks as layers get wider")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_19.png", dpi=120, bbox_inches="tight")
print("graph_DAY_19.png saved")
