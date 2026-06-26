import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_layers = 15
n_units  = 32

def grad_norms_by_layer(n_layers, n_units, init_scale, activation_deriv_max):
    """Approximate gradient norm at each layer using the chain-rule product."""
    # Each layer multiplies the gradient by W * diag(f')
    # Approximate the spectral norm as init_scale * sqrt(n_units) * activation_deriv_max
    single_factor = init_scale * np.sqrt(n_units) * activation_deriv_max
    # Gradient norm at layer l (from output) ≈ factor^(n_layers - l)
    norms = [single_factor ** k for k in range(n_layers, -1, -1)]
    return norms

# Sigmoid: max derivative 0.25, Xavier-like scale sqrt(1/n)
norms_sigmoid = grad_norms_by_layer(n_layers, n_units,
                                     init_scale=np.sqrt(1.0/n_units),
                                     activation_deriv_max=0.25)
# ReLU: max derivative 1.0, He scale sqrt(2/n)
norms_relu   = grad_norms_by_layer(n_layers, n_units,
                                    init_scale=np.sqrt(2.0/n_units),
                                    activation_deriv_max=1.0)

layers = list(range(n_layers + 1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Day 21 — Vanishing Gradients: Sigmoid vs ReLU Across 15 Layers", fontsize=13, fontweight="bold")

ax = axes[0]
ax.semilogy(layers, norms_sigmoid, "o-", color="#e06c75", linewidth=2, markersize=5,
            label="Sigmoid (max σ'=0.25, Xavier)")
ax.semilogy(layers, norms_relu,    "s-", color="#98c379", linewidth=2, markersize=5,
            label="ReLU (max d=1.0, He)")
ax.axhline(1e-6, color="grey", linewidth=0.8, linestyle=":", label="Effectively zero (1e-6)")
ax.set_xlabel("Layer (0=input, 15=output)")
ax.set_ylabel("‖∂L/∂W^(l)‖  (log scale)")
ax.set_title("Gradient magnitude across layers")
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

ax2 = axes[1]
# Show the individual multiplied factors
factors_sig = [np.sqrt(1.0/n_units) * 0.25] * n_layers
factors_relu = [np.sqrt(2.0/n_units) * 1.0] * n_layers
ax2.bar(np.arange(n_layers) - 0.2, factors_sig,  width=0.35, color="#e06c75",
        alpha=0.8, label="Per-layer factor (sigmoid + Xavier)")
ax2.bar(np.arange(n_layers) + 0.2, factors_relu, width=0.35, color="#98c379",
        alpha=0.8, label="Per-layer factor (ReLU + He)")
ax2.axhline(1.0, color="k", linewidth=1, linestyle="--", label="Factor=1.0 (no shrink)")
ax2.set_xlabel("Layer index")
ax2.set_ylabel("‖W^(l)‖ · |f'|  per layer")
ax2.set_title("Per-layer multiplier: <1 → vanishing, >1 → exploding")
ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("graph_DAY_21.png", dpi=120, bbox_inches="tight")
print("graph_DAY_21.png saved")
