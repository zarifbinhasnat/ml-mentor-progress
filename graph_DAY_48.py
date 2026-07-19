"""
Day 48 — Why RNNs Forget
Shows the two compounding causes of vanishing gradients in a real (non-scalar,
non-linear) RNN: (1) the spectral radius of W_hh scaling the Jacobian product
step by step, and (2) tanh saturation, whose derivative is <=1 everywhere and
often much smaller -- so even a "safe" W_hh doesn't save you once activations
saturate.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(47)

hidden_size = 32
T = 40

fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# --- Top: simulated ||gradient|| through T steps for different spectral radii ---
ax = axes[0]
spectral_radii = [0.6, 0.95, 1.0, 1.5]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for rho, c in zip(spectral_radii, colors):
    W = np.random.randn(hidden_size, hidden_size)
    W *= rho / max(abs(np.linalg.eigvals(W)))  # rescale so spectral radius is exactly rho

    grad = np.random.randn(hidden_size)
    grad /= np.linalg.norm(grad)
    norms_with_saturation = [1.0]
    norms_linear_only = [1.0]
    g_sat = grad.copy()
    g_lin = grad.copy()
    for t in range(T):
        # linear-only: just the recurrent weight matrix, no activation nonlinearity
        g_lin = W.T @ g_lin
        norms_linear_only.append(np.linalg.norm(g_lin))

        # with saturation: multiply by a representative tanh'(x) diagonal, mildly
        # saturated activations (|x| ~ 1.5 typical of a trained RNN's steady state)
        tanh_prime = 1 - np.tanh(1.5) ** 2  # ~0.19, a fixed representative saturation level
        g_sat = tanh_prime * (W.T @ g_sat)
        norms_with_saturation.append(np.linalg.norm(g_sat))

    ax.semilogy(norms_with_saturation, color=c, label=f"spectral radius={rho}")

ax.set_xlabel("steps backward through BPTT")
ax.set_ylabel("||gradient|| (log scale)")
ax.set_title("Gradient norm with tanh saturation folded in: even rho=1.5 can still vanish")
ax.legend(fontsize=8)

# --- Bottom: tanh derivative -- bounded by 1, shrinks fast away from 0 ---
ax2 = axes[1]
x = np.linspace(-4, 4, 400)
tanh_deriv = 1 - np.tanh(x) ** 2
ax2.plot(x, tanh_deriv, color="#9467bd")
ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.7, label="max possible value (x=0)")
ax2.fill_between(x, 0, tanh_deriv, alpha=0.15, color="#9467bd")
ax2.set_xlabel("pre-activation x")
ax2.set_ylabel(r"$\tanh'(x) = 1-\tanh^2(x)$")
ax2.set_title("Every timestep multiplies by this factor too -- it is NEVER > 1")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_48.png", dpi=120, bbox_inches="tight")
